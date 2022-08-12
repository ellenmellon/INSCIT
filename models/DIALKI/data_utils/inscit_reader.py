import os
import sys
import math
import json
import pickle
import logging
import concurrent.futures
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from typing import List

import torch

from .data_class import ReaderSample, ReaderPassage, SpanPrediction
from .utils import get_word_idxs
from config import TOKENS, AGENT, USER


logger = logging.getLogger()


class InscitReader:
    def __init__(self, args, input_dir, output_dir, tokenizer, max_seq_len,
                 max_history_len, max_num_spans_per_passage,
                 num_sample_per_file=1000):

        assert max_history_len < max_seq_len, \
            f'max_history_len {max_history_len} must shorter than max_seq_len {max_seq_len}'

        self.args = args
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_history_len = max_history_len
        self.max_num_spans_per_passage = max_num_spans_per_passage
        self.num_sample_per_file = num_sample_per_file

        self.dont_mask_words = {
            tokenizer.cls_token,
            tokenizer.sep_token,
        }
        self.dont_mask_words.update(set(TOKENS))
        self.party_tokens = {AGENT, USER}


    @staticmethod
    def load_data(split, input_dir):
        input_path = os.path.join(input_dir, f'{split}.json') 
        with open(input_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        return samples


    def convert_json_to_finetune_pkl(self, split):
        self.convert_json_to_pkl(split, self.preprocess_chunk_for_finetune)


    def convert_json_to_pkl(self, split, callback):
        output_dir = os.path.join(self.output_dir, split)
        os.makedirs(output_dir, exist_ok=True)

        # NOTE
        # do not modify global variable to prevent from errors
        # read-only
        global global_samples
        global_samples = InscitReader.load_data(split.replace('_span',''), self.input_dir)

        chunks = []
        for i in range(math.ceil(len(global_samples) / self.num_sample_per_file)):
            chunks.append((i, i*self.num_sample_per_file, (i+1)*self.num_sample_per_file))

        with concurrent.futures.ProcessPoolExecutor() as executor:
            feature2chunk = {executor.submit(callback, split, chunk): chunk for chunk in chunks}
            iterator = tqdm(
                concurrent.futures.as_completed(feature2chunk),
                total=len(chunks),
                desc=f'Preprocess {split:>5s} data from {self.input_dir}:',
            )

            finish = 0
            no_positive_passages = 0
            for feature in iterator:
                chunk_idx, _, _ = feature2chunk[feature]
                try:
                    npp = feature.result()
                    iterator.set_description(f'Preprocess {split:>5s} data from {self.input_dir}: chunk {chunk_idx:>5d} finished!')
                    finish += 1
                    no_positive_passages += npp
                except Exception as e:
                    sys.exit(f'[Error]: {e}. {feature.result()}')

        logger.info(f'# of samples = {len(global_samples)}')
        logger.info(f'no positive_passages = {no_positive_passages}')
        logger.info(f'lost answer % = {no_positive_passages / len(global_samples) * 100:.2f}')
        return len(chunks)


    def preprocess_chunk_for_finetune(self, split, chunk):
        chunk_idx, start, end = chunk
        results = []
        no_positive_passages = 0
        is_train = True if split == 'train' else False
        for sample in global_samples[start:end]:
            sample = self.preprocess_sample(sample, is_train)
            if sample is None:
                no_positive_passages += 1
                continue
            results.append(sample)
    
        output_path = os.path.join(self.output_dir, f'{split}/{chunk_idx}.pkl')
        with open(output_path, mode='wb') as f:
            pickle.dump(results, f)
    
        return no_positive_passages
    

    def preprocess_sample(self, sample, is_train=True):
        q = sample["question"]
        q_type = sample["question_type"]
    
        positive_passages, negative_passages = InscitReader.select_passages(sample, is_train)
        # create concatenated sequence ids for each passage and adjust answer spans
        positive_passages = [
            self.create_passage(s, q, q_type) for s in positive_passages
        ]

        negative_passages = [
            self.create_passage(s, q, q_type) for s in negative_passages
        ]
    
        for passage in positive_passages:
            num_history_questons = len(passage.question_boundaries)
            passage.dialog_act_id = sample['dialog_act']
            passage.history_dialog_act_ids = sample['history_dialog_act'][:num_history_questons]
        
        # no positive
        if is_train and any(not p.has_answer for p in positive_passages):
            return None
    
        if is_train:
            return ReaderSample(
                q,
                sample["answers"],
                id=sample["id"],
                positive_passages=positive_passages,
                negative_passages=negative_passages,
            )
        else:
            return ReaderSample(
                q,
                sample["answers"],
                id=sample["id"],
                passages=negative_passages,
            )


    def create_passage(self, passage, questions, question_types):
    
        """
            history question
        """
        # 0 is for the first CLS token, so we start from 1
        question_boundaries = [1]
        question_tokens, question_type_ids, history_question_lens = [], [], []
        for idx, q in enumerate(questions):
            tokens = self.tokenizer.tokenize(q) 
            type_ids = [question_types[idx]] * len(tokens)
    
            question_tokens.extend(tokens)
            question_type_ids.extend(type_ids)
    
            history_question_lens.append(len(tokens))
            question_boundaries.append(question_boundaries[-1] + len(tokens))
    
            # -2 is for CLS
            if sum(history_question_lens) >= self.max_history_len-2:
                break
    
        # -2 is for CLS
        question_tokens = question_tokens[:self.max_history_len-2]
        question_type_ids = question_type_ids[:self.max_history_len-2]
        question_boundaries[-1] = min(question_boundaries[-1], self.max_history_len-2)
        num_history_questions = len(history_question_lens)
        
        passage.history_answers_spans = passage.history_answers_spans[:num_history_questions]
        passage.history_has_answers = passage.history_has_answers[:num_history_questions]
        passage.question_boundaries = np.array(list(zip(question_boundaries[:-1], question_boundaries[1:])))
    
        """
            title
        """
        title_tokens = self.tokenizer.tokenize(passage.title)
        title_type_ids = len(title_tokens) * [0]
    
        history_and_title_tokens = [self.tokenizer.cls_token] + question_tokens \
            + [self.tokenizer.sep_token] + title_tokens + [self.tokenizer.sep_token]
        history_and_title_type_ids = [0] + question_type_ids + [0] + title_type_ids + [0]
    
        # -1 for the last SEP
        assert len(history_and_title_tokens) < self.max_seq_len-1, \
            "No space for passage tokens"
    
    
        """
            passage (spans)
        """
        shift = len(history_and_title_tokens)
        passage_tokens, passage_type_ids = [], []
        clss = [shift] # clss[0] is for the dummy span of <text> or <parent_text>
        ends = []
        for i, span in enumerate(passage.span_texts):
            if not self.args.use_sep_span_start:
                span_tokens = [self.tokenizer.cls_token] + self.tokenizer.tokenize(span)
            else:
                span_tokens = [self.tokenizer.sep_token] + self.tokenizer.tokenize(span)
            span_type_ids = [passage.span_types[i]] * len(span_tokens)
            next_cls_pos = clss[-1] + len(span_tokens)
    
            ends.append(next_cls_pos-1)
            clss.append(next_cls_pos)
    
            passage_tokens.extend(span_tokens)
            passage_type_ids.extend(span_type_ids)
    
        # -1 for the last SEP
        final_tokens = (history_and_title_tokens + passage_tokens)[:self.max_seq_len-1] + [self.tokenizer.sep_token]
        final_type_ids = (history_and_title_type_ids + passage_type_ids)[:self.max_seq_len-1] + [0]

    
        assert len(final_tokens) == len(final_type_ids)
    
        passage.sequence_ids = np.array(self.tokenizer.convert_tokens_to_ids(final_tokens))
        passage.sequence_type_ids = np.array(final_type_ids)
        passage.word_idxs = np.array(get_word_idxs(self.tokenizer, final_tokens, self.party_tokens, self.dont_mask_words))
    
        # the last item of clss is a dummy
        clss = clss[:-1]
    
        # (0, shift-1) is for history_and_title
        clss = [0] + clss
        ends = [shift-1] + ends
    
        # make sure all ends are less than max length
        ends = list(filter(lambda idx: idx < self.max_seq_len-1, ends))
        clss = clss[:len(ends)]
    
        num_spans = min(len(clss), self.max_num_spans_per_passage)
        clss = clss[:num_spans] + [-1] * (self.max_num_spans_per_passage - num_spans)
        ends = ends[:num_spans] + [-1] * (self.max_num_spans_per_passage - num_spans)
        clss = np.array(clss)
        ends = np.array(ends)
        mask_cls = 1 - (clss == -1)
        mask_cls[0] = 0 # 1st CLS (before history)
        clss[clss == -1] = 0
        ends[ends == -1] = 0
    
        passage.clss = clss
        passage.ends = ends
        passage.mask_cls = mask_cls
    
        if passage.has_answer:
            # +1 for <text> CLS offset 
            passage.answers_spans = [(s[0]+1, s[1]+1) for s in passage.answers_spans if s[1] + 1 < num_spans]
            passage.has_answer = (len(passage.answers_spans) > 0)

    
        """
            history spans
        """
        for i, s in enumerate(passage.history_answers_spans):
            if not s:
                continue
    
            # +1 for <text> CLS offset 
            s = [s[0]+1, s[1]+1]
            if not (s[1] < num_spans):
                s = None
            passage.history_answers_spans[i] = s
            passage.history_has_answers[i] = (s is not None)
    
        return passage


    @staticmethod
    def select_passages(sample, is_train):
        answers = sample["answers"]
    
        ctxs = [ReaderPassage(**ctx) for ctx in sample["ctxs"]]
    
        if is_train:
            positive_passages = list(filter(lambda ctx: ctx.has_answer, ctxs))
            negative_passages = list(filter(lambda ctx: not ctx.has_answer, ctxs))
        else:
            positive_passages = []
            negative_passages = ctxs
    
        return positive_passages, negative_passages
