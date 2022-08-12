# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os

import torch
import random
import json
import numpy as np

import logging
logger = logging.getLogger()

QUESTION_TOKEN_VALUE = 822
ANSWER_TOKEN_VALUE = 1525
COLON_TOKEN_VALUE = 10

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_context=None,
                 tokenizer=None,
                 max_length=None,
                 question_prefix='question:',
                 title_prefix='title:',
                 sub_title_prefix='sub-title:',
                 passage_prefix='context:'):
        self.data = data
        for i, _ in enumerate(data):
            if i % 100 == 0:
                logger.info("Processing example %d" % i)
            self.data[i] = format_conversational_example(self.data[i], \
                                                        title_prefix, \
                                                        sub_title_prefix, \
                                                        passage_prefix, \
                                                        tokenizer, \
                                                        max_length)
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.sub_title_prefix = sub_title_prefix
        self.passage_prefix = passage_prefix
        self.sort_data()

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target + ' </s>'
        elif 'answers' in example:
            return example['answers'][0] + ' </s>'
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]

        question = example['question']
        target = self.get_target(example)

        if 'ctxs' in example and self.n_context is not None:
            contexts = example['ctxs'][:self.n_context]
            passages = [c['text'] for c in contexts]
            scores = [float(c['score']) for c in contexts]
            scores = torch.tensor(scores)
            if len(contexts) == 0:
                contexts = [question]
        else:
            passages, scores = None, None


        return {
            'index' : index,
            'question' : question,
            'target' : target,
            'passages' : passages,
            'scores' : scores
        }

    def sort_data(self):
        if self.n_context is None or not 'score' in self.data[0]['ctxs'][0]:
            return
        for ex in self.data:
            ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

    def get_example(self, index):
        return self.data[index]

def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()


def format_conversational_example(example, title_prefix, sub_title_prefix, passage_prefix, tokenizer, max_length):
    question = example['question']
    conversation = [q.strip() for q in question.split('[SEP]')]
    for i, q in enumerate(conversation):
        if i % 2 == 0:
            conversation[i] = "question " + str(int(i/2 + 1)) + ": " + conversation[i]
        else:
            conversation[i] = "answer " + str(int(i/2+1)) + ": " + conversation[i]
    question = ' '.join(conversation)
    example['question'] = question

    f = "evidence: {} " + title_prefix + " {} " + passage_prefix + " {}"
    ctxs = []
    added = set()
    
    if 'prev_ctxs' in example:
        for i, ctx in enumerate(example['prev_ctxs']):
            ctx['text'] = f.format(ctx['id'], '|'.join([t.strip() for t in ctx['title'].split('[SEP]')]), ctx['text'])
            ctx['text'] = truncate_manually(question, ctx['text'], tokenizer, max_length)
            ctxs += [ctx]
            added.add(ctx['id'])

    f = "evidence: {} " +title_prefix + " {} " + passage_prefix + " {}"
    for i, ctx in enumerate(example['ctxs']):
        if ctx['id'] in added:
            continue

        ctx['text'] = f.format(ctx['id'], '|'.join([t.strip() for t in ctx['title'].split('[SEP]')]), ctx['text'])
        ctx['text'] = truncate_manually(question, ctx['text'], tokenizer, max_length)
        ctxs += [ctx]
    example['ctxs'] = ctxs
    
    return example


def truncate_manually(question, passage, tokenizer, max_length):

    q_enc = tokenizer.encode(question)
    p_enc = tokenizer.encode(passage)
    if len(q_enc) + len(p_enc) < max_length:
        return question + " " + passage

    assert q_enc[0] == QUESTION_TOKEN_VALUE
    assert q_enc[2] == COLON_TOKEN_VALUE

    passage_threshold = max_length / 2

    query_sub_parts = []
    start_idx = 0
    for i in range(3, len(q_enc)):
        if q_enc[i] in [QUESTION_TOKEN_VALUE, ANSWER_TOKEN_VALUE] and i + 2 < len(q_enc) and q_enc[i + 2] == COLON_TOKEN_VALUE:
            query_sub_parts.append(q_enc[start_idx : i])
            start_idx = i
    query_sub_parts.append(q_enc[start_idx:])
    query_sub_part_lengths = [len(sub_part) for sub_part in query_sub_parts]
    if len(query_sub_parts) == 1:
        query_sub_parts.append([])
    effective_length = len(query_sub_parts[0]) + len(query_sub_parts[-1]) + len(p_enc)
    while effective_length > max_length and len(p_enc) > passage_threshold:
        p_enc = p_enc[:-1]
        effective_length -=1
    query_sub_parts_idxs_to_add = []
    for i in range(len(query_sub_parts) - 2, 0, -1):
        if query_sub_part_lengths[i] + effective_length < max_length:
            query_sub_parts_idxs_to_add.append(i)
            effective_length += query_sub_part_lengths[i]
        else:
            break
    t1_token_ids = list(query_sub_parts[0])
    for idx in reversed(query_sub_parts_idxs_to_add):
        t1_token_ids += query_sub_parts[idx]
    t1_token_ids += query_sub_parts[-1]
    truncated_question = tokenizer.decode(t1_token_ids)
    truncated_passage = tokenizer.decode(p_enc)

    return truncated_question + " " + truncated_passage

class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        text_passages = [example['passages'] for example in batch]
        passage_ids, passage_masks = encode_passages(text_passages,
                                                     self.tokenizer,
                                                     self.text_maxlength)

        return (index, target_ids, target_mask, passage_ids, passage_masks)

def load_data(data_path=None, gold_data_path=None, n_context=-1, global_rank=-1, world_size=-1):
    assert data_path
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    examples = []
    for k, example in enumerate(data):
        if global_rank > -1 and not k%world_size==global_rank:
            continue
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
        if not 'id' in example:
            example['id'] = k
        for c in example['ctxs']:
            if not 'score' in c:
                c['score'] = 1.0 / (k + 1)
        if len(example['ctxs']) >= n_context:
            examples.append(example)

    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()
    if gold_data_path is not None:
        assert os.path.exists(gold_data_path)
        with open(gold_data_path, 'r', encoding='utf-8') as fin:
            gold_data = json.load(fin)
        gold_data_map = {}
        for k, example in enumerate(gold_data):
            gold_data_map[(example["conv_id"], int(example['turn_id']))] = example['ctxs']
        for ex in examples:
            if (ex['conv_id'], int(ex['turn_id'])) in gold_data_map:
                gold_passages = []
                all_pids = [c['id'] for c in ex['ctxs']]
                for gp in gold_data_map[(ex['conv_id'], int(ex['turn_id']))]:
                    if gp['id'] in all_pids:
                        continue
                    gold_passages += [{"id": gp['id'], "title": gp["title"], "text": gp["text"], "score": 10000.0}]
            else:
                print(ex['conv_id'], ex['turn_id'])
                assert False

            all_pids = [c['id'] for c in gold_data_map[(ex['conv_id'], int(ex['turn_id']))]]
            for i,_ in enumerate(ex['ctxs']):
                if ex['ctxs'][i]['id'] in all_pids:
                    ex['ctxs'][i]['score'] = 10000.0

            for gold_passage in gold_passages:
                ex['ctxs'].append(gold_passage)
    return examples

class RetrieverCollator(object):
    def __init__(self, tokenizer, passage_maxlength=200, question_maxlength=40):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength

    def __call__(self, batch):
        index = torch.tensor([ex['index'] for ex in batch])

        question = [ex['question'] for ex in batch]
        question = self.tokenizer.batch_encode_plus(
            question,
            pad_to_max_length=True,
            return_tensors="pt",
            max_length=self.question_maxlength,
            truncation=True
        )
        question_ids = question['input_ids']
        question_mask = question['attention_mask'].bool()

        if batch[0]['scores'] is None or batch[0]['passages'] is None:
            return index, question_ids, question_mask, None, None, None

        scores = [ex['scores'] for ex in batch]
        scores = torch.stack(scores, dim=0)

        passages = [ex['passages'] for ex in batch]
        passage_ids, passage_masks = encode_passages(
            passages,
            self.tokenizer,
            self.passage_maxlength
        )

        return (index, question_ids, question_mask, passage_ids, passage_masks, scores)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 title_prefix='title:',
                 passage_prefix='context:'):
        self.data = data
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        text = self.title_prefix + " " + example[2] + " " + \
            self.passage_prefix + " " + example[1]
        return example[0], text

class TextCollator(object):
    def __init__(self, tokenizer, maxlength=200):
        self.tokenizer = tokenizer
        self.maxlength = maxlength

    def __call__(self, batch):
        index = [x[0] for x in batch]
        encoded_batch = self.tokenizer.batch_encode_plus(
            [x[1] for x in batch],
            pad_to_max_length=True,
            return_tensors="pt",
            max_length=self.maxlength,
            truncation=True
        )
        text_ids = encoded_batch['input_ids']
        text_mask = encoded_batch['attention_mask'].bool()

        return index, text_ids, text_mask
