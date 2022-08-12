from typing import Any, Dict, List, Optional, Tuple, Union, Iterator

import json
import os
from collections import namedtuple
import random
import argparse

from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm

import csv
import sys


DEFAULT_TYPE_ID = 0
USER_TYPE_ID = 1
AGENT_TYPE_ID = 2
TYPE_OFFSET = 2


class ICQADatasetReader:

    def __init__(self,
                 data_dir: str = None,
                 ir_dir: str = None,
                 corpus_path: str = None) -> None:
        self._datapath = data_dir
        self._irpath = ir_dir
        self._corpus_path = corpus_path
        self.corpus = self._load_corpus()

    def _load_corpus(self):
        pid2info = {}
        id_col= 0
        text_col= 1
        title_col = 2
        csv.field_size_limit(sys.maxsize)
        with open(self._corpus_path, 'r') as fin:
            reader = csv.reader(fin, delimiter="\t")
            for i, row in enumerate(tqdm(reader)):
                if row[id_col] == "id":
                    continue
                title = row[title_col]
                text = row[text_col]
                docid = row[id_col]
    
                doc_title = title
                doc_contents = text
                pid2info[docid] = {"title": doc_title, "contents": doc_contents} 
        return pid2info

    def _load_conversations(self, mode: str):
        conversations = {}
        with open(f'{self._datapath}/{mode}.json') as fin:
            content = json.loads(fin.read())
            for cid in content:
                conversations[cid] = content[cid]['turns']
        return conversations

    def _load_ir_results(self, mode: str, filename: str):
        id2ir = {}
        with open(f'{self._irpath}/{filename}') as fin:
            examples = json.loads(fin.read())
            for e in examples:
                id = (e['conv_id'], int(e['turn_id']))
                id2ir[id] = e['ctxs']
        return id2ir


class ICQAPassage:

    def __init__(self, id, title, position, sentences):
        self.id = id 
        self.title = title
        self.position = position
        self.text = sentences
        self.type = [0] * len(sentences) # put a dummy one
        self.has_answer = None
        self.history_answers_spans = None
        self.history_has_answers = None
        self.answers_spans = None


class ICQADataSample:

    def __init__(self, conv_id, turn_id, chosen_passage):
        self.id = f'conv_{conv_id}_turn_{turn_id}'

        self.question = None

        self.question_type = None # list of 1s and 2s indicating previous turn roles
        self.history_dialog_act = None 
        self.dialog_act = 0  # put a dummy one
        self.answers = [chosen_passage]
        
        self.ctxs = None


    def set_question(self, question, question_type):
        self.question = question
        self.question_type = question_type
        assert len(question_type) == len(question)
        self.history_dialog_act = [0] * len(question) # put a dummy one


    def add_passage(self, 
                    pid, 
                    title, 
                    position, 
                    sentences, 
                    has_answer, 
                    history_has_answers):

        if self.ctxs is None:
            self.ctxs = []
        
        title = '|'.join([subtitle.strip() for subtitle in title.split('[SEP]')])
        passage = ICQAPassage(pid, title, position, sentences)
        passage.has_answer = has_answer
        if has_answer:
            passage.answers_spans = [[0, 0]]
        
        history_answers_spans = []
        last_prev_turn_has_ans = -1
        for i, hist_has_ans in enumerate(history_has_answers):
            if not hist_has_ans:
                history_answers_spans += [None]
            else:
                history_answers_spans += [[0, 0]]
                last_prev_turn_has_ans = i
        if last_prev_turn_has_ans >= 0:
            passage.type = [len(history_has_answers) - last_prev_turn_has_ans + TYPE_OFFSET]
        else:
            passage.type = [DEFAULT_TYPE_ID]
        passage.history_has_answers = list(reversed(history_has_answers))
        passage.history_answers_spans = list(reversed(history_answers_spans))
        
        self.ctxs += [vars(passage)]
    
    def _get_history_has_answers(self, prev_answer_pids, pid):
        history_has_answers = []
        for pids in prev_answer_pids:
            if pid in pids:
                history_has_answers += [False, True]
            else:
                history_has_answers += [False, False]
        history_has_answers += [False] # last user turn
        return history_has_answers

    def _process_one_passage(self, pid, position, has_answer, prev_answer_pids, pid2info):
        history_has_answers = self._get_history_has_answers(prev_answer_pids, pid)
        total_found_prev_pids = sum([int(ha) for ha in history_has_answers])
        self.add_passage(pid, 
                         pid2info[pid]['title'], 
                         position, 
                         [pid2info[pid]['contents']], 
                         has_answer, 
                         history_has_answers)
        return total_found_prev_pids


class ICQADialog():
    
    def __init__(self, conv_id, turns, exclude_prev_passages, inference_only):
        self.dialid = conv_id
        self.utterances = turns
        self.exclude_prev_passages = exclude_prev_passages
        self.inference_only = inference_only

    def _get_pid(self, evidence):
        return evidence['passage_id']

    def read_samples(self, pid2info, id2ir):

        samples = []
        perc_found_prev_answers = []

        for i, turn in enumerate(self.utterances):

            turn_id = i + 1

            ir_results = id2ir[(self.dialid, turn_id)]

            context = turn['context']
            prev_answers = turn['prevEvidence']
            prev_answer_pids = []
            total_prev_pids = 0
            total_found_prev_pids = 0
            for prev in prev_answers:
                prev_answer_pids += [[self._get_pid(prev_evidence) for prev_evidence in prev]]
                total_prev_pids += len(set(prev_answer_pids[-1]))

            labels = turn['labels']

            all_answer_pids = []
            for label in labels:
                for evidence in label['evidence']:
                    all_answer_pids += [self._get_pid(evidence)]
            
            if not self.inference_only and len(all_answer_pids) == 0:
                continue

            prev_type = []
            prev_turns = []
            for i, utterance in enumerate(context):
                if i%2 == 0:
                    prev_type += [USER_TYPE_ID]
                    prev_turns += [f"<user> {utterance.strip()}"]
                else:
                    prev_type += [AGENT_TYPE_ID]
                    prev_turns += [f"<agent> {utterance.strip()}"]

            # make data sample
            if not self.inference_only:
                sample = ICQADataSample(self.dialid, turn_id, pid2info[all_answer_pids[0]]['contents'])
            else:
                sample = ICQADataSample(self.dialid, turn_id, "placeholder answer")
            sample.set_question(list(reversed(prev_turns)), list(reversed(prev_type)))

            position = 0
            added = set()
            # first add positive passage
            if not self.inference_only:
                pid = all_answer_pids[0]
                total_found_prev_pids += sample._process_one_passage(pid, position, True, prev_answer_pids, pid2info)
                added.add(pid)
                position += 1

            if not self.exclude_prev_passages:
                for pids in prev_answer_pids:
                    for pid in pids:
                        if pid in added:
                            continue
                        if not self.inference_only and pid in all_answer_pids:  # make sure to only include one answer for training and development purposes
                            continue
                        if position >= 50:  # only include 50 passages
                            break
                        total_found_prev_pids += sample._process_one_passage(pid, position, False, prev_answer_pids, pid2info)
                        added.add(pid)
                        position += 1
                        

            # add more passages
            for ctx in ir_results:
                pid = ctx['id']
                
                if pid in added:
                    continue
                if not self.inference_only and pid in all_answer_pids:  # make sure to only include one answer for training and development purposes
                    continue                
                if position >= 50:  # only include 50 passages
                    break

                total_found_prev_pids += sample._process_one_passage(pid, position, False, prev_answer_pids, pid2info)
                added.add(pid)
                position += 1
            
            samples += [sample]
            if total_prev_pids > 0:
                perc_found_prev_answers += [total_found_prev_pids * 1.0 / total_prev_pids]

        return samples, perc_found_prev_answers


def write_examples(dialogues, outdir, split, exclude_prev_passages, inference_only, pid2info, id2ir):
    dialogues = [ICQADialog(id, dialogues[id], exclude_prev_passages, inference_only) for id in dialogues]
    qas = []
    perc_found_prev_answers_list = []
    for dial in dialogues:
        samples, perc_found_prev_answers = dial.read_samples(pid2info, id2ir)
        perc_found_prev_answers_list += perc_found_prev_answers
        qas += [vars(sample) for sample in samples]
            
    print(f'{split} examples: {len(qas)}')
    print(f'percentage of previous answers can be found in current candidate passages: {sum(perc_found_prev_answers_list)/len(perc_found_prev_answers_list)}')
    with open(os.path.join(outdir, f'{split}.json'), 'w', encoding="utf-8") as fout:
        fout.write(json.dumps(qas, indent=4))


def main(args):

    reader = ICQADatasetReader(args.data_dir, args.ir_dir, args.corpus_path)
    dialogues = reader._load_conversations('train')
    ir_results = reader._load_ir_results("train", 'dpr_train.json')
    write_examples(dialogues, args.out_dir, 'train', args.exclude_prev_passages, False, reader.corpus, ir_results)
    write_examples(dialogues, args.out_dir, 'train_infer', args.exclude_prev_passages, True, reader.corpus, ir_results)
    
    dialogues = reader._load_conversations('dev')
    ir_results = reader._load_ir_results("dev", 'dpr_dev.json')
    write_examples(dialogues, args.out_dir, 'dev', args.exclude_prev_passages, False, reader.corpus, ir_results)
    write_examples(dialogues, args.out_dir, 'dev_infer', args.exclude_prev_passages, True, reader.corpus, ir_results)


    if not os.path.exists(f'{args.data_dir}/test.json'):
        print('Skipping data preparation for the test set ... ')
        return

    dialogues = reader._load_conversations('test')
    ir_results = reader._load_ir_results("test", 'dpr_test.json')
    write_examples(dialogues, args.out_dir, 'test_infer', args.exclude_prev_passages, True, reader.corpus, ir_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default='../../data',
        help="directory path of the data",
    )
    parser.add_argument(
        "--ir_dir",
        type=str,
        default='../DPR/retrieval_outputs/results',
        help="directory path of the retrieval output data",
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default='../DPR/retrieval_data/wikipedia/full_wiki_segments.tsv',
        help="directory path of the corpus tsv file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default='inscit/data',
        help="directory path of the output data",
    )
    parser.add_argument(
        "--exclude_prev_passages",
        action='store_true',
        help="exclude prev passages or not",
    )
    args = parser.parse_args()


    main(args)
