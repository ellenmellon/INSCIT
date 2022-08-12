import math
import json
import datasets
from argparse import ArgumentParser
import collections
import spacy
import string
import re


nlp = spacy.load("en_core_web_sm")


def eval_evidence_passages(pred_list, golds_list):
    f1_list = []
    for pred, golds in zip(pred_list, golds_list):
        max_f1 = 0.0
        for gold in golds:
            for p in pred:
                tp = len(set(pred).intersection(set(gold)))
                fp = len(set(pred)) - tp
                fn = len(set(gold)) - tp
                if tp + fp + fn == 0:
                    f1 = 0.0
                else:
                    f1 = tp / (tp + 0.5*(fp+fn))
                if f1 > max_f1:
                    max_f1 = f1
        f1_list += [max_f1]
    return sum(f1_list)*100 / len(f1_list)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()


def compute_f1(pred_list, golds_list):
    scores = []
    for pred, golds in zip(pred_list, golds_list):
        new_pred = ' '.join([t.text for t in list(nlp(pred))])
        max_f1 = 0.0
        for gold in golds:
            new_gold = ' '.join([t.text for t in list(nlp(gold))])
            gold_toks = get_tokens(new_gold)
            pred_toks = get_tokens(new_pred)
            common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
            num_same = sum(common.values())
            if len(gold_toks) == 0 or len(pred_toks) == 0:
                # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
                f1 = int(gold_toks == pred_toks)
            elif num_same == 0:
                f1 = 0
            else:
                precision = 1.0 * num_same / len(pred_toks)
                recall = 1.0 * num_same / len(gold_toks)
                f1 = (2 * precision * recall) / (precision + recall)
            if f1 > max_f1:
                max_f1 = f1
        scores += [max_f1]
    return sum(scores)*100.0 / len(scores)


def eval_all_responses(pred_list, golds_list):
    sacrebleu = datasets.load_metric("sacrebleu")
    new_golds_list = []
    for golds in golds_list:
        if len(golds) == 1:
            new_golds_list += [golds * 2]  # sacrebleu requires each reference list has the same length
        else:
            new_golds_list += [golds]
    score = sacrebleu.compute(predictions=pred_list, references=new_golds_list)['score']
    f1 = compute_f1(pred_list, golds_list)
    return score, f1


def read_id2labels(fname):
    id2labels = {}
    with open(fname) as fin:
        content = json.loads(fin.read())
        for convid in content:
            for turnid, turn in enumerate(content[convid]['turns']):
                id = (convid, turnid+1, ' '.join(turn['context']))
                if len(turn['context']) == 1:
                    id2labels[id] = turn['labels']
                else:
                    id2labels[id] = turn['labels']
    return id2labels


def read_id2pred(fname):
    id2pred = {}
    with open(fname) as fin:
        content = json.loads(fin.read())
        for example in content:
            id = (example['conv_id'], int(example['turn_id']), ' '.join(example['context']))
            id2pred[id] = example['output']             
    return id2pred


def main(args):
    id2labels = read_id2labels(args.data_file)
    id2pred = read_id2pred(args.results_file)
    pred_evidence_list = []
    golds_evidence_list = []
    pred_r_list = []
    golds_r_list = []
    for id in id2labels:

        assert id in id2pred
        labels = id2labels[id]
        pred = id2pred[id]

        pred_evidence_list += [[e['passage_id'] for e in pred['evidence']]]
        golds_evidence_list += [[[e['passage_id'] for e in l['evidence']] for l in labels]]

        pred_r_list += [' '.join(pred['response'].lower().split())]
        golds_r_list += [[' '.join(l['response'].lower().split()) for l in labels]]
    
    evidence_f1 = eval_evidence_passages(pred_evidence_list, golds_evidence_list)
    bleu, response_f1 = eval_all_responses(pred_r_list, golds_r_list)
    print('Evidence F1: ', round(evidence_f1, 1))
    print('Response sacrebelu score: ', round(bleu, 1))
    print('Response F1: ', round(response_f1, 1))
    print('Total # examples: ', len(pred_evidence_list))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_file", type=str, default='./data/dev.json')
    parser.add_argument("--results_file", type=str, default='./results/dialki_fid_dev.json')
    args = parser.parse_args()
    main(args)
