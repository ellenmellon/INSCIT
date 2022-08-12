import sys
import json
from collections import defaultdict
import argparse

def read_id2pid2score(input_fname, pred_fname):
    with open(input_fname) as fin:
        data = json.loads(fin.read())
        id2idx2pid = defaultdict(lambda: defaultdict(str))
        for e in data:
            id = e['id'].replace('conv_', '').replace('turn_', '')
            id = ('_'.join(id.split('_')[:-1]), int(id.split('_')[-1]))
            for ctx in e['ctxs']:
                idx = ctx['position']
                pid = ctx['id']
                id2idx2pid[id][idx] = pid

    with open(pred_fname) as fin:
        data = json.loads(fin.read())
        id2pid2score = defaultdict(lambda: defaultdict(float))
        for e in data:
            id = e['question'].replace('conv_', '').replace('turn_', '')
            id = ('_'.join(id.split('_')[:-1]), int(id.split('_')[-1]))
            for ctx in e['predictions']:
                pid = id2idx2pid[id][ctx["passage_idx"]]
                score = ctx['spans'][0]["prediction"]["relevance_score"]
                if score == None:
                    score = 0.0
                id2pid2score[id][pid] = score
    return id2pid2score


def read_id2labels(fname):
    id2labels = {}
    with open(fname) as fin:
        content = json.loads(fin.read())
        for convid in content:
            for turnid, turn in enumerate(content[convid]['turns']):
                id = (convid, turnid+1)
                if len(turn['context']) == 1:
                    id2labels[id] = turn['labels']
                else:
                    id2labels[id] = turn['labels']
    return id2labels

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
    return sum(f1_list) / len(f1_list)


def main(data_fname, input_fname, pred_fname):
    id2labels = read_id2labels(data_fname)
    id2pid2score = read_id2pid2score(input_fname, pred_fname)
    ave_n_evidence = 0
    selected_threshold = 0
    for t in range(0, 50, 1):  # candidate threshold value range: 0-5, selected empirically 
        threshold = t/10
        pred_evidence_list = []
        golds_evidence_list = []
        for id in id2labels:
            labels = id2labels[id]
            golds_evidence_list += [[[e['passage_id'] for e in l['evidence']] for l in labels]]
            pred = []
            pid2score = id2pid2score[id]
            for pid, score in sorted(pid2score.items(), key=lambda item: -item[1]):
                if pid == '':
                    continue
                if len(pred) == 0 or score > threshold:
                    pred += [pid]
                if len(pred) >= 4:
                    break
            pred_evidence_list += [pred]
        
        evidence_score = eval_evidence_passages(pred_evidence_list, golds_evidence_list)
        cur_ave_n_evidence = sum([len(p) for p in pred_evidence_list])/len(pred_evidence_list)

        # we observe that evidence scores are similar with many candidate threshold values,
        # thus we choose the one leading to the average number of evidence passages closest to training set 
        if abs(cur_ave_n_evidence - 1.5) < abs(ave_n_evidence - 1.5):
            ave_n_evidence = cur_ave_n_evidence
            selected_threshold = threshold
    
    print('selected threshold: ', selected_threshold)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_filename",
        type=str,
        default='../../data/dev.json',
        help="path of the data file",
    )
    parser.add_argument(
        "--dialki_input_filename",
        type=str,
        default='./inscit/data/dev_infer.json',
        help="path of the input file for dialki during inference",
    )
    parser.add_argument(
        "--pred_filename",
        type=str,
        default='./inscit/exp/dev_infer_predictions.json',
        help="path of the prediction file",
    )

    args = parser.parse_args()
    main(args.data_filename, args.dialki_input_filename, args.pred_filename)
