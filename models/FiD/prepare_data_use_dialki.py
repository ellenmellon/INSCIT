import json
import sys
from tqdm import tqdm
import os
import argparse
import csv
from collections import defaultdict

def _load_corpus(corpus_path):
    pid2info = {}
    id_col= 0
    text_col= 1
    title_col = 2
    csv.field_size_limit(sys.maxsize)
    with open(corpus_path, 'r') as fin:
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

def _get_pid(evidence):
    return evidence['passage_id']

def _load_examples(datapath, mode):
    id2example = {}
    with open(f'{datapath}/{mode}.json') as fin:
        content = json.loads(fin.read())
        for cid in content:
            for tid, turn in enumerate(content[cid]['turns']):
                id = (cid, tid+1)
                context = turn['context']
                label = turn['labels'][0]
                pids = [_get_pid(e) for e in label['evidence']]
                target = [' '.join(l['response'].split()) for l in turn['labels']]
                id2example[id] = (context, pids, target)
    return id2example

def _load_eval_pids(pipath, mode, threshold):
    with open(f'{pipath}/data/{mode}_infer.json') as fin:
        data = json.loads(fin.read())
        id2idx2pid = defaultdict(lambda: defaultdict(str))
        for e in data:
            id = e['id'].replace('conv_', '').replace('turn_', '')
            id = ('_'.join(id.split('_')[:-1]), int(id.split('_')[-1]))
            for ctx in e['ctxs']:
                idx = ctx['position']
                pid = ctx['id']
                id2idx2pid[id][idx] = pid
    
    id2pids = {}
    with open(f'{pipath}/exp/{mode}_infer_predictions.json') as fin:
        content = json.loads(fin.read())
        for e in content:
            id = e['question'].replace('conv_', '').replace('turn_', '')
            id = ('_'.join(id.split('_')[:-1]), int(id.split('_')[-1]))
            pids = []
            pid2score = {}
            for p in e["predictions"]:
                pid = id2idx2pid[id][p['passage_idx']]
                if pid == '':
                    continue
                score = p['spans'][0]["prediction"]["relevance_score"]
                if score == None:
                    score = 0.0
                pid2score[pid] = score
            for pid, score in sorted(pid2score.items(), key=lambda item: -item[1]):
                if len(pids) == 0 or score > threshold:
                    pids += [pid]
                if len(pids) >= 4:
                    break

            id2pids[id] = pids
    return id2pids


def process_example(id, pid2info, context, pids, target):
    question = ' [SEP] '.join(context)
    ctxs = []
    for pid in pids:
        title = pid2info[pid]['title']
        text = pid2info[pid]['contents']
        ctx = {"id": pid, "title": title, "text": text, "score": 1000.0, "has_answer": False}
        ctxs += [ctx]
    for i in range(4-len(ctxs)):
        dummy_ctx = {"id": "", "title": "", "text": "", "score": 0.0, "has_answer": False}
        ctxs += [dummy_ctx] 
    return {"question": question, "conv_id": id[0], "turn_id": id[1], "ctxs": ctxs, "answers": target}


def write_examples(id2example, pid2info, out_fname, is_train, id2pids=None):
    examples = []
    with open(out_fname, 'w') as fout:
        for id in id2example:
            context, pids, target = id2example[id]
            if not is_train:
                pids = id2pids[id]
            example = process_example(id, pid2info, context, pids, target)
            examples += [example]
        fout.write(json.dumps(examples, indent=2))


def main(args):
    pid2info = _load_corpus(args.corpus_path)

    os.makedirs(args.out_dir, exist_ok=True)

    id2example = _load_examples(args.data_dir, "train")
    out_fname = args.out_dir + '/train.json'
    write_examples(id2example, pid2info, out_fname, True, id2pids=None)


    id2example = _load_examples(args.data_dir, "dev")
    id2pids = _load_eval_pids(args.pi_dir, "dev", args.threshold)
    out_fname = args.out_dir + '/dev.json'
    write_examples(id2example, pid2info, out_fname, False, id2pids=id2pids)


    if not os.path.exists(f'{args.data_dir}/test.json'):
        print('Skipping data preparation for the test set ... ')
        return

    id2example = _load_examples(args.data_dir, "test")
    id2pids = _load_eval_pids(args.pi_dir, "test", args.threshold)
    out_fname = args.out_dir + '/test.json'
    write_examples(id2example, pid2info, out_fname, False, id2pids=id2pids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default='../../data',
        help="directory path of the data",
    )
    parser.add_argument(
        "--pi_dir",
        type=str,
        default='../DIALKI/inscit',
        help="directory path of the passage identification output data",
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
        default='./reader_data/use_dialki',
        help="directory path of the output data",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.4,
        help="score threhold of predicted passages",
    )
    args = parser.parse_args()


    main(args)
