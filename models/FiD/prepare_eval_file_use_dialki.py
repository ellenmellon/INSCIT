import json
import os
from tqdm import tqdm
import sys
import csv
from argparse import ArgumentParser
from collections import defaultdict


def get_id(question_id):
    id = question_id.replace('conv_', '').replace('turn_', '')
    id = ('_'.join(id.split('_')[:-1]), int(id.split('_')[-1]))
    return id


def read_pid2info(fname):
    pid2info = {}
    id_col= 0
    text_col= 1
    title_col = 2
    csv.field_size_limit(sys.maxsize)
    with open(fname, 'r') as fin:
        reader = csv.reader(fin, delimiter="\t")
        for i, row in enumerate(tqdm(reader)):
            if row[id_col] == "id":
                continue
            title = row[title_col]
            text = row[text_col]
            docid = row[id_col]

            doc_titles = [t.strip() for t in title.split('[SEP]')]
            doc_contents = text
            pid2info[docid] = {"titles": doc_titles, "contents": doc_contents} 
    return pid2info


def read_data(fname):
    examples = {}
    with open(fname) as fin:
        content = json.loads(fin.read())
        for cid in content:
            for tid, turn in enumerate(content[cid]['turns']):
                examples[(cid, tid+1)] = turn
    return examples


def read_id2evidence(pi_dir, split, threshold):

    with open(f'{pi_dir}/data/{split}_infer.json') as fin:
        data = json.loads(fin.read())
        id2idx2pid = defaultdict(lambda: defaultdict(str))
        for e in data:
            id = get_id(e['id'])
            for ctx in e['ctxs']:
                idx = ctx['position']
                pid = ctx['id']
                id2idx2pid[id][idx] = pid

    with open(f'{pi_dir}/exp/{split}_infer_predictions.json') as fin:
        data = json.loads(fin.read())
        id2pid2score = defaultdict(lambda: defaultdict(float))
        for e in data:
            id = get_id(e['question'])
            for ctx in e['predictions']:
                pid = id2idx2pid[id][ctx["passage_idx"]]
                score = ctx['spans'][0]["prediction"]["relevance_score"]
                if score == None:
                    score = 0.0
                id2pid2score[id][pid] = score

    id2evidence = defaultdict(list)
    for id in id2pid2score:
        pid2score = id2pid2score[id]
        for pid, score in sorted(pid2score.items(), key=lambda item: -item[1]):
            if pid == '':
                continue
            if len(id2evidence[id]) == 0 or score > threshold:
                id2evidence[id] += [pid]
            if len(id2evidence[id]) >= 4:
                break

    return id2evidence


def read_id2response(rg_file):
    with open(rg_file) as fin:
        predictions = json.loads(fin.read())
        id2response = {}
        for pred in predictions:
            response = pred["predictions"][0]
            id = (pred['conv_id'], int(pred['turn_id']))
            id2response[id] = response
        return id2response


def main(args):
    pid2info = read_pid2info(args.corpus_tsv_file)
    data = read_data(args.data_file)
    if args.threshold is None:
        print("Warning: no threshold is set, and thus generating empty evidence ..")
        id2evidence = defaultdict(list)
    else:
        id2evidence = read_id2evidence(args.pi_dir, args.split, args.threshold)
    id2response = read_id2response(args.rg_file)

    eval_examples = []
    for id in data:

        output = {"response": id2response[id]}

        evidence = []
        for pid in id2evidence[id]:
            evidence += [{"passage_id": pid, 
                          "passage_text": pid2info[pid]['contents'], 
                          "passage_titles": pid2info[pid]['titles']}]
        
        output['evidence'] = evidence

        eval_example = {"conv_id": id[0], 
                        "turn_id": id[1], 
                        "context": data[id]["context"],
                        "output": output}
        eval_examples += [eval_example]

    with open(args.output_file, 'w') as fout:
        fout.write(json.dumps(eval_examples, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--corpus_tsv_file", 
        type=str, 
        default='../DPR/retrieval_data/wikipedia/full_wiki_segments.tsv'
    )
    parser.add_argument(
        "--data_file", 
        type=str, 
        default='../../data/dev.json'
    )
    parser.add_argument(
        "--pi_dir",
        type=str,
        default='../DIALKI/inscit',
        help="directory path of dialki data and model",
    )
    parser.add_argument(
        "--rg_file",
        type=str,
        default='./reader_outputs/use_dialki/dev.json',
        help="file path of the fid response generation prediction",
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default='../../results/dialki_fid_dev.json'
    )
    parser.add_argument("--split", type=str, default='dev')
    parser.add_argument("--threshold", type=float, default=2.4)
    args = parser.parse_args()
    main(args)
