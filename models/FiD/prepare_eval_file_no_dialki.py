import json
import os
from tqdm import tqdm
import sys
import csv
from argparse import ArgumentParser

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


def convert_predictions(pid2info, data, predictions, out_fname):
    eval_examples = []
    for pred in predictions:

        output = {}
        
        pred_string = pred["predictions"][0]

        pred_passages = pred_string.split('answer: ')[0].replace('evidence:', '').split('|')
        pred_passages = set([p.strip() for p in pred_passages])
        if 'answer: ' in pred_string:
            pred_response = 'answer: '.join(pred_string.split('answer: ')[1:])
        else:
            pred_response = 'Sorry, I did not find any useful information.'  # default to assume no info found
    
        evidence = []
        for pid in list(pred_passages):
            if pid not in pid2info:
                continue
            evidence += [{"passage_id": pid, 
                          "passage_text": pid2info[pid]['contents'], 
                          "passage_titles": pid2info[pid]['titles']}]
        output['evidence'] = evidence
        output['response'] = pred_response       

        id = (pred["conv_id"], int(pred["turn_id"]))
        eval_example = {"conv_id": pred["conv_id"], 
                        "turn_id": pred["turn_id"], 
                        "context": data[id]["context"],
                        "output": output}
        eval_examples += [eval_example]
        
    with open(out_fname, 'w') as fout:
        fout.write(json.dumps(eval_examples, indent=2))


def main(args):
    data = read_data(args.data_file)

    with open(args.input_file) as fin:
        predictions = json.loads(fin.read())

    pid2info = read_pid2info(args.corpus_file)
    convert_predictions(pid2info, data, predictions, args.output_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--corpus_file", type=str, default='../DPR/retrieval_data/wikipedia/full_wiki_segments.tsv')
    parser.add_argument("--data_file", type=str, default='../../data/dev.json')
    parser.add_argument("--input_file", type=str, default='./reader_outputs/no_dialki/dev.json')
    parser.add_argument("--output_file", type=str, default='../../results/fid_dev.json')
    args = parser.parse_args()
    main(args)