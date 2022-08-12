import json
import csv
from tqdm import tqdm
import sys
import os
import pathlib
from collections import defaultdict
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

            doc_title = title
            doc_contents = text
            pid2info[docid] = {"title": doc_title, "contents": doc_contents} 
    return pid2info


def read_data(fname):
    examples = []
    with open(fname) as fin:
        content = json.loads(fin.read())
        for cid in content:
            for tid, turn in enumerate(content[cid]['turns']):
                examples += [(cid, tid+1, turn)]
    return examples


def read_ir_results(fname, pid2info, data):
    with open(fname) as fin:
        examples = json.loads(fin.read())
        for e, d in zip(examples, data):
            assert d[0] == e['conv_id']
            assert int(d[1]) == int(e['turn_id'])
            
            e['answers'] = []
            for i, ans in enumerate(d[2]['labels']):
                evi_list = d[2]['labels'][i]['evidence']
                evidenceString_list = [evi['passage_id'] for evi in evi_list]
                evidenceString = ' | '.join(sorted(list(set(evidenceString_list))))
                ansString = ' '.join(d[2]['labels'][i]['response'].split())
                e['answers'] += [f"evidence: {evidenceString} answer: {ansString}"]

            for i, ctx in enumerate(e['ctxs']):
                e['ctxs'][i]['title'] = pid2info[e['ctxs'][i]['id']]['title']
                e['ctxs'][i]['text'] = pid2info[e['ctxs'][i]['id']]['contents']

            prev_ctxs = []
            added = set()
            for i, es in enumerate(reversed(d[2]['prevEvidence'])):
                for j, evi in enumerate(es):
                    pid = evi['passage_id']
                    if pid in added:
                        continue
                    prev_ctxs += [{"id": pid, 
                                   "title": pid2info[pid]['title'], 
                                   "text": pid2info[pid]['contents'], 
                                   "score": 10000.0,
                                   "turn_distance": i+1}]
                    added.add(pid)
            e['prev_ctxs'] = prev_ctxs
            e['question'] = e['question'].strip()
    return examples


def create_gp_info(data, pid2info, ir_results):
    gp_info = []
    for d, ir in zip(data, ir_results):
        assert d[0] == ir['conv_id']
        assert int(d[1]) == int(ir['turn_id'])
        info = {'question': ir['question'], 
                'short_answers': ir['answers'], 
                'conv_id': d[0], 
                'turn_id': d[1]}

        ctxs = []
        idx = 0
        for e in d[2]['labels'][idx]['evidence']:
            pid = e['passage_id']
            ctxs += [{"id": pid, 
                      "title": pid2info[pid]['title'], 
                      "text": pid2info[pid]['contents']}]
        info['ctxs'] = ctxs
        gp_info += [info]
    return gp_info



def main(corpus_file, data_folder, ir_result_folder, output_folder):

    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
    pid2info = read_pid2info(corpus_file)
    
    for split in ['train', 'dev', 'test']:
    
        data_file = f'{data_folder}/{split}.json'
        if split == 'test' and not os.path.exists(data_file):
            print('Skipping data preparation for the test set ... ')
            return

        data = read_data(data_file)
        
        ir_result_file = f'{ir_result_folder}/dpr_{split}.json'
        ir_results = read_ir_results(ir_result_file, pid2info, data)
        
        with open(f'{output_folder}/{split}.json', 'w') as fout:
            fout.write(json.dumps(ir_results, indent=2))
        
        if split == 'train':
            gp_info = create_gp_info(data, pid2info, ir_results)
            with open(f'{output_folder}/train_gold.json', 'w') as fout:
                fout.write(json.dumps(gp_info, indent=2))


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--corpus_file", type=str, default='../DPR/retrieval_data/wikipedia/full_wiki_segments.tsv')
    parser.add_argument("--data_folder", type=str, default='../../data')
    parser.add_argument("--ir_result_folder", type=str, default='../DPR/retrieval_outputs/results')
    parser.add_argument("--output_folder", type=str, default='./reader_data/no_dialki')
    args = parser.parse_args()
    main(args.corpus_file, args.data_folder, args.ir_result_folder, args.output_folder)

