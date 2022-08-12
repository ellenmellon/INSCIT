import sys
import os
import json
from tqdm import tqdm
import pathlib
import csv
import random
from argparse import ArgumentParser


def read_corpus(corpus_file):
    corpus = {}
    id_col= 0
    text_col= 1
    title_col = 2
    csv.field_size_limit(sys.maxsize)
    with open(corpus_file, 'r') as fin:
        reader = csv.reader(fin, delimiter="\t")
        for i, row in enumerate(tqdm(reader)):
            if row[id_col] == "id":
                continue
            title = row[title_col]
            text = row[text_col]
            docid = row[id_col]
            
            doc_title = title
            doc_contents = text
            corpus[docid] = {"title": doc_title, "contents": doc_contents}
    return corpus


def create_no_neg_examples(data_file):
    examples = []
    with open(data_file, 'rb') as fin:
        content = json.load(fin)
        for cid in content:
            for tid, turn in enumerate(content[cid]['turns']):
                example = {}
                if 'train' in data_file:
                    example['dataset'] = 'inscit_train'
                else:
                    example['dataset'] = 'inscit_dev'
                example['question'] = ' [SEP] '.join([' '.join(t.split()) for t in turn['context']])
                example['answers'] = [' '.join(l['response'].split()) for l in turn['labels']]
                example["conv_id"] = cid
                example['turn_id'] = tid + 1
                
                pos_ctxs = []
                added = set()
                for label in turn['labels']:
                    for e in label['evidence']:
                        pos_ctx = {}
                        titles = [t[:-1] if t.endswith('.') else t for t in e['passage_titles']]
                        pos_ctx['title'] = ' [SEP] '.join(titles)
                        pos_ctx['text'] = e["passage_text"]
                        pos_ctx["score"] = 1000
                        pos_ctx["title_score"] = 1
                        pid = e['passage_id']
                        pos_ctx['passage_id'] = pid
                        if pid in added:
                            continue
                        added.add(pid)
                        pos_ctxs += [pos_ctx]
                example["positive_ctxs"] = pos_ctxs
                examples += [example]
    return examples


def sample_negatives(positive_pids, candidates, corpus):
    negatives = [p for p in candidates['ctxs'] if p['id'] not in positive_pids]
    negatives = random.sample(negatives, 30)
    sampled_negatives = []
    for n in negatives:
        sampled_negatives += [{"title": corpus[n["id"]]["title"], 
                               "text": corpus[n["id"]]["contents"], 
                               "score": n["score"], 
                               "passage_id": n["id"]}]
    return sampled_negatives
    

def create_has_neg_examples(corpus, no_neg_examples, bm25_results, bm25_grounding_results):

    has_neg_examples = []

    for example, bm25_result, bm25_g_result in zip(no_neg_examples, bm25_results, bm25_grounding_results):
    
        assert example['conv_id'] == bm25_result['conv_id']
        assert example['conv_id'] == bm25_g_result['conv_id']
        assert int(example['turn_id']) == int(bm25_result['turn_id'])
        assert int(example['turn_id']) == int(bm25_g_result['turn_id'])
        
        positive_pids = [p["passage_id"] for p in example['positive_ctxs']]
    
        example['hard_negative_ctxs'] = sample_negatives(positive_pids, bm25_g_result, corpus)
        example['negative_ctxs'] = sample_negatives(positive_pids, bm25_result, corpus)
        has_neg_examples += [example]

    return has_neg_examples


def create_qas_file(output_qas_file, examples):
    with open(output_qas_file, 'w') as fout:
        for e in examples:
            line = f'{e["question"]}\t{e["conv_id"]}\t{e["turn_id"]}\t{e["answers"]}\n'
            fout.write(line)


def main(corpus_file, data_dir, ir_dir, output_dir, output_qas_dir):

    corpus = read_corpus(corpus_file)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_qas_dir).mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'dev', 'test']:

        data_file = os.path.join(data_dir, f'{split}.json')
        if split == 'test' and not os.path.exists(data_file):
            print('Skipping data preparation for the test set ... ')
            return
        no_neg_examples = create_no_neg_examples(data_file)
        
        if split != 'test':
            bm25_file = os.path.join(ir_dir, f'bm25_{split}.json')
            bm25_grounding_file = os.path.join(ir_dir, f'bm25_grounding_{split}.json')
            
            output_file = os.path.join(output_dir, f'{split}.json')
            with open(bm25_file) as fin, open(bm25_grounding_file) as fin_g:
                bm25_results = json.loads(fin.read())
                bm25_grounding_results = json.loads(fin_g.read())

            has_neg_examples = create_has_neg_examples(corpus, no_neg_examples, bm25_results, bm25_grounding_results)
            with open(output_file, 'w') as fout:
                fout.write(json.dumps(has_neg_examples, indent=2))
        
        output_qas_file = os.path.join(output_qas_dir, f'{split}.tsv')
        create_qas_file(output_qas_file, no_neg_examples)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--corpus_file", type=str, default='./retrieval_data/wikipedia/full_wiki_segments.tsv')
    parser.add_argument("--data_dir", type=str, default='../../data')
    parser.add_argument("--ir_dir", type=str, default='./retrieval_outputs/results')
    parser.add_argument("--output_dir", type=str, default='./retrieval_data/train')
    parser.add_argument("--output_qas_dir", type=str, default='./retrieval_data/qas')

    args = parser.parse_args()

    main(args.corpus_file, 
         args.data_dir, 
         args.ir_dir, 
         args.output_dir, 
         args.output_qas_dir)