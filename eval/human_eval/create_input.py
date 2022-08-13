import json
import re
import glob
import random
from collections import Counter
from argparse import ArgumentParser


def get_pid2info(corpus_folder, all_needed_pids):
    pid2info = {}
    for i, fname in enumerate(glob.glob(corpus_folder+'/*')):
        print("processing: ", i, fname)
        with open(fname) as fin:
            for page in json.loads(fin.read()):
                for p in page["passages"]:
                    if p["id"] in all_needed_pids:
                        pid2info[p["id"]] = {"titles": p["titles"], "text": p["text"]}
    return pid2info


def get_qid2example(system_filename):
    qid2example = {}
    with open(system_filename) as fin:
        for example in json.loads(fin.read()):
            qid = f'{example["conv_id"]}_turn_{example["turn_id"]}'
            qid2example[qid] = {"context": example["context"], 
                                "response": example["output"]["response"], 
                                "evidence": example["output"]["evidence"]}
            if "responseType" in example["output"]:
                qid2example[qid]["responseType"] = example["output"]["responseType"]
    return qid2example


def read_eval_conv_ids(eval_ids_filename):
    eval_conv_ids = set()
    with open(eval_ids_filename) as fin:
        for line in fin:
            eval_conv_ids.add(line.strip())
    return eval_conv_ids


def normalize_titles(titles):
    titles = [t.strip() for t in titles if t.strip() != '']
    result_title = ''
    for i, t in enumerate(titles):
        if i == 0:
            result_title += f'- <em>Article Title</em>: {t} '
        elif i == 1:
            result_title += f'<br>- <em>Section Title</em>: {t}'
        else:
            result_title += f'<br>- <em>Subsection #{i-1} Title</em>: {t}'
    return result_title


def normalize_text(text):
    lines = [line.strip() for line in text.split('\n')]
    normalized_lines = []
    list_marks = ['*', ';', ':', '#']
    for line in lines:
        end_list_mark_index = 0
        for idx, char in enumerate(line):
            if char in list_marks:
                end_list_mark_index = idx + 1
            else:
                break
        if end_list_mark_index > 0:
            normalized_lines += [end_list_mark_index * '--' + line[end_list_mark_index:]]
        else:
            normalized_lines += [line]
    result_string = ''
    cur_l = 0
    for i, line in enumerate(normalized_lines):
        if i == 0:
            result_string += line
            cur_l += len(line.split())
        elif cur_l + len(line.split()) > 200:
            result_string += ' ...'
            break
        else:
            result_string += f'<br>{line}'
            cur_l += len(line.split())
    return result_string


def construct_eval_input(qid, systems, examples, pid2info):
    eval_input_json = {"id": qid}
    system_example_tuples = list(zip(systems, examples))
    random.shuffle(system_example_tuples)
    assert len(system_example_tuples) == 3

    prompt_json = {"utterances": examples[0]["context"]}
    for i, (system, example) in enumerate(system_example_tuples):
        system_json = {"name": system}
        system_json["response"] = example["response"]
        evidence = []
        processed = set()
        for e in example["evidence"]:
            pid = e["passage_id"]
            if pid in processed:
                continue
            processed.add(pid)
            title = normalize_titles(pid2info[pid]["titles"])
            text = normalize_text(pid2info[pid]["text"])
            evidence += [{"title": title, "text": text}]

        system_json["evidence"] = evidence
        prompt_json[f"system {i+1}"] = system_json
    eval_input_json["question"] = json.dumps(prompt_json)
    return eval_input_json


def main(args):
    human_qid2example = get_qid2example(args.human_filename)
    sys1_qid2example = get_qid2example(args.system1_filename)
    sys2_qid2example = get_qid2example(args.system2_filename)

    eval_conv_ids = read_eval_conv_ids(args.eval_ids_filename)
        
    all_needed_pids = set()   # for memory saving purpose
    for qid in human_qid2example:
        if '_'.join(qid.split('_')[:-2]) not in eval_conv_ids:
            continue
        examples = [human_qid2example[qid], sys1_qid2example[qid], sys2_qid2example[qid]]
        for example in examples:
            for e in example["evidence"]:
                pid = e["passage_id"]
                all_needed_pids.add(pid)
    pid2info = get_pid2info(args.corpus_folder, all_needed_pids)

    eval_inputs = []
    for qid in human_qid2example:
        if '_'.join(qid.split('_')[:-2]) not in eval_conv_ids:
            continue
        systems = ['human', args.system1_name, args.system2_name]
        examples = [human_qid2example[qid], sys1_qid2example[qid], sys2_qid2example[qid]]
        eval_input = construct_eval_input(qid, systems, examples, pid2info)
        eval_inputs += [eval_input]

    print("total num examples: ", len(eval_inputs))

    with open(args.output_fname, 'w') as fout:
        for eval_input in eval_inputs:
            fout.write(json.dumps(eval_input)+'\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--corpus_folder", type=str, default='/data/zeqiuwu1/projects/InteractiveCQA/data/text_0420_processed')#'../../data/text_0420_processed')
    parser.add_argument("--eval_ids_filename", type=str, default='./sample/eval_ids.txt')
    parser.add_argument("--human_filename", type=str, default='./sample/human.json')
    parser.add_argument("--system1_filename", type=str, default='./sample/dialki_fid.json')
    parser.add_argument("--system2_filename", type=str, default='./sample/fid.json')
    parser.add_argument("--system1_name", type=str, default='dialki_fid')
    parser.add_argument("--system2_name", type=str, default='fid')
    parser.add_argument("--gold_filename", type=str, default='../../data/test.json')
    parser.add_argument("--output_fname", type=str, default='amt-human-eval/data/sources_eval.jsonl')
    args = parser.parse_args()
    main(args)