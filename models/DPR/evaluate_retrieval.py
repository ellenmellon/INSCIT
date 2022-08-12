import json
from argparse import ArgumentParser
from collections import defaultdict

def hit_at_n(ranks, n):
    if len(ranks) == 0:
        return 0
    else:
        return len([x for x in ranks if x <= n]) * 100.0 / len(ranks)

def recall_at_n(ranks, n):
    return sum(ranks[n]) / len(ranks[n])

def main(data_file, results_file, metric):

    final_scores = {}
    with open(data_file, 'r') as f:
        raw_data = json.load(f)

    data = []
    for conv_id in raw_data:
        for i, turn in enumerate(raw_data[conv_id]["turns"]):
            turn_id = i+1
            positive_ctxs = set()
            for l in turn["labels"]:
                for p in l["evidence"]:
                    positive_ctxs.add(p["passage_id"])
            data += [{"conv_id": conv_id, "turn_id": turn_id, "positive_ctxs": list(positive_ctxs)}]

    with open(results_file, 'r') as f:
        results = json.load(f)

    assert len(data) == len(results)
    
    if metric == 'hit':
        ranks = []
    else:
        ranks = defaultdict(list)

    for i, sample in enumerate(data):

        assert str(sample["conv_id"]) == str(results[i]["conv_id"])
        assert str(sample["turn_id"]) == str(results[i]["turn_id"])
        
        gold_ctxs = sample["positive_ctxs"]
        if len(gold_ctxs) == 0:  # ignore examples with no gold passages
            continue
    
        if metric == 'hit':
            rank_assigned = False
        else:
            hits = 0
        for rank, ctx in enumerate(results[i]["ctxs"]):
            if ctx["id"] in gold_ctxs:
                if metric == 'hit':
                    ranks.append(float(rank + 1))
                    rank_assigned = True
                    break
                else:
                    hits += 1
            if metric == 'recall':
                ranks[rank+1] += [hits*100.0 / len(gold_ctxs)]
        
        if metric == 'hit' and not rank_assigned:
            ranks.append(1000.0)


    for n in [20, 50]:
        if metric == 'hit':
            score = hit_at_n(ranks, n)
        else:
            score = recall_at_n(ranks, n)
        final_scores[f"{metric}@{n}"] = round(score, 1)

    print(json.dumps(final_scores, indent=4))

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--data_file", type=str, default='../../data/dev.json')
    parser.add_argument("--results_file", type=str, default='retrieval_outputs/results/dpr_dev.json')
    parser.add_argument("--metric", type=str, default='hit', choices=['hit', 'recall'])
    args = parser.parse_args()
    main(args.data_file, args.results_file, args.metric)

