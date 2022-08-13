import json
from argparse import ArgumentParser


def convert_score(score, four_scale=False):
    if score == -10:    # No evidence labeled or no information in the response
        score = 2
    if four_scale:
        score_map = {2: 5, 1: 1+4.0/3*2, -1: 1+4.0/3, -2: 1}
        return score_map[score]
    else:
        return score + 3


def get_average_scores(hits, sysnames):
    """Get scores for a single example, averaged over multiple evaluators"""

    if len(hits) == 0:
        return 0, None
    
    score_map = {}
    for sysname in sysnames:
        score_map[sysname] = {"utility": 0.0, "consistency": 0.0, "coherence": 0.0}
    
    n = 0
    for hit in hits:
        annotation = hit["response"]["annotations"][0]
        for i, name in enumerate(annotation["system names"]):
            score_map[name]["utility"] += convert_score(annotation[f"utility{i+1}"])
            score_map[name]["consistency"] += convert_score(annotation[f"consistency{i+1}"])
            score_map[name]["coherence"] += convert_score(annotation[f"coherence{i+1}"], four_scale=True)
        n += 1

    for key1, value1 in score_map.items():
        for key2, value2 in value1.items():
            score_map[key1][key2] = value2 / n
    return n, score_map


def get_comprehensiveness_ranks(hits, sysnames):
    """Get comprehensiveness ranks for a single example, averaged over multiple evaluators"""

    if len(hits) == 0:
        return 0, None
    
    score_map = {}
    for sysname in sysnames:
        score_map[sysname] = 0

    n = 0
    for hit in hits:
        annotation = hit["response"]["annotations"][0]
        total_most = sum([int(annotation[f"most_comprehensive_{i+1}"]) for i in range(3)])
        total_least = sum([int(annotation[f"least_comprehensive_{i+1}"]) for i in range(3)])
        assert total_most >= 1
        assert total_least + total_most <= 3
        for i, name in enumerate(annotation["system names"]):
            if annotation[f"most_comprehensive_{i+1}"]:
                score_map[name] += 1    # (tied) rank 1
            elif annotation[f"least_comprehensive_{i+1}"]:
                if total_most + total_least == 3:
                    score_map[name] += 2   # tied rank 2
                else:
                    score_map[name] += 3   # rank 3
            else:
                score_map[name] += 2    # rank 2
        n += 1

    for key1, value1 in score_map.items():
        score_map[key1] = value1 / n
    return n, score_map    


def get_scores(json_object, sysnames):
    """Get Likert scores for utility, consistency and coherence ratings."""

    score_map = {}
    for sysname in sysnames:
        score_map[sysname] = {"utility": 0.0, "consistency": 0.0, "coherence": 0.0}
    
    total_n = 0
    for key, hits in json_object.items():
        tmp_n, tmp_score_map = get_average_scores(hits, sysnames)
        if tmp_n == 0:
            continue
        for key1, value1 in tmp_score_map.items():
            for key2, value2 in value1.items():
                score_map[key1][key2] += value2
        total_n += 1
    
    for key1, value1 in score_map.items():
        for key2, value2 in value1.items():
            score_map[key1][key2] = value2 / total_n

    print(json.dumps(score_map, indent=2))


def get_comparison_results(json_object, sysnames):
    """Get system pair comparison scores for all dimension ratings."""
    
    score_map = {}
    for i, sys1 in enumerate(sysnames):
        for sys2 in sysnames[i+1:]:
            pair_name = f"{sys1}_{sys2}"
            score_map[pair_name] = {}
            for key in ["utility", "consistency", "coherence", "comprehensiveness"]:
                score_map[pair_name][key] = {}
                for status in ["win", "tie", "lose"]:
                    score_map[pair_name][key][status] = 0

    for qid, hits in json_object.items():
        tmp_n, tmp_score_map = get_average_scores(hits, sysnames)
        if tmp_n == 0:
            continue
        _, tmp_comp_map = get_comprehensiveness_ranks(hits, sysnames)
        for key in tmp_score_map:
            tmp_score_map[key]["comprehensiveness"] = tmp_comp_map[key]
        for pair in [("human", "dialki"), ("human", "fid"), ("dialki", "fid")]:
            for key in ["utility", "consistency", "coherence", "comprehensiveness"]:
                if tmp_score_map[pair[0]][key] > tmp_score_map[pair[1]][key]:
                    if key != "comprehensiveness":
                        status = "win"
                    else:
                        status = "lose"
                    score_map["_".join(pair)][key][status] += 1
                elif tmp_score_map[pair[0]][key] == tmp_score_map[pair[1]][key]:
                    score_map["_".join(pair)][key]['tie'] += 1
                else:
                    if key != "comprehensiveness":
                        status = "lose"
                    else:
                        status = "win"
                    score_map["_".join(pair)][key][status] += 1
    
    print(json.dumps(score_map, indent=2))


def get_system_names(json_object):
    return list(json_object.values())[0][0]["response"]["annotations"][0]["system names"]


def main(args):
    with open(args.eval_output_file) as fin:
        json_object = json.loads(fin.read())
    sysnames = get_system_names(json_object)
    get_scores(json_object, sysnames)
    get_comparison_results(json_object, sysnames)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--eval_output_file", type=str, default='amt-human-eval/data/example/main/live/validationPromptIdToAssignments.txt')
    args = parser.parse_args()
    main(args)