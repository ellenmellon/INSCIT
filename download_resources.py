#!/usr/bin/env python3

# This script is adapted from DPR CC-BY-NC 4.0 licensed repo (https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py)

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import pathlib
import wget

from typing import Tuple


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# TODO: replace google links to zenodo if the issue with large file upload gets solved
RESOURCE_MAP = {
    "wiki_corpus": {
        "url": "https://drive.google.com/uc?export=download&id=1rwQNmvJIRpBEs58ewcrxczZYPOa0nuQS&confirm=t&uuid=f05ab49b-df8e-4a2e-b551-56be01cfaad8",
        "filename": "corpus.zip",
        "desc": "All processed Wikipedia passages"
    },
    "bm25_results": {
        "url": ["https://zenodo.org/record/6983918/files/bm25_train.json", 
                "https://zenodo.org/record/6983918/files/bm25_dev.json", 
                "https://zenodo.org/record/6983918/files/bm25_grounding_train.json", 
                "https://zenodo.org/record/6983918/files/bm25_grounding_dev.json"],
        "filename": ["bm25_train.json", "bm25_dev.json", "bm25_grounding_train.json", "bm25_grounding_dev.json"],
        "desc": "All BM25 result files, used for creating negatives in training DPR"
    },
    "pretrained_dpr": {
        "url": "https://zenodo.org/record/6151076/files/model_checkpoints/retriever/dpr_retriever_all_history",
        "filename": "final_checkpoint",
        "desc": "Retriever DPR model checkpoint pretrained on TopioCQA"
    },
    "dpr": {
        "url": "https://drive.google.com/uc?export=download&id=1kYWSV1LK0SJn7x793bpgdVNEYKSJH5MQ&confirm=t&uuid=0190781c-89dc-49cd-bafb-91d331b3ccdb",
        "filename": "final_checkpoint",
        "desc": "Retriever DPR model checkpoint finetuned on INSCIT"
    },
    "dpr_results": {
        "url": ["https://zenodo.org/record/6983918/files/dpr_train.json", 
                "https://zenodo.org/record/6983918/files/dpr_dev.json"],
        "filename": ["dpr_train.json", "dpr_dev.json"],
        "desc": "All DPR result files"
    },
    "fid_no_dialki": {
        "url": "https://drive.google.com/uc?export=download&id=1Yhpvv9yeNw-bpQEcaOyWFxx9vp8kHUvf&confirm=t&uuid=71790cf5-5005-4bb1-ae1f-cf8f6662c674",
        "filename": "checkpoint.zip",
        "desc": "FiD model checkpoint finetuned on INSCIT, with top-50 DPR output as the model input"
    },
    "dialki": {
        "url": "https://zenodo.org/record/6983918/files/dialki",
        "filename": "best_em",
        "desc": "DIALKI model checkpoint finetuned on INSCIT"
    },
    "fid_use_dialki": {
        "url": "https://drive.google.com/uc?export=download&id=1DxnYE9UzwoOEjDifJnlcG2cjNGw3j_c6&confirm=t&uuid=95cebc92-2bdf-4654-8f7c-560e201eda57",
        "filename": "checkpoint.zip",
        "desc": "FiD model checkpoint finetuned on INSCIT, with DIALKI output as the model input"
    },
    "baseline_results": {
        "url": ["https://zenodo.org/record/6983918/files/fid_dev.json",
                "https://zenodo.org/record/6983918/files/dialki_fid_dev.json"],
        "filename": ["fid_dev.json", "dialki_fid_dev.json"],
        "desc": "Model predictions for two end tasks; input for the evaluation script."
    },
}


def download(out_dir: str, resource_name: str):
    resource_info = RESOURCE_MAP[resource_name]
    url = resource_info["url"]
    filename = resource_info["filename"]
    logger.info("Requested resource from %s", url)

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    if isinstance(url, str):
        url = [url]
        filename = [filename]

    for u, f in zip(url, filename):
        local_file = os.path.abspath(os.path.join(out_dir, f))
        logger.info("File to be downloaded as %s", local_file)

        if os.path.exists(local_file):
            logger.info("File already exist %s, skipping downloading ...", local_file)
            continue

        wget.download(u, out=local_file)
        logger.info("Downloaded to %s", local_file)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="The output directory to download file",
    )
    parser.add_argument(
        "--resource_name",
        required=True,
        type=str,
        help="Name of the resource to be downloaded",
    )
    print('hi')
    args = parser.parse_args()
    download(args.output_dir, args.resource_name)

if __name__ == "__main__":
    main()
