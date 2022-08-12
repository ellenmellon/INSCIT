# INSCIT
This repo provides the data and code for evaluation as well as baseline models for the paper: [INSCIT: Information-Seeking Conversations with Mixed-Initiative Interactions](https://arxiv.org/pdf/2207.00746.pdf)

## Content
1. [Data](#data)
2. [Baseline Models](#baseline-models)
    * [Set Up](#set-up)
    * [DPR Retriever](#dpr-retriever)
    * [FiD Reader](#fid-reader)
    * [DIALKI + FiD Reader](#dialki-and-fid-reader)
3. [Evaluation](#evaluation)
    * [Automatic Evaluation](#automatic-evaluation)
    * [Human Evaluation](#human-evaluation)
4. [Citation](#citation)

## Data
We provide the train and dev sets of INSCIT at the `./data` folder. Test set is currently being hid for the purpose of a potential leaderboard, and **we will post updates soon**!

Each data file contains task examples in each conversation, where each example maps to an agent turn and has the following format. "*context*" contains the dialogue history of alternating user and agent utterances. "*prevEvidence*" contains information of evidence passages used in the previous agent turns. "*labels*" contains annotated agent turns with three elements: response type, response utterance and evidence passages. Each example has one or two annotations in the "*labels*" field. Annotation details can be found in Section 4 in the paper. Each evidence passage comes with "*passage_id*" (format: `Wiki_article_name:passage_position`), "*passage_text*" and "*passage titles*" (document title and all parent section titles). We also provide information of the seed article used for triggering the start of each conversation. Example:
```
{
    "context": [
      "Who scored the winning goal for Miracle on Ice?",
      "Mark Pavelich passed to Eruzione, who was left undefended in the high slot ...",
      "Did anything else happen in the game during the remaining 10 minutes?"
    ],
    "prevEvidence": [
      [
        {
          "passage_id": "Miracle on Ice:22",
          "passage_text": "...",
          "passage_titles": [
            "Miracle on Ice",
            "Game summary",
            "Third period"
          ]
        }
      ]
    ],
    "labels": [
      {
        "responseType": "directAnswer",
        "response": "The Soviets, trailing for the first time in the game, attacked ferociously ...",
        "evidence": [
          {
            "passage_id": "Miracle on Ice:23",
            "passage_text": "...",
            "passage_titles": [
              "Miracle on Ice",
              "Game summary",
              "Third period"
            ]
          }
        ]
      },
    ]
}
```

To download the corpus containing all processed passages in Wikipedia articles (as of 04/20/2022), simply run:
```
pip install wget
python download_resources.py --output_dir=./data --resource_name=wiki_corpus
unzip data/corpus.zip -d data
```
The corpus will be saved at `./data`.

We used [wikiextractor](https://github.com/attardi/wikiextractor) for data processing. We preserve the content outline of each article and make it a passage in the corpus, which always has the passage id as `Wiki_article_name:0`.



## Baseline Models
Our two baselines are based on [DPR](https://github.com/facebookresearch/DPR), [FiD](https://github.com/facebookresearch/FiD) and [DIALKI](https://github.com/ellenmellon/DIALKI) models.


### Set Up
Our code is tested on cuda 11.6 and A40 GPUs. Each model has their own python environment, with details in the following sections.

Run `bash setup.sh` to create all data / model / results folders for later uses.

- - -

### DPR Retriever
To set up the environment, run 
```
conda env create -f environment_dpr_fid.yml
conda activate dpr_fid
cd models/DPR
```

#### Data Preparation
First, download BM25 (we used [Pyserini](https://github.com/castorini/pyserini)) results, used as hard negatives for DPR training: 
```
python ../../download_resources.py --output_dir=./retrieval_outputs/results --resource_name=bm25_results
```

To prepare training and inference input data:
```
python create_corpus_tsv.py
python prepare_data.py
```

#### Training
Our model is initialized with pretraining on TopioCQA[], which is >30x larger than our training data.
Download the pretrained checkpoint from the original TopioCQA repo:
```
python ../../download_resources.py --output_dir=./retrieval_outputs/models/pretrain --resource_name=pretrained_dpr
```
Then, simply run the following to start training:
```
bash train.sh
```

#### Inference
To do inference, run
```
bash encode_passages.sh   # this can take > 20 hours to finish
bash infer.sh  # we do infernece for both train and dev sets, required for training reader models
```

To check retrieval performance, run `python evaluate_retrieval.py`.

Our own finetuned DPR model checkpoint can be downloaded by running the following line. However, note that if you don't want your own trained model to be overwritten, change the `--output_dir` argument accordingly.
```
python ../../download_resources.py --output_dir=./retrieval_outputs/models/inscit --resource_name=dpr
```
If you change the `--output_dir` value above, you also need change the `--model_file` in both `encode_passages.sh` and `infer.sh` for inference with our provided DPR checkpoint.

Our own retrieval output files (for both train and dev) can be downloaded by running:
```
python ../../download_resources.py --output_dir=./retrieval_outputs/results --resource_name=dpr_results
```

- - -

### FiD Reader
To set up the environment, run 
```
conda activate dpr_fid
cd models/FiD
```

#### Data Preparation
```
python prepare_data_no_dialki.py
```

#### Training
```
bash train.sh no_dialki
```

#### Inference
```
bash infer.sh no_dialki
python prepare_eval_file_no_dialki.py    # convert the FiD output file into the evaluation script (details below) input format
```

Our own finetuned FiD model checkpoint can be downloaded by running the following line. However, note that if you don't want your own trained model to be overwritten, change the `unzip` target path accordingly.
```
python ../../download_resources.py --output_dir=./reader_outputs/no_dialki --resource_name=fid_no_dialki
unzip ./reader_outputs/no_dialki/checkpoint.zip -d ./reader_outputs/no_dialki
```
If you change the `unzip` target path above, you also need change the `--model_file` in `infer.sh`.

- - -

### DIALKI and FiD Reader
This pipelined approach starts with the DIALKI model to perform passage identification. To set up the environment for DIALKI, run:
```
conda env create -f environment_dialki.yml
conda activate dialki
cd models/DIALKI
```

#### Data Preparation & Training

##### DIALKI
```
# Preparation

python download_hf_model.py
python preprocess_data.py
python prepare_data.py


# Training

bash train.sh
```

Run the following to get output results from DIALKI, that will be used for the next stage FiD model.
```
bash infer.sh  # don't panic if you see all printed out numbers are 0.0
```

Our own finetuned DIALKI model checkpoint can be downloaded by running the following line. However, note that if you don't want your own trained model to be overwritten, change the `--output_dir` argument accordingly.
```
python ../../download_resources.py --output_dir=./inscit/exp --resource_name=dialki
```
If you change the `--output_dir` value above, you also need change `checkpoint_path` in `infer.sh`.

Then, you need to find the threshold $\gamma$ (details in paper Section 5) for multi passage prediction.
```
python find_threshold.py
```
Record the threshold and change the corresponding parameter values in the following FiD scripts. If you use our own finetuned DIALKI model to do inference, you can simply use the default value we set in the following scripts.


##### FiD

Switch to FiD environment:
```
conda activate dpr_fid
cd ../FiD
```

Prepare data:
```
python prepare_data_use_dialki.py    # add --threshold argument accordingly
```

Model training:
```
bash train.sh use_dialki
```


#### Inference

```
bash infer.sh use_dialki
python prepare_eval_file_use_dialki.py    # add --threshold argument accordingly
```

Our own finetuned FiD model checkpoint can be downloaded by running the following line. However, note that if you don't want your own trained model to be overwritten, change the `unzip` target path accordingly.
```
python ../../download_resources.py --output_dir=./reader_outputs/use_dialki --resource_name=fid_use_dialki
unzip ./reader_outputs/use_dialki/checkpoint.zip -d ./reader_outputs/use_dialki
```
If you change the `unzip` target path above, you also need change the `--model_file` in `infer.sh`.


## Evaluation 
We provide automatic evaluation script and instructions of how to set up human evaluation for both passage identification and response generation tasks.

### Automatic Evaluation
Go to the root directory and running the following prints out automatic evaluation scores of both tasks for the above two models:
```
python eval/eval.py --results_file results/fid_dev.json
python eval/eval.py --results_file results/dialki_fid_dev.json
```

To download our own model prediction files (change `--output_dir` if you do not want to overwrite your own prediction files, and change `--results_file` above accordingly):
```
python download_resources.py --output_dir=./results --resource_name=baseline_results
```


### Human Evaluation

To be updated!


## Citation
```
@misc{https://doi.org/10.48550/arxiv.2207.00746,
  doi = {10.48550/ARXIV.2207.00746},
  url = {https://arxiv.org/abs/2207.00746},
  author = {Wu, Zeqiu and Parish, Ryu and Cheng, Hao and Min, Sewon and Ammanabrolu, Prithviraj and Ostendorf, Mari and Hajishirzi, Hannaneh},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {INSCIT: Information-Seeking Conversations with Mixed-Initiative Interactions},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
