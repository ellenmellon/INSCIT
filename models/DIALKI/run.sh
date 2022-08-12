set -e

python download_hf_model.py

python preprocess_data.py

python prepare_data.py


bash train.sh
bash infer.sh


python find_threshold.py # you need to record the printed out threshold

