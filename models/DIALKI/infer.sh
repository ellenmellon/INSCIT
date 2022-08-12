export WORLD_SIZE=1
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="10123"
export OMP_NUM_THREADS=1

data_name=inscit
split=dev_infer   # dev_infer is different from dev_infer that the candidate passages from retrieval, instead of from "1 gold positive + 49 negatives"
checkpoint_path=inscit/exp/best_em
prediction_results_path=inscit/exp/dev_infer_predictions.json

exp_name="exp"


base_dir=${data_name}
output_dir=${base_dir}/${exp_name}
mkdir -p $output_dir

max_seq_len=384
passages_per_question=50
max_answer_length=1 # one sentence only
hist_loss_weight=1.0


python -m torch.distributed.launch \
    --nproc_per_node $WORLD_SIZE \
    train_reader.py \
    --pretrained_model_cfg ${base_dir}/pretrained_models/bert-base-uncased \
    --seed 42 \
    --learning_rate 3e-5 \
    --eval_step 300 \
    --do_lower_case \
    --eval_top_docs 50 \
    --warmup_steps 1000 \
    --max_seq_len ${max_seq_len} \
    --batch_size 1 \
    --passages_per_question ${passages_per_question} \
    --num_train_epochs 20 \
    --dev_batch_size 2 \
    --max_answer_length ${max_answer_length} \
    --passages_per_question_predict 50 \
    --checkpoint_file $checkpoint_path \
    --prediction_results_file $prediction_results_path \
    --dev_file ${base_dir}/cache/${split} \
    --output_dir $output_dir \
    --gradient_accumulation_steps 1 \
    --ignore_token_type \
    --decision_function 1 \
    --hist_loss_weight ${hist_loss_weight} \
    --fp16 \
    --fp16_opt_level O2 \
    --data_name ${data_name} \
    --adv_loss_type js \
    --adv_loss_weight 5 \
    --use_z_attn
