export WORLD_SIZE=4
export MASTER_ADDR="127.0.0.1" # change this and the following line if needed
export MASTER_PORT="10129" 
export OMP_NUM_THREADS=1

exp_name=$1  # no_dialki or use_dialki

if [ $exp_name == "no_dialki" ]; then
    additional_arg="--gold_passages_train reader_data/no_dialki/train_gold.json --use_pi_f1"
    n_context=50
else
    additional_arg=""
    n_context=4
fi


torchrun \
    --nproc_per_node $WORLD_SIZE \
    --standalone \
    --nnodes=1 \
    train_reader.py \
        --model_size base \
        --use_checkpoint \
        --lr 0.00005 \
        --optim adamw \
        --scheduler linear \
        --weight_decay 0.01 \
        --text_maxlength 384 \
        --answer_maxlength 100 \
        --per_gpu_batch_size 3 \
        --accumulation_steps 6 \
        --n_context ${n_context} \
        --total_step 800 \
        --warmup_step 50 \
        --eval_freq 50 \
        --checkpoint_dir reader_outputs/${exp_name} \
        --train_data reader_data/${exp_name}/train.json \
        --eval_data reader_data/${exp_name}/dev.json \
        ${additional_arg}
