exp_name=$1  # no_dialki or use_dialki
split=dev

if [ $exp_name == "no_dialki" ]; then
    n_context=50
    additional_arg="--use_pi_f1"
else
    n_context=4
    additional_arg=""
fi

CUDA_VISIBLE_DEVICES=0 python test_reader.py \
        --model_path ./reader_outputs/${exp_name}/checkpoint/best_dev \
        --eval_data ./reader_data/${exp_name}/${split}.json \
        --text_maxlength 384 \
        --answer_maxlength 100 \
        --per_gpu_batch_size 8 \
        --n_context ${n_context} \
        --checkpoint_dir ./reader_outputs/${exp_name} \
        --name ${split} \
        --write_results \
        --eval_print_freq 10 \
        ${additional_arg}
