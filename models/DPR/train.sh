python -m torch.distributed.launch --nproc_per_node=4 \
        train_dense_encoder.py \
        train_datasets=[inscit_train_all_history] \
        dev_datasets=[inscit_dev_all_history] \
        train=biencoder_inscit \
	fix_ctx_encoder=True \
        ignore_checkpoint_offset=True \
        ignore_checkpoint_optimizer=True \
        model_file=$(pwd)/retrieval_outputs/models/pretrain/final_checkpoint \
        output_dir=./retrieval_outputs/models/inscit
