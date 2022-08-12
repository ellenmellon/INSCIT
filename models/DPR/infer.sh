
for split in train dev   # running for train set as well for the purpose of training readers
do
        python dense_retriever.py \
                model_file=$(pwd)/retrieval_outputs/models/inscit/final_checkpoint \
                qa_dataset=inscit_${split}_all_history \
                ctx_datatsets=[dpr_wiki_inscit] \
                encoded_ctx_files=[retrieval_data/encoded_wikipedia/wikipedia_passages_*] \
                out_file=retrieval_outputs/results/dpr_${split}.json
done