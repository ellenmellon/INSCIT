set -e

i=0
while [ $i -ne 10 ]
do
	CUDA_LAUNCH_BLOCKING=1 python generate_dense_embeddings.py model_file=retrieval_outputs/models/inscit/final_checkpoint ctx_src=dpr_wiki_inscit shard_id=$i num_shards=10 out_file=retrieval_data/encoded_wikipedia/wikipedia_passages batch_size=4608
	i=$(($i+1))
done
