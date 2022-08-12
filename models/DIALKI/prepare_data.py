import argparse
from data_utils import InscitReader
from transformers import AutoTokenizer
from config import TOKENS

def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_dir)
    tokenizer.add_special_tokens(
        {
            'additional_special_tokens': TOKENS,
        }
    )

    reader = InscitReader(
        args,
        args.input_dir,
        args.output_dir,
        tokenizer, 
        args.max_seq_len,
        args.max_history_len,
        args.max_num_spans_per_passage,
        args.num_sample_per_file,
    )

    reader.convert_json_to_finetune_pkl('train')
    reader.convert_json_to_finetune_pkl('dev')

    reader.convert_json_to_finetune_pkl('train_infer')
    reader.convert_json_to_finetune_pkl('dev_infer')
    
    #reader.convert_json_to_finetune_pkl('test_infer')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pretrained_model_dir',
        default='inscit/pretrained_models/bert-base-uncased',
        type=str,
        help='',
    )
    parser.add_argument(
        '--input_dir',
        default='inscit/data',
        type=str,
        help='',
    )
    parser.add_argument(
        '--output_dir',
        default='inscit/cache',
        type=str,
        help='',
    )
    parser.add_argument(
        '--max_seq_len',
        default=384,
        type=int,
        help='',
    )
    parser.add_argument(
        '--max_history_len',
        default=128,
        type=int,
        help='',
    )
    parser.add_argument(
        '--max_num_spans_per_passage',
        default=2,
        type=int,
        help='',
    )
    parser.add_argument(
        '--num_sample_per_file',
        default=1000,
        type=int,
        help='',
    )
    parser.add_argument(
        '--use_sep_span_start',
        action='store_true',
        help='',
    )

    args = parser.parse_args()

    main(args)
