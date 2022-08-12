from transformers import AutoTokenizer

def get_bert_reader_components(args, inference_only: bool = False, **kwargs):

    tokenizer = AutoTokenizer(args.pretrained_model_cfg)
    encoder = HFBertEncoder.init_encoder(args, len(tokenizer))

    reader = Reader(args, encoder)

    optimizer = (
        get_optimizer(
            reader,
            learning_rate=args.learning_rate,
            adam_eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
        if not inference_only
        else None
    )

    return tokenizer, reader, optimizer
