import logging

import torch
from torch import nn
import transformers as tfs
from transformers.models.bert import modeling_bert


logger = logging.getLogger(__name__)


class HFBertEncoder(modeling_bert.BertModel):
    def __init__(self, config, coordinator_config, args):
        modeling_bert.BertModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"

        if args.use_coordinator:
            self.coordinator = modeling_bert.BertEncoder(coordinator_config)
            # assumes number of docs does not exceed 30
            self.doc_pos_embeddings = nn.Embedding(30, config.hidden_size)
            self.coord_head_mask = [
                None for i in range(coordinator_config.num_hidden_layers)]
        else:
            self.coordinator = None

        if args.projection_dim != 0:
            self.encode_proj = nn.Linear(
                config.hidden_size, args.projection_dim)
        else:
            self.encode_proj = None
        
        self.activation = nn.Tanh()
        self.init_weights()

    @property
    def output_dim(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size

    @classmethod
    def init_encoder(cls, args, vocab_size):

        cfg = tfs.AutoConfig.from_pretrained(args.pretrained_model_cfg)
        coordinator_cfg = modeling_bert.BertConfig.from_pretrained(
            'bert-base-uncased')
        coordinator_cfg.num_hidden_layers = args.coordinator_layers
        coordinator_cfg.num_attention_heads = args.coordinator_heads

        dropout = args.dropout if hasattr(args, 'dropout') else 0.0
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
            coordinator_cfg.attention_probs_dropout_prob = dropout
            coordinator_cfg.hidden_dropout_prob = dropout

        encoder = cls.from_pretrained(
            args.pretrained_model_cfg,
            config=cfg,
            coordinator_config=coordinator_cfg,
            args=args)

        if cfg.vocab_size != vocab_size:
            logger.info(f"Resize embedding from {cfg.vocab_size} to {vocab_size}")
            encoder.resize_token_embeddings(vocab_size)

        # Hacky way to duplicate position embeddings.
        if args.max_seq_len > 512:
            my_pos_embeddings = nn.Embedding(args.max_seq_len, cfg.hidden_size)
            my_pos_embeddings.weight.data[:512] = encoder.embeddings.position_embeddings.weight.data
            n_assigned = 512
            while n_assigned < args.max_seq_len:
                next_n_assigned = min(n_assigned+512, args.max_seq_len)
                my_pos_embeddings.weight.data[n_assigned:next_n_assigned] = encoder.embeddings.position_embeddings.weight.data[:next_n_assigned-n_assigned,:]
                n_assigned = next_n_assigned
            encoder.embeddings.position_embeddings = my_pos_embeddings

        if args.num_token_types > 2:
            my_type_embeddings = nn.Embedding(
                args.num_token_types, cfg.hidden_size)
            my_type_embeddings.weight.data[:] = encoder.embeddings.token_type_embeddings.weight.data[0][None,:].repeat(args.num_token_types,1)
            encoder.embeddings.token_type_embeddings = my_type_embeddings

        encoder.embeddings.register_buffer(
            "position_ids", torch.arange(args.max_seq_len).expand(1, -1))

        return encoder

    def forward(
        self,
        N,
        M,
        input_ids,
        token_type_ids,
        position_ids,
        attention_mask,
        inputs_embeds=None,
    ):
        if self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            )
        else:
            hidden_states = None
            sequence_output, pooled_output = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            )
        # (N * M, 1, hidden_size).
        pooled_output = sequence_output[:, 0, :]
        pooled_output = self.coordinate(
            N, M, pooled_output.unsqueeze(1), position_ids)
        return sequence_output, pooled_output, hidden_states
    
    def coordinate(self, N, M, pooled_output, position_ids):
        # (N * M, l, hidden_size).
        hidden_size = pooled_output.size(-1)
        L = pooled_output.size(1)
        if self.coordinator:
            # (N * M, l, hidden_size). => (l, N * M, hidden_size).
            pooled_output = pooled_output.transpose(0, 1)
            # (l, N * M, hidden_size). => (l*N, M, hidden_size).
            pooled_output = pooled_output.view(-1, M, hidden_size)
            
            if self.doc_pos_embeddings:
                # (N, M) => (N, M, hidden_size). => (l*N, M, hidden_size).
                doc_position_embeddings = self.doc_pos_embeddings(
                    position_ids).repeat(L, 1, 1)

                # (l*N, M, hidden_size). => (l*N, M, hidden_size). 
                pooled_output = pooled_output + doc_position_embeddings
            
            pooled_output = self.coordinator(
                pooled_output, head_mask=self.coord_head_mask)[0]
            # (l*N, M, hidden_size). => (l, N * M, hidden_size).
            pooled_output = pooled_output.view(-1, N * M, hidden_size)
            # (l, N * M, hidden_size). => (N * M, l, hidden_size).
            pooled_output = pooled_output.transpose(0, 1)
        if self.encode_proj:
            pooled_output = self.activation(self.encode_proj(pooled_output))
        return pooled_output
