import logging

import torch
import torch.nn as nn
from torch import Tensor as T

from data_utils import utils as d_utils
from models import loss
from models import perturbation
from utils import model_utils

logger = logging.getLogger()


def _pad_to_len(seq: T, pad_id: int, max_len: int):
    s_len = seq.size(0)
    if s_len > max_len:
        return seq[0: max_len]
    return torch.cat([seq, torch.Tensor().new_full(
        (max_len - s_len,), pad_id, dtype=torch.long).to(seq.device)], dim=0)


class Reader(nn.Module):

    def __init__(self, args, encoder):
        super(Reader, self).__init__()
        self.args = args
        self.encoder = encoder

        hidden_size = encoder.config.hidden_size
        self.hidden_size = hidden_size

        if args.compute_da_loss:
            # TODO: replace hard-coded da label number
            self.da_classifier = nn.Linear(hidden_size, 7)
            self.segment_transform_da = nn.Linear(hidden_size, hidden_size)
            self.history_da_classifier = nn.Linear(hidden_size, 7)
            model_utils.init_weights(
                [self.da_classifier,
                 self.history_da_classifier,
                 self.segment_transform_da])
        
        to_init = []
        if args.hist_loss_weight > 0:
            self.segment_transform_start = nn.Linear(hidden_size, hidden_size)
            self.segment_transform_end = nn.Linear(hidden_size, hidden_size)
            self.segment_transform_relevance = nn.Linear(
                hidden_size, hidden_size)
            self.history_qa_classifier = nn.Linear(hidden_size, 1)

            to_init.extend(
                [
                    self.history_qa_classifier,
                    self.segment_transform_start,
                    self.segment_transform_end,
                    self.segment_transform_relevance,
                ]
            )
        
        if args.user2agent_loss_weight > 0:
            self.user2agent_transform_start = nn.Linear(
                hidden_size, hidden_size)
            self.user2agent_transform_end = nn.Linear(hidden_size, hidden_size)
            self.user2agent_transform_relevance = nn.Linear(
                hidden_size, hidden_size)
            self.user2agent_qa_classifier = nn.Linear(hidden_size, 1)

            to_init.extend(
                [
                    self.user2agent_transform_start,
                    self.user2agent_transform_end,
                    self.user2agent_transform_relevance,
                    self.user2agent_qa_classifier,
                ]
            )

        if args.span_marker:
            assert args.decision_function == 3
            self.marker_embs = nn.Embedding(2, hidden_size)
            to_init.append(self.marker_embs)

        if to_init:
            model_utils.init_weights(to_init)

        if (args.decision_function == 0
            or args.decision_function == 2
            or args.decision_function == 3):
            self.qa_outputs = nn.Linear(hidden_size, 2)
            self.qa_classifier = nn.Linear(hidden_size, 1)
            
        elif args.decision_function == 1:
            self.W_k_u = nn.Linear(hidden_size, hidden_size)
            self.W_v_u = nn.Linear(hidden_size, hidden_size)
            
            self.W_k_a = nn.Linear(hidden_size, hidden_size)
            self.W_v_a = nn.Linear(hidden_size, hidden_size)
            
            model_utils.init_weights([self.W_k_u, self.W_v_u, 
                          self.W_k_a, self.W_v_a])
            self.dropout = nn.Dropout(args.dropout)
            self.softmax = nn.Softmax(dim=-1)

            if self.args.use_z_attn:
                self.W_u_a = nn.Linear(hidden_size, hidden_size)
                self.W_u_u = nn.Linear(hidden_size, hidden_size)
                model_utils.init_weights([self.W_u_u, self.W_u_a])
            
            self.qa_classifier = nn.Linear(hidden_size, 1)

            self.qa_outputs = nn.Linear(3 * hidden_size, 2)

        else:
            raise NotImplementedError(
                f'unknown decision function {args.decision_function}')

        model_utils.init_weights([self.qa_outputs, self.qa_classifier])

        self._setup_adv_training()

    def _setup_adv_training(self):
        self.adv_teacher = None
        if self.args.adv_loss_weight > 0:
            self.adv_teacher = perturbation.SmartPerturbation(
                self.args.adv_epsilon,
                self.args.adv_step_size,
                self.args.adv_noise_var,
                self.args.adv_norm_p,
                self.args.adv_k,
                norm_level=self.args.adv_norm_level)


    def forward(
        self,
        batch,
        global_step,
        fwd_type='input_ids',
        inputs_embeds=None,
        end_task_only=False):

        # Notations:
        # (1) N - number of questions in a batch.
        # (2) M - number of passages per questions.
        # (3) L - sequence length.
        N, M, L = batch['input_ids'].size()
        N, M, Ls = batch['clss'].size()

        # TODO: check question_broundaries when not training
        question_boundaries = batch['question_boundaries'].view(N * M, -1, 2)
        if 'user_starts' in batch:
            user_starts = batch['user_starts'].view(N * M, -1, 1)

        input_ids = batch['input_ids'].view(N*M, L)
        passage_positions = batch['passage_positions'].view(N, M)

        if batch['attention_mask'].dim() == 4:
            attention_mask = batch['attention_mask'].view(N*M, L, L)
        else:
            attention_mask = batch['attention_mask'].view(N*M, L)
        clss = batch['clss'].view(N*M, Ls)
        ends = batch['ends'].view(N*M, Ls)
        mask_cls = batch['mask_cls'].view(N*M, Ls)
        type_ids = batch['type_ids'].view(N*M, L)

        if self.args.ignore_token_type:
            encoder_type_ids = None
        else:
            encoder_type_ids = batch['type_ids'].view(N*M, L)

        if fwd_type == 'get_embs':
            # Gets embeddings only.
            assert inputs_embeds is None
            return self.encoder.embeddings(input_ids, encoder_type_ids)
        else:
            # Skips input_ids to inputs_embeds.
            if fwd_type == 'inputs_embeds':
                assert inputs_embeds is not None
                _input_ids = None
            elif fwd_type == 'input_ids':
                _input_ids = input_ids
            else:
                raise ValueError(f'fwd_type = {fwd_type} is not available')
                
            sequence_output, _pooled_output, _hidden_states = self.encoder(
                N,
                M,
                _input_ids,
                encoder_type_ids,
                passage_positions,
                attention_mask,
                inputs_embeds)

        # TODO: use batched_index_select
        span_start_embs = sequence_output[
            torch.arange(sequence_output.size(0)).unsqueeze(1), clss]

        logits = {}
        if self.args.decision_function == 0:
            logits.update(
                self._forward_df0(
                    N,
                    M,
                    sequence_output,
                    _pooled_output,
                    span_start_embs,
                    question_boundaries,
                    passage_positions,
                    end_task_only))
        elif self.args.decision_function == 1:
            logits.update(
                self._forward_df1(
                    N,
                    M,
                    sequence_output,
                    _pooled_output,
                    type_ids,
                    span_start_embs,
                    question_boundaries,
                    passage_positions,
                    end_task_only))
        elif self.args.decision_function == 2:
            logits.update(
                self._forward_df2(
                    N,
                    M,
                    sequence_output,
                    _pooled_output,
                    span_start_embs,
                    question_boundaries,
                    passage_positions,
                    user_starts,
                    end_task_only))
        elif self.args.decision_function == 3:
            if self.args.adv_loss_weight > 0:
                raise NotImplementedError

            logits.update(
                self._forward_df3(
                    N,
                    M,
                    sequence_output,
                    _pooled_output,
                    span_start_embs,
                    question_boundaries,
                    passage_positions,
                    mask_cls,
                    global_step))

        logits['start'] = logits['start'].view(N, M, Ls)
        logits['end'] = logits['end'].view(N, M, Ls)
        logits['relevance'] = logits['relevance'].view(N, M)

        others = {}
        if not end_task_only and self.args.adv_loss_weight > 0:
            adv_logits, emb_val, eff_perturb = self.adv_forward(
                batch,
                logits,
                global_step,
                self.args.adv_calc_logits_keys)
            logits.update(adv_logits)
            others['emb_val'] = emb_val
            others['eff_perturb'] = eff_perturb

        return logits, others

    def adv_forward(self, batch, logits, global_step, calc_logits_keys):
        assert self.adv_teacher is not None
        adv_logits, emb_val, eff_perturb = self.adv_teacher.forward(
            self,
            logits,
            batch,
            global_step,
            calc_logits_keys)
        return adv_logits, emb_val, eff_perturb

    def _forward_df0(
        self, 
        N, 
        M, 
        sequence_output, 
        _pooled_output, 
        span_start_embs, 
        question_boundaries, 
        passage_positions,
        end_task_only):
        
        logits = {}
        
        if self.training and not end_task_only:
            start = self._get_question_boundary_start(question_boundaries)
            question_segments = model_utils.batched_index_select(
                sequence_output, 1, start)
            logits.update(
                self._calc_history_logits(
                    N, M, sequence_output, span_start_embs,
                    start, passage_positions))
            logits.update(
                self._calc_da_logits(question_segments, sequence_output))

        logits['start'], logits['end'] = self.qa_outputs(
            span_start_embs).split(1, dim=-1)
        logits['relevance'] = self.qa_classifier(_pooled_output)

        return logits   

    def _forward_df2(
        self,
        N,
        M,
        sequence_output,
        _pooled_output,
        span_start_embs,
        question_boundaries,
        passage_positions,
        user_starts,
        end_task_only):
        logits = self._forward_df0(N, M, sequence_output, _pooled_output,
                                   span_start_embs, question_boundaries,
                                   passage_positions)

        if self.training and not end_task_only:
            # _get_question_boundary_start takes two idxs
            start = self._get_question_boundary_start(user_starts)
            logits.update(
                self._calc_user2agent_logits(N, M, sequence_output,
                                             span_start_embs, start,
                                             passage_positions))

        return logits

    def _forward_df3(
        self,
        N,
        M,
        sequence_output,
        _pooled_output,
        span_start_embs,
        question_boundaries,
        passage_positions,
        mask_cls,
        global_step,
    ):
        logits = {}
        
        assert self.args.hist_loss_weight > 0

        start = self._get_question_boundary_start(question_boundaries)
        question_segments = model_utils.batched_index_select(
            sequence_output, 1, start)
        logits.update(
            self._calc_history_logits(N, M, sequence_output, span_start_embs,
                                      start, passage_positions))
        logits.update(self._calc_da_logits(question_segments, sequence_output))

        Ls = mask_cls.size(-1)
        mask_cls = mask_cls.view(N, M, Ls)
        max_num_history_questions = logits['history_relevance'].size(1)
        device = mask_cls.device

        history_start_logits = logits['history_start'].view(N, M, -1, Ls)
        history_end_logits = logits['history_end'].view(N, M, -1, Ls)

        # before: history_relevance_logits size = (N*M, max_num_history_questions, 1)
        # after:  history_relevance_logits size = (N*max_num_history_questions, M, 1)
        history_relevance_logits = torch.cat([t.transpose(0, 1) for t in logits['history_relevance'].split(M, dim=0)], dim=0)

        # idxs size = (N*max_num_history_questions, M, 1)
        _, idxs = torch.sort(history_relevance_logits, dim=1, descending=True)
        top1 = idxs[:, 0].view(N, max_num_history_questions)

        marker_idxs = torch.zeros((N, M, Ls), dtype=torch.long)

        if global_step >= self.args.marker_after_steps:
            for n in range(N):
                for hq in range(max_num_history_questions):
                    if self.args.skip_mark_last_user and hq == 0:
                        continue
                    passage_idx = top1[n][hq]
                    p_start_logits = history_start_logits[n, passage_idx].tolist()
                    p_end_logits = history_end_logits[n, passage_idx].tolist()
                    start_index, end_index, _ = next(d_utils.start_end_finder(p_start_logits, p_end_logits, self.args.max_answer_length, None, mask_cls[n, passage_idx]))
                    if start_index != -1 and end_index != -1:
                        h = torch.arange(Ls)
                        h = ((h >= start_index) & (h <= end_index)).long()
                        marker_idxs[n, passage_idx] += h

        marker_idxs.clamp_(0, 1)
        marker_idxs = marker_idxs.to(device)

        # mark_embs size = (N, M, Ls, hidden_size)
        marker_embs = self.marker_embs(marker_idxs).view(N*M, Ls, -1)

        logits['start'], logits['end'] = self.qa_outputs(
            span_start_embs+marker_embs).split(1, dim=-1)
        logits['relevance'] = self.qa_classifier(_pooled_output)

        return logits

    def _forward_df1(
        self,
        N, 
        M, 
        sequence_output, 
        _pooled_output, 
        type_ids, 
        span_start_embs, 
        question_boundaries, 
        passage_positions,
        end_task_only,
    ):

        logits = {}

        start = self._get_question_boundary_start(question_boundaries)
        question_types = torch.gather(type_ids, 1, start)
        # question_segments size = (N*M, max_num_history_questions, 768)
        question_segments = model_utils.batched_index_select(
            sequence_output, 1, start)

        if self.training and not end_task_only:
            logits.update(
                self._calc_history_logits(N, M, sequence_output,
                                          span_start_embs, start,
                                          passage_positions))
            logits.update(
                self._calc_da_logits(question_segments, sequence_output))


        ### start next turn logits calculation
        span_embs_user = self._calc_ctx_span_emb_by_role_history(
            question_types, question_segments, _pooled_output,
            span_start_embs, is_agent=False)
        span_embs_agent = self._calc_ctx_span_emb_by_role_history(
            question_types, question_segments, _pooled_output,
            span_start_embs, is_agent=True)

        s = torch.cat(
            (span_start_embs, span_embs_user, span_embs_agent), dim=-1)
        
        logits['start'], logits['end'] = self.qa_outputs(s).split(1, dim=-1)
        
        logits['relevance'] = self.qa_classifier(_pooled_output)

        return logits


    def _calc_ctx_span_emb_by_role_history(
        self,
        question_types,
        question_segments,
        _pooled_output,
        span_start_embs,
        is_agent=True
    ):
        # Calculates question_segments and start_mask for user or agent.
        # role_segments size = (N*M, max_role_questions, 768).
        # role_start_index size = (N*M, max_role_questions)
        # list of index tensor of role turns for each passage.
        qtype = 2 if is_agent else 1

        role_start_index = [
            (question_types[i, :] == qtype).nonzero(as_tuple=True)[0]
            for i in range(question_segments.size(0))]
        max_role_questions = max([t.size(0) for t in role_start_index])
        role_start_index = torch.stack(
            [_pad_to_len(t, -1, max_role_questions) for t in role_start_index],
            dim=0)
        role_start_mask = role_start_index == -1
        role_start_index = role_start_index.masked_fill(role_start_mask, 0)
        role_segments = model_utils.batched_index_select(
            question_segments, 1, role_start_index)
        
        # k: (N*M, max_n_spans, hid)
        k = span_start_embs
        _, max_n_spans, _ = k.size()
        v = k
        n_questions = max_role_questions
        if not is_agent:
            n_questions = min(max_role_questions, 2)
        mask = ~role_start_mask

        for i in range(n_questions):

            idx = n_questions - i - 1
            # u: (N*M, 1, hid)
            u = role_segments[:, idx:idx+1, :]
            extended_pooled = _pooled_output.repeat(1, max_n_spans, 1)
            extended_u = u.repeat(1, max_n_spans, 1)
                
            if is_agent:
                if self.args.use_z_attn:
                    _v = torch.relu(self.W_k_a(self.dropout(k))
                                    + self.W_v_a(self.dropout(extended_pooled))
                                    + self.W_u_a(self.dropout(extended_u)))
                else:
                    _v = torch.relu(self.W_k_a(self.dropout(k))
                                    + self.W_v_a(self.dropout(extended_pooled)))
            else:
                if self.args.use_z_attn:
                    _v = torch.relu(self.W_k_u(self.dropout(k))
                                    + self.W_v_u(self.dropout(extended_pooled))
                                    + self.W_u_u(self.dropout(extended_u)))
                else:
                    _v = torch.relu(self.W_k_u(self.dropout(k))
                                    + self.W_v_u(self.dropout(extended_pooled)))
            
            # q = 1; g: (N*M, 1, max_n_spans)
            if self.args.use_z_attn:
                g = torch.sigmoid(
                    torch.einsum('bqh,bsh->bqs', u, extended_pooled)
                    + torch.einsum('bqh,bsh->bqs', u, k))
            else:
                g = torch.sigmoid(
                    torch.einsum('bqh,bsh->bqs', u, extended_pooled))
            # set g to be 0 on padded turns
            g = g.mul(
                mask.type(g.type())[:, idx:idx+1, None].repeat(1, 1, max_n_spans))
            g = g.squeeze(1).unsqueeze(-1).repeat(1, 1, self.hidden_size)
            v = v + g.mul(_v)
        v = v / torch.abs(torch.norm(v, dim=(2)))[:,:,None]
        return v

    
    def _calc_da_logits(self, question_segments, sequence_output):

        logits = {}
        if self.args.compute_da_loss:
            da_segment = torch.relu(self.segment_transform_da(question_segments))
            # (N * M, max_num_history_questions, 7)
            logits['history_da'] = self.history_da_classifier(da_segment)
            # (N * M, 7)
            logits['da'] = self.da_classifier(sequence_output[:, 0, :])

        return logits

    
    def _calc_history_logits(
        self,
        N,
        M,
        sequence_output,
        span_start_embs,
        question_boundary_start,
        passage_positions
    ):
        if self.args.hist_loss_weight == 0:
            return {}
        # question_segments size = (N*M, max_num_history_questions, 768).
        question_segments = model_utils.batched_index_select(
            sequence_output, 1, question_boundary_start)

        logits = {}
        # einsum notations:
        # (1) b: NM.
        # (2) q: max_num_history_questions.
        # (3) h: 768.
        # (4) s: max_seq_len.
        start_segments = self.segment_transform_start(question_segments)
        logits['history_start'] = torch.einsum(
            'bqh,bsh->bqs', start_segments, span_start_embs)

        end_segments = self.segment_transform_end(question_segments)
        logits['history_end'] = torch.einsum(
            'bqh,bsh->bqs', end_segments, span_start_embs)

        # segments size = (N*M, max_num_history_questions, 768)
        # segment_transform_relevance size = (768, 768)
        relevance_segment = torch.relu(
            self.segment_transform_relevance(question_segments))
        relevance_segment = self.encoder.coordinate(
            N, M, relevance_segment, passage_positions)

        # history_qa_classifier size = (768, 1)
        # history_relevance_logits size = (N*M, max_num_history_questions, 1)
        logits['history_relevance'] = self.history_qa_classifier(
            relevance_segment)

        return logits

    def _calc_user2agent_logits(
        self,
        N,
        M,
        sequence_output,
        span_start_embs,
        question_boundary_start,
        passage_positions,
    ):

        if self.args.user2agent_loss_weight == 0:
            return {}
        # question_segments size = (N*M, max_num_history_questions, 768)
        question_segments = model_utils.batched_index_select(
            sequence_output, 1, question_boundary_start)

        logits = {}
        # einsum notations:
        # (1) b: NM.
        # (2) q: max_num_history_questions.
        # (3) h: 768.
        # (4) s: max_seq_len.
        start_segments = self.user2agent_transform_start(question_segments)
        logits['user2agent_start'] = torch.einsum(
            'bqh,bsh->bqs', start_segments, span_start_embs)

        end_segments = self.user2agent_transform_end(question_segments)
        logits['user2agent_end'] = torch.einsum(
            'bqh,bsh->bqs', end_segments, span_start_embs)

        # segments size = (N*M, max_num_history_questions, 768)
        # segment_transform_relevance size = (768, 768)
        relevance_segment = torch.relu(
            self.user2agent_transform_relevance(question_segments))
        relevance_segment = self.encoder.coordinate(
            N, M, relevance_segment, passage_positions)

        # history_qa_classifier size = (768, 1)
        # history_relevance_logits size = (N*M, max_num_history_questions, 1)
        logits['user2agent_relevance'] = self.user2agent_qa_classifier(
            relevance_segment)

        return logits
    

    def _get_question_boundary_start(self, question_boundaries):
        # start size = (N*M, max_num_history_questions)
        # sequence_output size = (N*M, max_num_history_questions, 768)
        start = question_boundaries[:, :, 0]
        start_mask = start == -1
        start = start.masked_fill(start_mask, 0)
        return start


def compute_loss(args, logits, batch, others):

    loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

    N, M, L = batch['input_ids'].size()
    device = logits['start'].device
    dtype = logits['start'].dtype

    answer_starts = batch['answer_starts'].view(N * M, -1)
    answer_ends = batch['answer_ends'].view(N * M, -1)

    # (N*M) * Ls.
    start_logits = logits['start'].view(N * M, -1)
    end_logits = logits['end'].view(N * M, -1)

    # Next turn loss
    relevance_labels = torch.zeros(N, dtype=torch.long).to(device)
    passage_loss = loss_fct(logits['relevance'], relevance_labels)

    num_answers = answer_starts.size(1)
    start_loss = loss_fct(start_logits.repeat(num_answers, 1), answer_starts.T.reshape(-1))
    end_loss = loss_fct(end_logits.repeat(num_answers, 1), answer_ends.T.reshape(-1))
    span_loss = start_loss + end_loss


    # Dialog act (da) loss.
    if args.compute_da_loss:
        da_logits = logits['da']
        history_da_logits = logits['history_da']
        
        da_logits = da_logits.view(N * M, 7)
        # da_label: (N, M)
        da_loss = loss_fct(da_logits, batch['da_label'].view(-1))
        history_da_logits = history_da_logits.view(-1, 7)
        # da_label: (N, M, max_turns)
        history_da_loss = loss_fct(history_da_logits, batch['history_da_label'].view(-1))

    # History loss.
    if args.hist_loss_weight > 0:
        loss_fct2 = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        max_num_spans = logits['history_start'].size(2)
        # history_start_logits size = (N*M, max_num_history_questions, Ls)
        history_start_loss = loss_fct2(
            logits['history_start'].view(-1, max_num_spans),
            batch['history_answer_starts'].view(-1))
        history_end_loss = loss_fct2(
            logits['history_end'].view(-1, max_num_spans),
            batch['history_answer_ends'].view(-1))

        start_mask = batch['history_answer_starts'] != -1
        end_mask = batch['history_answer_ends'] != -1

        # If the batch has no history spans, it will result in NaN loss.
        # Thus, we chech if there is at least one span and compute mean of loss.
        if start_mask.sum() == 0:
            history_start_loss = history_start_loss.mean()
        else:
            history_start_loss = history_start_loss.sum() / start_mask.sum()

        if end_mask.sum() == 0:
            history_end_loss = history_end_loss.mean()
        else:
            history_end_loss = history_end_loss.sum() / end_mask.sum()

        # history_relevance_logits size = (N*M, max_num_history_questions, 1)
        # history_relevance size = (N, max_num_history_questions)
        max_num_history_questions = logits['history_relevance'].size(1)
        history_relevance_logits = torch.cat(
            [t.transpose(0, 1)
            for t in logits['history_relevance'].split(M, dim=0)], dim=0)
        history_passage_loss = loss_fct2(history_relevance_logits.view(
            N*max_num_history_questions, M), batch['history_relevance'].view(-1))

        passage_mask = batch['history_relevance'] != -1
        if passage_mask.sum() == 0:
            history_passage_loss = history_passage_loss.mean()
        else:
            history_passage_loss = history_passage_loss.sum() / passage_mask.sum()
    
        history_span_loss = history_start_loss + history_end_loss 
    
    # user2agent.
    if args.user2agent_loss_weight > 0:
        max_num_spans = logits['user2agent_start'].size(2)
        # history_start_logits size = (N*M, max_num_history_questions, Ls)
        user2agent_start_loss = loss_fct(logits['user2agent_start'].view(
            -1, max_num_spans), batch['user2agent_answer_starts'].view(-1))
        user2agent_end_loss = loss_fct(logits['user2agent_end'].view(
            -1, max_num_spans), batch['user2agent_answer_ends'].view(-1))

        # history_relevance_logits size = (N*M, max_num_history_questions, 1)
        # history_relevance size = (N, max_num_history_questions)
        max_num_user_questions = logits['user2agent_relevance'].size(1)
        user2agent_relevance_logits = torch.cat(
            [t.transpose(0, 1)
            for t in logits['user2agent_relevance'].split(M, dim=0)], dim=0)
        user2agent_passage_loss = loss_fct(user2agent_relevance_logits.view(
            N*max_num_user_questions, M), batch['user2agent_relevance'].view(-1))

        user2agent_span_loss = user2agent_start_loss + user2agent_end_loss 

    # Adv loss.
    if args.adv_loss_weight > 0:
        adv_loss = {}
        for k in args.adv_calc_logits_keys:
            if others['emb_val'] >= 0:
                adv_loss[f'adv_{k}'] = loss.LOSS[
                    args.adv_loss_type](logits[k], logits[f'adv_{k}'])
            else:
                adv_loss[f'adv_{k}'] = torch.tensor(0, dtype=dtype).to(device)
    # Total
    losses = {
        "start": start_loss,
        "end": end_loss,
        "span": span_loss,
        "passage": passage_loss}
    total_loss = span_loss + passage_loss

    if args.hist_loss_weight > 0:
        losses["history_span"] = history_span_loss
        losses["history_passage"] = history_passage_loss
        history_loss = history_span_loss + history_passage_loss
        total_loss += args.hist_loss_weight * history_loss

    if args.user2agent_loss_weight > 0:
        losses["user2agent_span"] = user2agent_span_loss
        losses["user2agent_passage"] = user2agent_passage_loss
        user2agent_loss = user2agent_span_loss + user2agent_passage_loss
        total_loss += args.user2agent_loss_weight * user2agent_loss

    if args.compute_da_loss:
        losses["da"] = da_loss
        losses["history_da"] = history_da_loss
        total_loss += (da_loss + history_da_loss)

    if args.adv_loss_weight > 0:
        losses.update(adv_loss)
        total_loss += args.adv_loss_weight * sum(adv_loss.values())

    losses["total"] = total_loss
    return losses
