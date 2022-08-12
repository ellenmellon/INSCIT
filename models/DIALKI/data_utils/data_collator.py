import collections

import torch
import numpy as np

import config


def _pad(target, fill_value, pad_len, dim=0):
    if pad_len == 0:
        return target
    size = list(target.size())
    size[dim] = pad_len
    pad = torch.full(size, fill_value)
    return torch.cat([target, pad], dim=dim)


class DataCollator:
    def __init__(
        self,
        data_name,
        tokenizer,
        max_seq_len,
        max_num_answers,
        max_num_passages_per_questions,
        special_attention,
        passage_attend_history,
        is_train,
        shuffle):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_num_answers = max_num_answers
        self.max_num_passages_per_questions = max_num_passages_per_questions
        self.special_attention = special_attention
        self.passage_attend_history = passage_attend_history
        self.is_train = is_train
        self.shuffle = shuffle
        self.data_name = data_name


    def __call__(self, samples):
        seq_lens, num_spans, num_passages, num_history_turns, num_user_turns = \
            [], [], [], [], []
        for sample in samples:
            seq_lens.extend([len(p.sequence_ids) for p in sample.all_passages])
            num_passages.append(len(sample.all_passages))
            num_spans.extend([len(p.clss) for p in sample.all_passages])
            num_history_turns.extend(
                [len(p.question_boundaries) for p in sample.all_passages])
            
            user_token_id = self.tokenizer.convert_tokens_to_ids(config.USER)
            user_idxs = (
                sample.all_passages[0].sequence_ids == user_token_id).nonzero(
                    as_tuple=True)[0]
            num_user_turns.append(user_idxs.nelement())

        max_seq_len = max(seq_lens)
        assert self.max_seq_len >= max_seq_len, \
            (f"max_seq_len ({max_seq_len}) > global max_seq_len ({self.max_seq_len})."
             f"Check preprocessing or data dir")

        max_num_spans = max(num_spans)
        passages_per_question = min(
            max(num_passages), self.max_num_passages_per_questions)
        max_num_history_turns = max(num_history_turns)
        max_num_user_turns = max(num_user_turns)


        ret = collections.defaultdict(list)
        for sample in samples:
            positive_ctxs = sample.positive_passages
            if self.is_train:
                negative_ctxs = sample.negative_passages
            else:
                negative_ctxs = sample.passages

            r = self._preprocess_sample(
                positive_ctxs,
                negative_ctxs,
                max_seq_len,
                max_num_spans,
                max_num_history_turns,
                max_num_user_turns,
                passages_per_question,
            )
            for k, v in r.items():
                ret[k].append(v)

        for k, v in ret.items():
            ret[k] = torch.stack(v)

        ret['samples'] = samples

        return ret


    def _preprocess_sample(
        self,
        positives,
        negatives,
        max_seq_len,
        max_num_spans,
        max_num_history_turns,
        max_num_user_turns,
        passages_per_question,
    ):
    
        def _get_answers_tensor(spans):
            starts = [span[0] for span in spans]
            ends = [span[1] for span in spans]

            starts_tensor = torch.full(
                (passages_per_question, self.max_num_answers),
                -1, dtype=torch.long)
            starts_tensor[0, :len(starts)] = torch.tensor(starts)
        
            ends_tensor = torch.full(
                (passages_per_question, self.max_num_answers),
                -1, dtype=torch.long)
            ends_tensor[0, :len(ends)] = torch.tensor(ends)

            return starts_tensor, ends_tensor
    
        # select one positive
        if positives:
            if self.shuffle:
                positive_idx = np.random.choice(len(positives))
            else:
                positive_idx = 0
            positive = positives[positive_idx]
            num_positives = 1
        else:
            num_positives = 0
    
        # select negatives
        negative_idxs = range(len(negatives))
        if self.shuffle:
            negative_idxs = np.random.permutation(negative_idxs)
        negative_idxs = negative_idxs[:passages_per_question-num_positives]
        negatives = [negatives[i] for i in negative_idxs]
    
    
        if self.is_train:
            passages = [positive] + negatives
        else:
            passages = negatives

        num_history_turns = len(passages[0].question_boundaries)

        """
            get labels
        """
        ret = {}
        if self.is_train:
            ret['answer_starts'], ret['answer_ends'] = _get_answers_tensor(
                positive.answers_spans)
            history_da_label = torch.full(
                (passages_per_question, max_num_history_turns),
                -1, dtype=torch.long)
            history_da_label[0, :num_history_turns] = torch.tensor(
                positive.history_dialog_act_ids)
            ret['history_da_label'] = history_da_label

            da_label = torch.full(
                (passages_per_question,), -1, dtype=torch.long)
            da_label[0] = positive.dialog_act_id
            ret['da_label'] = da_label

        if self.data_name == 'dialdoc':
            user_token_id = self.tokenizer.convert_tokens_to_ids(config.USER)
            agent_token_id = self.tokenizer.convert_tokens_to_ids(config.AGENT)
            user_idxs = (passages[0].sequence_ids == user_token_id).nonzero(
                as_tuple=True)[0]
            agent_idxs = (passages[0].sequence_ids == agent_token_id).nonzero(
                as_tuple=True)[0]
    
            user_idxs = [(u, 'u') for u in user_idxs]
            agent_idxs = [(a, 'a') for a in agent_idxs]
            party_idxs = sorted(agent_idxs + user_idxs, key=lambda x: x[0])
            assert party_idxs[0][1] == 'u', \
                'Make sure the first index is from user.'
            user_idxs = []
            user_starts = []
            for i, p in enumerate(party_idxs[1:], start=1):
                if p[1] == 'u' and party_idxs[i-1][1] == 'a':
                    user_idxs.append(i)
                    user_starts.append(p[0])
    
            # +1 for the first user turn (current user turn).
            num_user_turns = len(user_idxs) + 1
        

        ret2 = collections.defaultdict(list)

        # [1] is the first start position of the current user turn.
        if self.data_name == 'dialdoc':
            ret2['user_starts'] = [
                _pad(torch.tensor([1] + user_starts),
                     -1, max_num_user_turns - num_user_turns)
                for _ in range(len(passages))]

        history_relevance = torch.full(
            (max_num_history_turns,), -1, dtype=torch.long)
        if self.data_name == 'dialdoc':
            user2agent_relevance = torch.full(
                (max_num_user_turns,), -1, dtype=torch.long)

        if self.data_name == 'dialdoc':
            if self.is_train:
                user2agent_relevance[0] = 0

        for i, p in enumerate(passages):
            # history
            h_answer_starts, h_answer_ends = [], []
            for j, (span, has_answer) in enumerate(
                zip(p.history_answers_spans, p.history_has_answers)):
                if has_answer:
                    h_answer_starts.append(span[0])
                    h_answer_ends.append(span[1])
                    history_relevance[j] = i
                else:
                    h_answer_starts.append(-1)
                    h_answer_ends.append(-1)
            h_answer_starts = torch.tensor(h_answer_starts, dtype=torch.long)
            h_answer_ends = torch.tensor(h_answer_ends, dtype=torch.long)
            h_answer_starts = _pad(
                h_answer_starts, -1, max_num_history_turns - num_history_turns)
            h_answer_ends = _pad(
                h_answer_ends, -1, max_num_history_turns - num_history_turns)
            ret2['history_answer_starts'].append(h_answer_starts)
            ret2['history_answer_ends'].append(h_answer_ends)
            
            if self.data_name == 'dialdoc':
                if i == 0 and self.is_train:
                    user2agent_answer_starts = [positive.answers_spans[0][0]]
                    user2agent_answer_ends = [positive.answers_spans[0][1]]
                else:
                    user2agent_answer_starts = [-1]
                    user2agent_answer_ends = [-1]
    
                for j, u_i in enumerate(user_idxs, start=1):
                    has_answer = p.history_has_answers[u_i-1]
                    if has_answer: 
                        span = p.history_answers_spans[u_i-1]
                        user2agent_answer_starts.append(span[0])
                        user2agent_answer_ends.append(span[1])
                        user2agent_relevance[j] = i
                    else:
                        user2agent_answer_starts.append(-1)
                        user2agent_answer_ends.append(-1)
                user2agent_answer_starts = torch.tensor(
                    user2agent_answer_starts, dtype=torch.long)
                user2agent_answer_ends = torch.tensor(
                    user2agent_answer_ends, dtype=torch.long)
                user2agent_answer_starts = _pad(
                    user2agent_answer_starts,
                    -1, max_num_user_turns - num_user_turns)
                user2agent_answer_ends = _pad(
                    user2agent_answer_ends,
                    -1, max_num_user_turns - num_user_turns)
                ret2['user2agent_answer_starts'].append(user2agent_answer_starts)
                ret2['user2agent_answer_ends'].append(user2agent_answer_ends)
            

        ret['history_relevance'] = history_relevance
        if self.data_name == 'dialdoc':
            ret['user2agent_relevance'] = user2agent_relevance
    
        # Gets inputs.
        for i, p in enumerate(passages):
            seq_len = len(p.sequence_ids)
            pad_len = max_seq_len - seq_len
            ret2['input_ids'].append(
                _pad(p.sequence_ids, self.tokenizer.pad_token_id, pad_len))
            ret2['type_ids'].append(_pad(p.sequence_type_ids, 0, pad_len))
            ret2['passage_positions'].append(
                torch.tensor(p.position, dtype=torch.long))
    
            pad_len = max_num_spans - len(p.clss)
            ret2['clss'].append(_pad(p.clss, 0, pad_len))
            ret2['ends'].append(_pad(p.ends, 0, pad_len))
            ret2['mask_cls'].append(_pad(p.mask_cls, 0, pad_len))

            pad_len = max_num_history_turns - len(p.question_boundaries)
            ret2['question_boundaries'].append(_pad(p.question_boundaries, -1, pad_len))


            # Gets the special attention mask.
            if self.special_attention:
                user_token_id = self.tokenizer.convert_tokens_to_ids(config.USER)
                agent_token_id = self.tokenizer.convert_tokens_to_ids(config.AGENT)
                user_idxs = (ret2['input_ids'][-1] == user_token_id).nonzero(
                    as_tuple=True)[0]
                agent_idxs = (ret2['input_ids'][-1] == agent_token_id).nonzero(
                    as_tuple=True)[0]

                # 0 is for CLS
                special_idxs = [torch.tensor([0]), user_idxs]
                # agent_idx can be empty
                if agent_idxs.nelement() != 0:
                    special_idxs.append(agent_idxs)
                text_token_id = self.tokenizer.convert_tokens_to_ids(config.TEXT)
                parent_title_token_id = self.tokenizer.convert_tokens_to_ids(
                    config.PARENT_TITLE)
                text_idx = (ret2['input_ids'][-1] == text_token_id).nonzero(
                    as_tuple=True)[0]
                parent_title_idx = (
                    ret2['input_ids'][-1] == parent_title_token_id).nonzero(
                        as_tuple=True)[0]
                if parent_title_idx.nelement() == 0:
                    passage_first_idx = text_idx
                else:
                    passage_first_idx = parent_title_idx 

                assert passage_first_idx.nelement() != 0, \
                    (f'text_idx = {text_idx.nelement()}, '
                     f'parent_title_idx = {parent_title_idx.nelement()}.')

                special_idxs.append(passage_first_idx)
                special_idxs = torch.sort(torch.cat(special_idxs))[0]

                attention_mask = []
                for j in range(len(special_idxs)):
                    m = torch.arange(max_seq_len)

                    # passage can attend history, so start from 0 
                    if (j == len(special_idxs) - 1
                        and self.passage_attend_history):
                        m = ((m >= 0) & (m < seq_len)).long()
                    else:
                        m = ((m >= special_idxs[j]) & (m < seq_len)).long()

                    # we want tokens within the range have the same
                    # attention_mask, so we use repeat.
                    if j < len(special_idxs)-1:
                        repeat = special_idxs[j+1] - special_idxs[j]
                    else:
                        repeat = seq_len - special_idxs[j]

                    m = m.repeat(repeat, 1)
                    attention_mask.append(m)

                attention_mask.append(
                    torch.zeros(
                        (max_seq_len - seq_len, max_seq_len),dtype=torch.long))
                ret2['attention_mask'].append(torch.cat(attention_mask))

        name2fill_value = {
            'input_ids': self.tokenizer.pad_token_id,
            'type_ids': 0,
            'passage_positions': 29,
            'question_boundaries': -1,
            'attention_mask': 0, # for special attention
            'user_starts': -1,
        }
    
        pad_len = passages_per_question - len(passages)
        for k, v in ret2.items(): 
            v = torch.stack(v) 
            fill_value = name2fill_value.get(k, -1)
            ret2[k] = _pad(v, fill_value, pad_len)

        if not self.special_attention:
            assert 'attention_mask' not in ret2
            ret2['attention_mask'] = (
                ret2['input_ids'] != self.tokenizer.pad_token_id).long()

        ret.update(ret2)
    
        return ret