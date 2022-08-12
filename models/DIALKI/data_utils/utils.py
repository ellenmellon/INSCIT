from typing import List
from transformers import BertTokenizer, BertTokenizerFast

from .data_class import SpanPrediction 

def is_word_head(tokenizer, token):
    # TODO including all models that are not using BPE tokenization
    if isinstance(tokenizer, (BertTokenizerFast, BertTokenizer)):
        return not token.startswith('##')
    return token.startswith('Ġ')


def get_word_idxs(tokenizer, tokens, party_tokens, dont_mask_words):

    word_idxs = []
    curr_word_idx = 0
    prev_t = None
    # for whole word masking
    for t in tokens:
        if t in dont_mask_words:
            word_idxs.append(-1)
        # Handling : is for BPE tokenizer. Remember to add : if you use BPE
        # tokenizer, such as RoBertaTokenizer
        elif t == ':' and prev_t in party_tokens:
            word_idxs.append(-1)
        elif is_word_head(tokenizer, t):
            curr_word_idx += 1
            word_idxs.append(curr_word_idx)
        else:
            word_idxs.append(curr_word_idx)
        prev_t = t

    assert len(tokens) == len(word_idxs)

    return word_idxs


def start_end_finder(start_logits, end_logits, max_answer_length, span_type, mask_cls):
    scores = []
    for (i, s) in enumerate(start_logits):
        for (j, e) in enumerate(end_logits[i : i + max_answer_length]):
            if mask_cls[i] != 0 and mask_cls[i + j] != 0:
                if span_type and span_type[i+j-1] != span_type[i-1]:
                    break
                scores.append(((i, i + j), s + e))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    chosen_span_intervals = []
    for (start_index, end_index), score in scores:
        assert start_index <= end_index
        length = end_index - start_index + 1
        assert length <= max_answer_length

        if any(
            [
                start_index <= prev_start_index <= prev_end_index <= end_index
                or prev_start_index <= start_index <= end_index <= prev_end_index
                for (prev_start_index, prev_end_index) in chosen_span_intervals
            ]
        ):
            continue

        chosen_span_intervals.append((start_index, end_index))
        yield start_index, end_index, score

    yield -1, -1, -1


def get_best_spans(
    start_logits: List,
    end_logits: List,
    max_answer_length: int,
    passage_idx: int,
    span_text: str,
    span_type: str,
    mask_cls: List,
    relevance_score: float,
    top_spans: int = 1,
) -> List[SpanPrediction]:
    """
    Finds the best answer span for the extractive Q&A model
    """

    best_spans = []
    for start_index, end_index, score in start_end_finder(start_logits, end_logits, max_answer_length, span_type, mask_cls):
        if start_index == -1 and end_index == -1:
            break

        predicted_answer = ' '.join(span_text[start_index-1:end_index]) # offset the question and title segment

        best_spans.append(
            SpanPrediction(
                predicted_answer, score, relevance_score, passage_idx, ' '.join(span_text)
            )
        )

        if len(best_spans) == top_spans:
            break
    return best_spans
