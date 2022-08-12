import torch
import collections
from typing import List

class ReaderPassage:
    """
    Container to collect and cache all Q&A passages related attributes before generating the reader input
    """

    def __init__(
        self,
        id=None,
        text: List[str] = None,
        type: List[int] = None,
        title: str = None,
        position: int = None,
        has_answer: bool = None,
        answers_spans: List = None,
        history_has_answers: List[bool]=None,
        history_answers_spans: List[list]=None,
    ):
        self.id = id
        self.span_texts = text
        self.span_types = type
        self.title = title
        self.position = position
        self.has_answer = has_answer

        # index pair indicating the start/end span id of each answer
        self.answers_spans = answers_spans

        # passage token ids
        self.sequence_ids = None
        self.sequence_type_ids = None
        self.question_boundaries = None

        # indices of cls tokens in sequence_ids
        self.clss = None
        self.ends = None
        # mask of clss, where the first two and padded cls tokens are 0
        self.mask_cls = None

        self.history_has_answers = history_has_answers
        self.history_answers_spans = history_answers_spans

        self.dialog_act_id = None
        self.history_dialog_act_ids = None

    def to_tensor(self):
        self.sequence_ids = torch.from_numpy(self.sequence_ids)
        self.sequence_type_ids = torch.from_numpy(self.sequence_type_ids)
        self.clss = torch.from_numpy(self.clss)
        self.ends = torch.from_numpy(self.ends)
        self.mask_cls = torch.from_numpy(self.mask_cls)
        self.question_boundaries = torch.from_numpy(self.question_boundaries)


class ReaderSample:
    """
    Container to collect all Q&A passages data per singe question
    """

    def __init__(
        self,
        question: str,
        answers: List,
        id=None,
        positive_passages: List[ReaderPassage] = [],
        negative_passages: List[ReaderPassage] = [],
        passages: List[ReaderPassage] = [],
    ):
        self.id = id
        self.question = question
        self.answers = answers
        self.positive_passages = positive_passages
        self.negative_passages = negative_passages
        self.passages = passages

    @property
    def all_passages(self):
        return self.passages + self.negative_passages + self.positive_passages

    def to_tensor(self):
        for p in self.all_passages:
            p.to_tensor()


SpanPrediction = collections.namedtuple(
    "SpanPrediction",
    [
        "prediction_text",
        "span_score",
        "relevance_score",
        "passage_index",
        "passage_text",
    ],
)
