from enum import Enum


class StopWord(Enum):
    UNCOMMON = "<UNCOM>"  # min_count < 5
    COMMON = "<COMM>"  # belongs to stopwords set or max_count > 0.7
