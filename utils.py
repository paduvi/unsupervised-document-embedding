import re
import sys
import unicodedata

import numpy as np


def strip_unused_char(text):
    # reference: https://github.com/trungtv/pyvi
    specials = [r"==>", r"->", r"\.\.\.", r">>"]
    digit = r"\d+([\.,_]\d+)+"
    email = r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
    web = r"http[s]?:\/\/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    datetime = [
        r"\d{1,2}\/\d{1,2}(\/\d+)?",
        r"\d{1,2}-\d{1,2}(-\d+)?",
    ]
    word = r"\w+"
    # non_word = r"[^\w\s]"
    abbreviations = [
        r"[A-ZĐ]+\.(?:[A-ZĐ]+\.*)*",
        r"Tp\.",
        r"Mr\.", r"Mrs\.", r"Ms\.",
        r"Dr\.", r"ThS\."
    ]

    patterns = []
    patterns.extend(abbreviations)
    patterns.extend(specials)
    patterns.extend([web, email])
    patterns.extend(datetime)
    patterns.extend([digit, word])
    patterns = "(" + "|".join(patterns) + ")"

    tokens = re.findall(patterns, text, re.UNICODE)

    return [token[0] for token in tokens]


def normalize(text):
    text = unicodedata.normalize('NFC', text)
    tokens = strip_unused_char(text)

    # reference: https://int3ractive.com/2010/06/optimal-unicode-range-for-vietnamese.html
    pattern = re.compile(r"[\W]+", flags=re.UNICODE)
    vietnamese_regex = re.compile(
        r"[a-zA-ZĐ0-9_.,\u00C0-\u00C3\u00C8-\u00CA\u00CC-\u00CD\u00D0\u00D2-\u00D5\u00D9-\u00DA"
        r"\u00DD\u00E0-\u00E3\u00E8-\u00EA\u00EC-\u00ED\u00F2-\u00F5\u00F9-\u00FA\u00FD"
        r"\u0102-\u0103\u0110-\u0111\u0128-\u0129\u0168-\u0169\u01A0-\u01A3\u1EA0-\u1EF9]+",
        flags=re.UNICODE
    )

    tokens = [word.strip("_.") for word in tokens
              if vietnamese_regex.match(word) and not pattern.match(word)]
    return len(tokens), " ".join(tokens).lower()



def yes_or_no(question):
    c = str(input(question + " (Y/N): ")).lower().strip()[:1]
    if c not in ("y", "n"):
        print("Please enter valid inputs")
        return yes_or_no(question)
    return c == 'y'


def print_inline(text):
    sys.stdout.write('\r\033[K\r{}'.format(text)),
    sys.stdout.flush()


def get_sentences(items, fields=None):
    """
    Parameters
        items: dict
            all items, each element is (id: str, item: dict)
    Return
        results: dict
            Normalized sentences of each item
    """
    if fields is None:
        fields = ['title_token', 'sapo_token', 'content_token', 'tag_token']
    results = {}
    count = 0
    # print 'Number of items: {}'.format(len(items))

    for newsId, item in items.items():
        content = ""
        for col in fields:
            col_content = item.get(col)
            if col_content is None:
                continue
            col_content = col_content.strip('. ')
            if not col_content:
                continue
            content += col_content + ' . '
        content = content.strip('. ')

        sentences = re.compile(r'\s+\.\s+').split(content)
        results[newsId] = sentences

        count += 1
        print_inline('Calculating item sentences {}/{}'.format(count, len(items)))

    return results


def word_vectors_2_embedding(wv):
    return np.array([wv[word] for word in wv.index2word])
