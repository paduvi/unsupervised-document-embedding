import json
import pathlib
import pickle

import nltk

from utils import print_inline, normalize


def preprocess_dataset(fields=None):
    if fields is None:
        fields = ['title_token', 'sapo_token', 'content_token', 'tag_token']

    results = {}
    sentence_length_arr = []
    with open('dataset/items.txt', 'r') as fp:
        count = 0
        while True:
            line = fp.readline().strip()
            if not line:
                break
            count += 1
            print_inline(count)

            item = json.loads(line)
            for field in fields:
                text = item.get(field)
                if text is None:
                    continue
                sentences = nltk.tokenize.sent_tokenize(text)
                normalized_text = ""
                for sentence in sentences:
                    sentence_length, sentence = normalize(sentence)
                    sentence_length_arr.append(sentence_length)
                    normalized_text += sentence + " . "
                item[field] = normalized_text.strip(". ")
            results[item.get('newsId')] = item

    print("Average length of each sentence is: %.2f" % (sum(sentence_length_arr) / len(sentence_length_arr)))
    del sentence_length

    output = 'assets/items.pickle'
    pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'wb') as fp:
        pickle.dump(results, fp, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    nltk.download('punkt')
    preprocess_dataset()
