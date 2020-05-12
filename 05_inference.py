import functools
import os
import pickle
import re
import unicodedata
from argparse import Namespace

import numpy as np
import torch
from flask import request
from flask_api import FlaskAPI
from gensim.models.word2vec import Word2Vec
from numpy import linalg as LA
from pyvi import ViTokenizer
from torch import nn
from torch.autograd import Variable

from model import GatedCNN, StopWord
from utils import word_vectors_2_embedding, get_sentences, print_inline, strip_unused_char

# Using GPU: CUDA = True
print("Using CUDA: {}".format(torch.cuda.is_available()))

torch.manual_seed(3695)
params = Namespace(
    data_path="assets/items.pickle",
    dictionary_path="build/dictionary.pickle",
    common_words_path="build/common_words.pkl",
    uncommon_words_path="build/uncommon_words.pkl",
    num_channels=[1024] * 4,
    save_path='build/gated_cnn.pickle',
    embedding_path='assets/embedding.pickle'
)


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


def get_word_index(wv, token, common_words, uncommon_words):
    if token in common_words:
        token = StopWord.COMMON.value
    elif token in uncommon_words or token not in wv.vocab:
        token = StopWord.UNCOMMON.value

    return wv.vocab[token].index


def build_batch(items, wv, min_doc_length, common_words, uncommon_words):
    batch = {}
    count = 0

    docs_sentences = get_sentences(items)
    for newsId, sentences in docs_sentences.items():
        print_inline('Pre-process items {}/{}'.format(count, len(docs_sentences)))
        count += 1

        words = [w for s in sentences for w in s.strip().split()]
        if len(words) < min_doc_length:
            continue

        words_indices = [get_word_index(wv, word, common_words, uncommon_words) for word in words]
        batch[newsId] = words_indices

    return batch


class State:
    word_vectors = None
    common_words = None
    uncommon_words = None
    embedding_table = None
    net = None
    mean_vector = None
    std_vector = None
    embedding_items = None
    embedding_size = None
    items = None

    def __init__(self):
        dictionary = Word2Vec.load(params.dictionary_path)
        self.common_words = pickle.load(open(params.common_words_path, 'rb'))
        self.uncommon_words = pickle.load(open(params.uncommon_words_path, 'rb'))

        self.word_vectors = dictionary.wv
        del dictionary

        embeddings = word_vectors_2_embedding(self.word_vectors)
        self.embedding_table = nn.Embedding(*embeddings.shape)
        self.embedding_table.weight.data.copy_(torch.tensor(embeddings))
        self.embedding_table.weight.requires_grad = False

        print('Finish loading Word2Vec model! Size: ({},{})'.format(embeddings.shape[0], embeddings.shape[1]))

        self.embedding_size = embeddings.shape[1]

        self.net = GatedCNN(self.embedding_size, params.num_channels)
        if torch.cuda.is_available():
            self.net = self.net.cuda()
            self.net.load_state_dict(torch.load(params.save_path))
        else:
            self.net.load_state_dict(torch.load(params.save_path, map_location=lambda storage, loc: storage))
        self.net.eval()

        with open('assets/items.pickle', mode='rb') as fp:
            self.items = pickle.load(fp)

        if os.path.exists(params.embedding_path):
            with open(params.embedding_path, mode='rb') as fp:
                self.embedding_items = pickle.load(fp)
        else:
            batch = build_batch(self.items, self.word_vectors, 10, self.common_words, self.uncommon_words)

            self.embedding_items = {}
            count = 0
            for newsId, index_vector in batch.items():
                count += 1
                print_inline('Calculating item embedding {}/{}'.format(count, len(self.items)))

                try:
                    index_vector = torch.tensor(index_vector)
                    inputs = self.embedding_table(index_vector)
                    inputs = torch.FloatTensor(inputs)
                    inputs = inputs.unsqueeze(0).permute(0, 2, 1)  # (batch_size, embedding_size, seq_len)
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                    inputs = Variable(inputs)
                    doc_embedding = self.net(inputs)[0].cpu().detach().numpy()
                    doc_embedding = doc_embedding / LA.norm(doc_embedding)
                    self.embedding_items[newsId] = doc_embedding
                except Exception as e:
                    print(e)
            with open(params.embedding_path, mode='wb') as fp:
                pickle.dump(self.embedding_items, fp, pickle.HIGHEST_PROTOCOL)

        embedding_items = np.array(list(self.embedding_items.values()))
        self.mean_vector, self.std_vector = np.mean(embedding_items, axis=0), np.std(embedding_items, axis=0)

        self.embedding_items = {k: (v - self.mean_vector) for k, v in self.embedding_items.items()}


app = FlaskAPI(__name__)
state = State()


@app.route('/infer', methods=['POST'])
def infer():
    title = str(request.data.get('title', request.form.get('title', '')))
    sapo = str(request.data.get('sapo', request.form.get('sapo', '')))
    tag = str(request.data.get('tag', request.form.get('tag', '')))
    tokenize = bool(request.data.get('tokenize', request.form.get('tokenize', 'false')))
    if tokenize:
        title = ViTokenizer.tokenize(title)
        sapo = ViTokenizer.tokenize(sapo)
        tag = ViTokenizer.tokenize(tag)

    items = {
        "TEMP": {
            "title_token": title,
            "sapo_token": sapo,
            "tag_token": tag
        }
    }
    batch = build_batch(items, state.word_vectors, 3, state.common_words, state.uncommon_words)
    if batch.get("TEMP") is None:
        return [0] * state.embedding_table.weight.data.shape[1]

    index_vector = torch.tensor(batch["TEMP"])
    inputs = state.embedding_table(index_vector)
    inputs = torch.FloatTensor(inputs)
    inputs = inputs.unsqueeze(0).permute(0, 2, 1)  # (batch_size, embedding_size, seq_len)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    inputs = Variable(inputs)
    doc_embedding = state.net(inputs)[0].cpu().detach().numpy()
    doc_embedding = (doc_embedding / LA.norm(doc_embedding)) - state.mean_vector

    def custom_comparator(id1, id2):
        score = (np.dot(doc_embedding, state.embedding_items[id2]) - np.dot(doc_embedding,
                                                                            state.embedding_items[id1])).item()
        if score > 0:
            return 1
        if score == 0:
            return 0
        return -1

    sorted_ids = sorted(state.embedding_items.keys(),
                        key=functools.cmp_to_key(lambda id1, id2: custom_comparator(id1, id2)))

    results = []
    for i in range(10):
        results.append(state.items[sorted_ids[i]]['title_token'])
    return results


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9411, debug=False)
