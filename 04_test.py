import functools
import os
import pickle
import re
import unicodedata
from argparse import Namespace

import numpy as np
import torch
from gensim.models.word2vec import Word2Vec
from numpy import linalg as LA
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


def build_batch(items, wv, common_words, uncommon_words):
    batch = {}
    count = 0

    docs_sentences = get_sentences(items)
    for newsId, sentences in docs_sentences.items():
        print_inline('Pre-process items {}/{}'.format(count, len(docs_sentences)))
        count += 1

        words = [w for s in sentences for w in s.strip().split()]
        if len(words) < 10:
            continue

        words_indices = [get_word_index(wv, word, common_words, uncommon_words) for word in words]
        batch[newsId] = words_indices

    return batch


def main():
    dictionary = Word2Vec.load(params.dictionary_path)
    common_words = pickle.load(open(params.common_words_path, 'rb'))
    uncommon_words = pickle.load(open(params.uncommon_words_path, 'rb'))

    word_vectors = dictionary.wv
    del dictionary

    embeddings = word_vectors_2_embedding(word_vectors)
    embedding_table = nn.Embedding(*embeddings.shape)
    embedding_table.weight.data.copy_(torch.tensor(embeddings))
    embedding_table.weight.requires_grad = False

    print('Finish loading Word2Vec model! Size: ({},{})'.format(embeddings.shape[0], embeddings.shape[1]))

    embedding_size = embeddings.shape[1]

    test_words = ['đẹp', 'Ronaldo', 'Covid']
    for test_word in test_words:
        _, test_word = normalize(test_word)
        print("*" * 90)
        print("Danh sách từ khóa cùng ngữ cảnh với từ: %s" % test_word)
        test_word_embedding = word_vectors[test_word]

        scores = np.matmul(embeddings, test_word_embedding)
        print([word_vectors.index2word[top_idx] for top_idx in np.argsort(scores)[-2:-12:-1]])

        print("*" * 90)

    net = GatedCNN(embedding_size, params.num_channels)
    if torch.cuda.is_available():
        net = net.cuda()
        net.load_state_dict(torch.load(params.save_path))
    else:
        net.load_state_dict(torch.load(params.save_path, map_location=lambda storage, loc: storage))
    net.eval()

    with open('assets/items.pickle', mode='rb') as fp:
        items = pickle.load(fp)

    if os.path.exists(params.embedding_path):
        with open(params.embedding_path, mode='rb') as fp:
            embedding_items = pickle.load(fp)
    else:
        batch = build_batch(items, word_vectors, common_words, uncommon_words)
        # save and clean
        del common_words, uncommon_words

        embedding_items = {}
        count = 0
        for newsId, index_vector in batch.items():
            count += 1
            print_inline('Calculating item embedding {}/{}'.format(count, len(items)))

            try:
                index_vector = torch.tensor(index_vector)
                inputs = embedding_table(index_vector)
                inputs = torch.FloatTensor(inputs)
                inputs = inputs.unsqueeze(0).permute(0, 2, 1)  # (batch_size, embedding_size, seq_len)
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                inputs = Variable(inputs)
                doc_embedding = net(inputs)[0].cpu().detach().numpy()
                doc_embedding = doc_embedding / LA.norm(doc_embedding)
                embedding_items[newsId] = doc_embedding
            except Exception as e:
                print(e)
        with open(params.embedding_path, mode='wb') as fp:
            pickle.dump(embedding_items, fp, pickle.HIGHEST_PROTOCOL)

    def item_sim(id1, id2):
        return np.dot(embedding_items.get(id1, np.zeros(embedding_size)),
                      embedding_items.get(id2, np.zeros(embedding_size))).item()

    while True:
        item_id = input("\nNhập vào ID cua bài viết: ").strip()
        if item_id == "":
            break
        if item_id not in embedding_items:
            print("ID không tồn tại")
            continue
        print("Bài đang xét: " + items[item_id]['title_token'])

        def custom_comparator(id1, id2):
            score = item_sim(item_id, id2) - item_sim(item_id, id1)
            if score > 0:
                return 1
            if score == 0:
                return 0
            return -1

        candidate_items = embedding_items.copy()
        candidate_items.pop(item_id)

        sorted_ids = sorted(candidate_items.keys(),
                            key=functools.cmp_to_key(lambda id1, id2: custom_comparator(id1, id2)))

        print("Danh sách top 10 bài liên quan được gợi ý:")

        count = 0
        i = 0
        title_set = set(normalize(items[item_id]['title_token']))
        while count < 10:
            title = normalize(items[sorted_ids[i]]['title_token'])
            i += 1
            if title in title_set:
                continue
            count += 1
            title_set.add(title)
            print("{}. {}".format(count, items[sorted_ids[i]]['title_token']))


if __name__ == '__main__':
    main()
