import logging
import multiprocessing
import os
import pathlib
import pickle
from argparse import Namespace
from enum import Enum

from gensim.models.word2vec import Word2Vec, FAST_VERSION

from model import StopWord
from utils import yes_or_no, get_sentences

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Using GPU: FAST_VERSION > -1
print('Current gensim FAST_VERSION: %d' % FAST_VERSION)


class Word2VecAlg(Enum):
    CBOW = 0
    SKIP_GRAM = 1


params = Namespace(
    n_negative=20,
    n_epochs=15,
    window_size=5,
    sample_rate=1e-5,
    alg=Word2VecAlg.SKIP_GRAM.value,
    word_embedding_size=300,
    data_path="assets/items.pickle",
    save_path="build/dictionary.pickle",
    common_words_path="build/common_words.pkl",
    uncommon_words_path="build/uncommon_words.pkl"
)


def load_stopwords():
    """
    Return: stopwords: set
    """
    stopwords = []
    with open('dataset/vietnamese-stopwords.txt', 'r') as f:
        for word in f.readlines():
            word = word.strip()
            stopwords.append(word)
    return set(stopwords)


class DictionaryGenerator(object):
    def __init__(self, docs, stopwords=None, min_count=5, max_count=0.7):
        if stopwords is None:
            stopwords = set()
        self.min_count = min_count
        self.max_count = max_count
        self.sentences = []
        self.count = {}
        self.inverted_index = {}
        self.doc_count = len(docs)
        self.common_words = set()
        self.stopwords = stopwords
        self.uncommon_words = set()
        for newsId, doc_sentences in docs.items():
            doc_words = " ".join(doc_sentences).strip().split()
            uniq_words = set(doc_words)
            for word in uniq_words:
                if word in self.inverted_index:
                    self.inverted_index[word].append(newsId)
                else:
                    self.inverted_index[word] = [newsId]
                # unique word
                if word in self.count:
                    self.count[word] += 1
                else:
                    self.count[word] = 1
            for sentence in doc_sentences:
                self.sentences.append(sentence.strip().split())

    def format_word(self, word):
        if self.count[word] < self.min_count:
            if word not in self.uncommon_words:
                self.uncommon_words.add(word)
            return StopWord.UNCOMMON.value
        if self.count[word] > self.max_count * self.doc_count:
            if word not in self.common_words:
                self.common_words.add(word)
            return StopWord.COMMON.value
        if word in self.stopwords:
            return StopWord.COMMON.value
        return word

    def save_special_words(self):
        self.common_words.update(self.stopwords)

        if len(self.common_words) > 0:
            pickle.dump(self.common_words, open(params.common_words_path, 'wb'), pickle.HIGHEST_PROTOCOL)
        print("Common words: ", len(self.common_words))
        del self.common_words

        if len(self.uncommon_words) > 0:
            uncommon_word_docs = {}
            for w in self.uncommon_words:
                uncommon_word_docs[w] = set(self.inverted_index[w])
            pickle.dump(uncommon_word_docs, open(params.uncommon_words_path, 'wb'), pickle.HIGHEST_PROTOCOL)
        print("Uncommon words: ", len(self.uncommon_words))

    def __iter__(self):
        for sentence in self.sentences:
            yield [self.format_word(word) for word in sentence]


def build_vocabulary():
    pathlib.Path(params.data_path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(params.save_path).parent.mkdir(parents=True, exist_ok=True)

    model = None
    if os.path.isfile(params.save_path):
        if not yes_or_no("Vocabulary existed! Do you want to overwrite?"):
            model = Word2Vec.load(params.save_path)

    if model is None:
        items = pickle.load(open(params.data_path, 'rb'))
        normalized_sentences = get_sentences(items)
        del items

        stopwords = load_stopwords()
        sentence_iterator = DictionaryGenerator(
            normalized_sentences,
            stopwords=stopwords)
        del normalized_sentences
        print("\nTotal of sentences: %d" % len(sentence_iterator.sentences))

        model = Word2Vec(sentence_iterator, seed=3695, min_count=1, sg=params.alg, size=params.word_embedding_size,
                         window=params.window_size, sample=params.sample_rate, negative=params.n_negative,
                         workers=max(1, multiprocessing.cpu_count()), iter=params.n_epochs)

        sentence_iterator.save_special_words()
        # del sentences
        del sentence_iterator

        print("Saving dictionary at " + params.save_path)
        model.save(params.save_path)

    word_vectors = model.wv
    del model
    print("Done. Vocabulary size is: %d" % len(word_vectors.vocab))


if __name__ == '__main__':
    build_vocabulary()
