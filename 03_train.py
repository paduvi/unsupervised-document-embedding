import os
import pathlib
import pickle
import random
import time
from argparse import Namespace
from math import log10, floor

import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models.word2vec import Word2Vec, FAST_VERSION
from torch.autograd import Variable

from model import UnsupervisedCNNEmbeddingNetwork, StopWord
from utils import word_vectors_2_embedding, get_sentences, print_inline

params = Namespace(
    min_offset=10,
    n_positive=10,
    n_negative=50,
    num_channels=[1024] * 4,
    n_epochs=40,
    batch_size=32,
    learning_rate=0.1,
    momentum=0.9,
    weight_decay=0.1,
    save_path='build/gated_cnn.pickle',
    dictionary_path='build/dictionary.pickle',
    common_words_path="build/common_words.pkl",
    uncommon_words_path="build/uncommon_words.pkl"
)

# Using GPU: FAST_VERSION > -1
print('Current gensim FAST_VERSION: %d' % FAST_VERSION)
# Using GPU: CUDA = True
print("Using CUDA: {}".format(torch.cuda.is_available()))

torch.manual_seed(3695)


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


def sample_prediction_point(input_samples, wv):
    min_length = min([len(sample) for sample in input_samples])
    available_words = set(range(len(wv.index2word)))
    for sample in input_samples:
        available_words -= set(sample)

    mini_batches = []
    for prediction_point in range(params.min_offset, min_length - params.n_positive + 1):
        inputs = [sample[:prediction_point] for sample in input_samples]
        targets = [sample[prediction_point:prediction_point + params.n_positive] for sample in input_samples]

        negatives = random.sample(available_words, params.n_negative)
        targets = [target + negatives for target in targets]
        mini_batches.append((inputs, targets))
    return mini_batches


def _get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def train(samples, word_vectors, net, optimizer, criterion, epoch):
    # shuffle train set
    random.shuffle(samples)
    acc_loss = 0.0
    total_step = 0
    len_samples = len(samples)
    n_batches = len_samples // params.batch_size
    if (len_samples - n_batches * params.batch_size) != 0:
        n_batches += 1

    for batch_idx, i in enumerate(range(n_batches), 1):
        start = i * params.batch_size
        end = start + params.batch_size
        batch_samples = samples[start:end]
        mini_batches = sample_prediction_point(batch_samples, word_vectors)

        batch_loss = 0.0
        for inputs, targets in mini_batches:
            inputs, targets = torch.LongTensor(inputs), torch.LongTensor(targets)
            bs = inputs.shape[0]
            labels = torch.cat([torch.ones(bs, params.n_positive),
                                torch.zeros(bs, params.n_negative)], 1)
            if torch.cuda.is_available():
                inputs, targets, labels = inputs.cuda(), targets.cuda(), labels.cuda()
            inputs, targets, labels = Variable(inputs), Variable(targets), Variable(labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            logits = net(inputs, targets)

            loss = criterion(logits, labels)
            batch_loss += loss.item()
            loss.backward()
            # torch.nn.utils.clip_grad_norm(net.network.cnn.parameters(), 0.25)
            optimizer.step()

        acc_loss += batch_loss
        total_step += len(mini_batches)
        print_inline('Train Epoch: {} [{} / {} ({:.1f}%)]   Learning Rate: {}   Loss: {:.6f}'
                     .format(epoch, str(batch_idx).ljust(int(floor(log10(n_batches))), ' '), n_batches,
                             100. * batch_idx / n_batches,
                             _get_learning_rate(optimizer)[0], batch_loss / len(mini_batches)))

    acc_loss /= total_step
    return acc_loss


def main():
    start_at = time.time()
    dictionary = Word2Vec.load(params.dictionary_path)
    word_vectors = dictionary.wv
    del dictionary

    embedding = word_vectors_2_embedding(word_vectors)
    print('Finish loading Word2Vec model! Size: ({},{})'.format(embedding.shape[0], embedding.shape[1]))

    net = UnsupervisedCNNEmbeddingNetwork(embedding, params.num_channels, pos=params.n_positive, neg=params.n_negative)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    net.train()
    # optimizer = optim.SGD([{
    #     'params': net.module.network.cnn.parameters() if torch.cuda.device_count() > 1 else net.network.cnn.parameters()
    # }, {
    #     'params': net.module.network.fc.parameters() if torch.cuda.device_count() > 1 else net.network.fc.parameters(),
    #     'weight_decay': params.weight_decay
    # }], lr=params.learning_rate, momentum=params.momentum)

    optimizer = optim.Adadelta([{
        'params': net.network.cnn.parameters()
    }, {
        'params': net.network.fc.parameters(),
        'weight_decay': params.weight_decay
    }], lr=params.learning_rate, rho=0.9, eps=1e-06)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=1, min_lr=1e-3,
                                                     verbose=True)
    criterion = nn.BCEWithLogitsLoss()

    try:
        data_path = 'assets/data.pickle'
        pathlib.Path(data_path).parent.mkdir(parents=True, exist_ok=True)

        if os.path.exists(data_path):
            with open(data_path, mode='rb') as fp:
                samples = pickle.load(fp)
        else:
            items = pickle.load(open('assets/items.pickle', mode='rb'))
            common_words = pickle.load(open(params.common_words_path, 'rb'))
            uncommon_words = pickle.load(open(params.uncommon_words_path, 'rb'))

            min_doc_length = params.min_offset + params.n_positive
            samples = build_batch(items, word_vectors, min_doc_length, common_words, uncommon_words)
            samples = list(samples.values())
            del items, common_words, uncommon_words

            with open(data_path, mode='wb') as fp:
                pickle.dump(samples, fp, pickle.HIGHEST_PROTOCOL)
        print('\nNumber of samples: %d' % len(samples))

        print("Training...")
        for epoch in range(1, params.n_epochs + 1):  # loop over the dataset multiple times
            acc_loss = train(samples, word_vectors, net, optimizer, criterion, epoch)
            # print statistics
            print_inline('[{:3d}] loss: {:.5f} - learning rate: {}\n'
                         .format(epoch, acc_loss, _get_learning_rate(optimizer)[0]))

            # Save the model if the validation loss is the best we've seen so far.
            if not scheduler.best or scheduler.is_better(acc_loss, scheduler.best):
                with open(params.save_path, 'wb') as f:
                    torch.save(net.network.state_dict(), f)
            scheduler.step(acc_loss)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
        print('-' * 89)
    finally:
        end_at = time.time()
    print("start at: {}\nend_at: {}\nruntime: {} min".format(time.ctime(start_at), time.ctime(end_at),
                                                             (end_at - start_at) / 60))
    print('Finished Training\n')


if __name__ == '__main__':
    main()
