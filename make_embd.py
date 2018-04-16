from src.torchutils.data import Vocab
import numpy as np
import spacy


def main():
    nlp = spacy.load('en_core_web_lg')
    vocab = Vocab('data/vocab', 50000)
    vocab_size = vocab.size
    vec_len = 300

    rand_max = 1e-1

    vectors = np.random.uniform(-rand_max, rand_max, (vocab_size, vec_len))
    no_vec = []
    for w, id in vocab._word_to_id.items():
        if nlp.vocab.has_vector(w):
            vectors[id] = nlp.vocab.get_vector(w)
        else:
            print('no vec:', w)
            no_vec.append(w)
    print('vocab size:', vocab_size)
    print('num:', len(no_vec))

    vectors[0] = np.zeros((vec_len))

    def print_vec(id):
        w = vocab.id2word(id)
        print(w, vectors[id] == nlp.vocab.get_vector(w))

    # print(vectors.size)
    # print_vec(0)
    # print_vec(1)
    # print_vec(9)

    return vectors


def save(vec):
    np.save('vector.npy', vec)


def load():
    return np.load('vector.npy')


if __name__ == '__main__':
    vec = main()
    save(vec)
    # print(load().dtype)
