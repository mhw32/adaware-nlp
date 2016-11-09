import numpy as np
import glove
from nose.tools import assert_equal

# from gensim
test_corpus = ("""human interface computer
survey user computer system response time
eps user interface system
system human system eps
user response time
trees
graph trees
graph minors trees
graph minors survey
I like graph and stuff
I like trees and stuff
Sometimes I build a graph
Sometimes I build trees""").split("\n")

vocab = glove.build_vocab(test_corpus)
cooccur = glove.build_cooccur_mat(vocab,
                                  test_corpus,
                                  window_size=10)
id2word = glove.make_id2word(vocab)

W = glove.train_glove(vocab,
                      cooccur,
                      vector_size=10,
                      iterations=500)

# Merge and normalize word vectors
W = glove.merge_main_context(W)

similar = glove.most_similar(W, vocab, id2word, 'graph')
print(similar)
assert_equal('trees', similar[0])
