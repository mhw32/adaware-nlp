import gensim

model=gensim.models.Word2Vec.load_word2vec_format('converted_glove_model.txt',binary=False)
