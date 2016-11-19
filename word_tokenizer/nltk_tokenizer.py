''' Treebank Tokenizer doesn't do well with punctuation.
    We can design our own custom regex but the NLTK one does the
    job as well (use simple wrapper)
'''

from nltk.tokenize import WordPunctTokenizer

class Tokenizer(object):
    def __init__(self):
        self.model = WordPunctTokenizer()

    def tokenize(self, sentence):
        return self.model.tokenize(sentence)
