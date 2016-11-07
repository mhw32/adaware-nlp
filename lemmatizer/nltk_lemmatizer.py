''' Since the Neural Lemmatizer is not working that well
    let's use a simple wrapper around the NLTK one for now
'''

from nltk.stem import WordNetLemmatizer

class Lemmatizer(object):
    def __init__(self):
        self.model = WordNetLemmatizer()

    def lemmatize(self, word, pos='n'):
        self.model.lemmatize(word, pos=pos)
