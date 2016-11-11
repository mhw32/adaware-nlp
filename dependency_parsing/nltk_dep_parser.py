''' This is a little out-of-our-reach. Let's try to build
    one but for the mean time, let's use the stanford
    parser built into NLTK.
'''

from nltk.parse.stanford import StanfordNeuralDependencyParser

class DependencyParser(object):
    def __init__(self):
        self.model = StanfordNeuralDependencyParser()

    def str_parse(self, sentence):
        ''' sentence is a string '''
        parsed = self.model.raw_parse(sentence)
        return [p.tree() for p in parsed]

    def lst_parse(self, sentence):
        ''' sentence is a list of words '''
        parsed = self.model.parse(sentence)
        return [p.tree() for p in parsed]
