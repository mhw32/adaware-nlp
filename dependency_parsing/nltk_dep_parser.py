''' This is a little out-of-our-reach. Let's try to build
    one but for the mean time, let's use the stanford
    parser built into NLTK.
'''

from nltk.parse.stanford import StanfordDependencyParser

class DependencyParser(object):
    def __init__(self, path_to_jar, path_to_models_jar):
        self.model = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

    def str_parse(self, sentence):
        ''' sentence is a string '''
        parsed = self.model.raw_parse(sentence)
        return [p for p in parsed]

    def lst_parse(self, sentence):
        ''' sentence is a list of words '''
        parsed = self.model.parse(sentence)
        return [p for p in parsed]
