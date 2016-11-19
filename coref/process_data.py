import gensim
import os
import numpy as np
import pdb
import itertools

# model=gensim.models.Word2Vec.load_word2vec_format('converted_glove_model.txt',binary=False)


class mention(object):
    def __init__(self, sentence_num, doc, mention_num, start, words, first_word):
        self.sentence = sentence_num
        self.words = words
        self.doc = doc
        self.mention_num = mention_num
        self.start = int(start)
        self.end = None
        self.first_word = first_word
        self.last_word = None
        self.mention_average = np.zeros(50)
        self.sentence_average = np.zeros(50)
        self.doc_average = np.zeros(50)
        self.previous_word = np.zeros(50)
        self.prev_prev_word = np.zeros(50)
        self.following_word = np.zeros(50)
        self.second_following_word = np.zeros(50)
        self.five_left = np.zeros(50)
        self.five_right = np.zeros(50)

class mention_pair(object):
    def __init__(self, mention1, mention2):
        self.mention1 = mention1
        self.mention2 = mention2

def average(words):
    if len(words) == 1:
        return words[0]
    return np.mean(words,axis=0)

def distance_to_onehot(dist): # bin distances
    if dist < 5:
        i = dist
    elif dist < 8:
        i = 5
    elif dist < 16:
        i = 6
    elif dist < 32:
        i = 7
    elif dist < 64:
        i = 8
    else:
        i = 9
    return np.eye(1,10,i).flatten()

def process_file(f, POSITIVE_XY, NEGATIVE_XY):
    print f
    sentences = []
    curr_sentence = []
    sentence_num = 0
    doc_words = []
    found_mentions = []
    constructing_mentions = {}
    if f.readline().split()[0] == '#begin':
        while True:
            line = f.readline()
            if not line:
                break
            line = line.split()
            if not line:
                sentence_num += 1
                sentences.append(curr_sentence)
                curr_sentence = []
            elif line[0] == '#end':
                break
            else:
                corefs = line[-1].split('|')
                try:
                    line[3] = model[line[3]]
                except KeyError:
                    line[3] = np.zeros(50)
                for key in constructing_mentions:
                    constructing_mentions[key].words.append(line[3])
                for i in corefs:
                    if i.startswith('('):
                        if i.strip('()') not in constructing_mentions:
                            constructing_mentions[i.strip('()')] = mention(sentence_num, line[0], i.strip('()'), line[2], [line[3]], line[3])
                    if i.endswith(')'):
                        if i.strip('()') in constructing_mentions:
                            constructing_mentions[i.strip('()')].end = int(line[2])
                            constructing_mentions[i.strip('()')].last_word = line[3]
                            constructing_mentions[i.strip('()')].mention_average = average(constructing_mentions[i.strip('()')].words)
                            found_mentions.append(constructing_mentions.pop(i.strip('()')))
                doc_words.append(line[3])
                curr_sentence.append({"doc":line[0], "word_num":line[2], "word":line[3], "coref":line[-1], 'sentence':sentence_num})
        if constructing_mentions != {}:
            print constructing_mentions
            raise Exception('unfinished mention')
    else:
        return
    doc_average = average(doc_words)
    for ment in found_mentions:
        ment.doc_average = doc_average
        curr_sentence = sentences[ment.sentence]
        if ment.start - 1 >= 0:
            ment.previous_word = curr_sentence[ment.start - 1]['word']
        if ment.start - 2 >= 0:
            ment.prev_prev_word = curr_sentence[ment.start - 2]['word']
        if ment.end + 1 < len(curr_sentence):
            ment.following_word = curr_sentence[ment.end + 1]['word']
            ment.five_right = average([x['word'] for x in curr_sentence[ment.end+1:ment.end+6]])
        if ment.end + 2 < len(curr_sentence):
            ment.second_following_word = curr_sentence[ment.end + 2]['word']
        if ment.start > 0:
            ment.five_left = average([x['word'] for x in curr_sentence[max(0, ment.start - 5):ment.start]])
        ment.sentence_average = average([x['word'] for x in curr_sentence])
    for pair in itertools.combinations(found_mentions, 2):

        features = []
        features.append(pair[0].first_word)
        features.append(pair[0].last_word)
        features.append(pair[0].mention_average)
        features.append(pair[0].mention_average)
        features.append(pair[1].first_word)
        features.append(pair[1].last_word)
        features.append(pair[1].mention_average)
        features.append(pair[1].mention_average)

        features.append(pair[0].previous_word)
        features.append(pair[0].prev_prev_word)
        features.append(pair[0].following_word)
        features.append(pair[0].second_following_word)
        features.append(pair[0].five_left)
        features.append(pair[0].five_right)
        features.append(pair[0].sentence_average)
        features.append(pair[0].doc_average)
        features.append(pair[1].previous_word)
        features.append(pair[1].prev_prev_word)
        features.append(pair[1].following_word)
        features.append(pair[1].second_following_word)
        features.append(pair[1].five_left)
        features.append(pair[1].five_right)
        features.append(pair[1].sentence_average)
        features.append(pair[1].doc_average)

        features.append(distance_to_onehot(abs(pair[0].sentence - pair[1].sentence)))
        features.append(distance_to_onehot(abs(found_mentions.index(pair[0]) - found_mentions.index(pair[1]))))
        features.append(np.array([int(pair[0].start < pair[1].end and pair[1].start < pair[0].end)]))

        Y_VALUE = int(pair[0].mention_num == pair[1].mention_num)]
        features.append(np.array([Y_VALUE]))

        if Y_VALUE:
            POSITIVE_XY.append(np.concatenate(features))
        else:
            NEGATIVE_XY.append(np.concatenate(features))


POSITIVE_XY = []
NEGATIVE_XY = []
for root, _, files in os.walk('./data'):
    for f in files:
        with open(os.path.join(root,f)) as data_file:
            process_file(data_file, POSITIVE_XY, NEGATIVE_XY)

np.save('pos_coref_data.npy', np.array(XY_MATRIX))
np.save('neg_coref_data.npy', np.array(XY_MATRIX))
