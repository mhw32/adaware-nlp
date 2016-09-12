'''
Python implementation of SATZ, a sentence
boundary disambiguator using a feed
forward neural network.

Created by David Palmer (Edited by us)
https://arxiv.org/pdf/cmp-lg/9503019.pdf

1. load data
2. tokenization
3. part-of-speech lookup
4. descriptor array construction
5. classification

'''

import numpy as np
import nltk
import io
import tokenize

''' Tokenizer code:
    using python native tokenize libary
'''


def generate_tokens_from_string(text):
    ''' wrapper for tokenizer.generate_tokens
        returns a generator of 5 tuples
        1. token type
        2. token string
        3. (srow, scol)
        4. (erow, ecol)
        5. line number

        Args
        ----
        text : string
               the text to be tokenized

        Returns
        -------
        tokens : list
                 list of separated tokens

    '''
    '''
    # manually do it
    string_io = io.StringIO(text)
    tokens = tokenize.generate_tokens(
        string_io.readline)
    '''
    tokens = nltk.tokenize.word_tokenize(text)
    return tokens


def init_prior_pos_proba(
        lexicon=None, simplify_tags=True):
    ''' Set prior proba probabilities using
        the part-of-speech frequencies for
        every word in a given lexicon. Default
        lexicon used is the BROWN corpus.

        Args
        ----
        lexicon : list of tuples
                each tuple is (word, part-of-speech)

        simplify_tags : boolean
                many of the tags have additional info
                like IN-HL or HVZ*. Setting simplify_tags
                to True, ignores these additional tags.

        Returns
        -------
        freqs : dict
                dictionary of word mapping to
                its frequency
    '''
    tag_counts = dict()
    if lexicon is None:
        lexicon = cPickle.load(
            open('storage/tagged_brown_corpus.pkl', 'rb'))

    # get all tags (take only first part: 70 tags)
    tags_fd = nltk.FreqDist(
        tag for (word, tag) in lexicon)
    tags_lst = np.array(dict(tags_fd).keys())
    num_tags = tags_lst.shape[0]

    # loop through words and fill out freq
    for word, tag in lexicon:
        if not word in tag_counts:
            tag_counts[word] = np.zeros(num_tags)

        tag_idx = np.where(tags_lst == tag)[0][0]
        tag_counts[word][tag_idx] += 1

    cPickle.dump(tag_counts,
        open('storage/tag_brown_distribution.pkl', 'wb'))
    cPickle.dump(tags_lst,
        open('storage/tag_brown_order.pkl', 'wb'))

    return tag_counts


def lookup_prior_pos_proba(tokens, tag_counts=None):
    ''' The context around a word can be approximated
        by a single part-of-speech (POS) per word. We
        approximate this part-of-speech using the
        prior probabilities for all parts-of-speech
        per word.

        If unknown, follow a prescribed list of
        assumptions.

        Args
        ----
        tokens : list
                 tokens abstracted from text

        Returns
        -------
        probas : 2D array
                 probabilities per token per P`OS
    '''

    if tag_counts is None:
        tag_counts = cPickle.load(
            open('storage/tagger_brown_corpus.pkl', 'rb'))

    num_tags = tag_counts.values[0].shape[0]

    # heuristics
    proper_noun_tag_idx =
    lemmatizer = nltk.stem.WordNetLemmatizer()
    has_number = lambda x: any(
        char.isdigit() for char in x)
    has_eos_punc = lambda x: x[0] in ".!?".split()
    reduce_morpho = lambda x: lemmatizer.lemmatize(x)
    is_abbrev = lambda x: '.' in x

    for token in tokens:
        if token in tag_counts:
            cur_tag_count = tag_counts[token]
            ''' capitalized words in lexicon but not
                registered as proper nouns can still be
                proper nouns with 0.5 probability
            '''
        else:
            ''' If a word is not in the lecxicon, use a
                list of heuristics to try to approx the
                tag probabilities.

                - if contains a 0-9, assume number
                - if begins with a ".!?", should be
                  end-of-sentence punctuation
                - handle morphological endings
                - internal period is an abbreviation
                - if capital word
                    - 0.9 pr of being a proper noun
                - uniform distribution over tags
            '''
            if has_number(token):
            elif has_eos_punc(token):
            elif reduce_morpho(token) in tag_counts:
            elif token.isupper():
                if np.random.uniform() >= 0.1:

            else:
                cur_tag_dcount = np.ones(num_tags)

        cur_tag_distrib = counts / np.sum(counts)
