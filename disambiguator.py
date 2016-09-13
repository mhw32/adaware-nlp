#!/usr/bin/env python

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

from __future__ import absolute_import, division
from __future__ import print_function

import numpy as np
import nltk
import io
import tokenize
import cPickle

# load in libraries for NN
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad
from autograd.util import flatten
from optimizers import adam


def generate_tokens_from_string(text):
    ''' wrapper for nltk.word_tokenize
        returns a list of strings

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


def get_loc_in_array(value, array):
    find = np.where(array == value)
    if find.shape[0] > 0:
        return find[0][0]
    raise RuntimeError("Couldn't find value: {} in array".format(value))


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
        with open('storage/tagged_brown_corpus.pkl', 'rb') as f:
            lexicon = cPickle.load(f)

    # get all tags (take only first part: 70 tags)
    tags_fd = nltk.FreqDist(
        tag for (word, tag) in lexicon)
    tags_lst = np.array(dict(tags_fd).keys())
    ''' three tags are added:
        - PEP : possible end-of-sentence punctuation
        - UHW : unknown hyphenated word
        - ABR : abbreviation
    '''
    # word_tokenize splits punctuation, so ?! isn't valid
    # might need to do some looping to smash adjacent punctuation together?
    tags_lst = np.concatenate((tags_lst, ['UHW', 'ABR']))
    num_tags = tags_lst.shape[0]

    # loop through words and fill out freq
    for word, tag in lexicon:
        if not word in tag_counts:
            tag_counts[word] = np.zeros(num_tags)

        tag_idx = get_loc_in_array(tag, tags_lst)
        tag_counts[word][tag_idx] += 1

    with open('storage/tag_brown_distribution.pkl', 'wb') as f:
        cPickle.dump(tag_counts, f)
    with open('storage/tag_brown_order.pkl', 'wb') as f:
        cPickle.dump(tags_lst, f)

    return tag_counts


def lookup_prior_pos_proba(
        tokens, tag_counts=None, tag_order=None):
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

        tag_counts : dict of word-to-subdict mapping
                     subdict is P-O-S to counts mapping
                     for the brown corpus

        tag_order : list of strings
                    order of P-O-S in frequency array

        Returns
        -------
        probas : 2D array
                 probabilities per token per P-O-S
    '''

    if tag_counts is None:
        with open('storage/tag_brown_distribution.pkl', 'rb') as f:
            tag_counts = cPickle.load(f)

    if tag_order is None:
        with open('storage/tag_brown_order.pkl', 'rb') as f:
            tag_order = cPickle.load(f)

    num_tags = tag_order.shape[0]

    # heuristics
    lemmatizer = nltk.stem.WordNetLemmatizer()
    has_number = lambda x: any(
        char.isdigit() for char in x)
    has_eos_punc = lambda x: x[0] in ".!?".split()
    reduce_morpho = lambda x: lemmatizer.lemmatize(x)
    is_abbrev = lambda x: '.' in x
    has_hyphen = lambda x: '-' in x

    def is_plural(x):
        lemma = lemmatizer.lemmatize(x, 'n')
        plural = True if word is not lemma else False # aka plural != lemma
        return plural

    def is_upper(x):
        if len(x) >= 0:
            return x[0].isupper()
        return False

    result = []
    for token in tokens:
        token_in_lexicon = False
        if token in tag_counts:
            token_in_lexicon = True
            cur_tag_count = tag_counts[token]
        elif reduce_morpho(token) in tag_counts:
            token_in_lexicon = True
            cur_tag_count = tag_counts[reduce_morpho(token)]
        else:
            ''' If a word is not in the lexicon, use a
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
            cur_tag_count = np.zeros(num_tags)
            if has_number(token):
                cur_tag_count[get_loc_in_array('CD', tag_order)] += 1
            elif has_eos_punc(token):
                cur_tag_count[get_loc_in_array('.', tag_order)] += 1
            elif is_abbrev(token):
                cur_tag_count[get_loc_in_array('ABR', tag_order)] += 1
            elif has_hyphen(token):
                cur_tag_count[get_loc_in_array('UHW', tag_order)] += 1
            else:
                cur_tag_count = np.ones(num_tags)

        # divide counts to get probabilities
        cur_tag_distrib = cur_tag_count / np.sum(counts)

        ''' capitalized words in lexicon but not
            registered as proper nouns can still be
            proper nouns with 0.5 probability. If not
            in lexicon, use 0.9 probability.
        '''
        if is_upper(token):
            proper_pr = 0.5 if token_in_lexicon else 0.9
            cur_tag_distrib *= (1 - proper_pr)
            code = 'NNPS' if is_plural(token) else 'NNP'
            cur_tag_distrib[get_loc_in_array(
                code, tag_order)] = proper_pr
        
        result.append(cur_tag_distrib)

    return np.array(result)


def group_categories():
    '''
    Mapping into these words:

    noun, verb, article, modifier,
    conjunction, pronoun, preposition,
    proper noun, number, comma or semicolon,
    left parentheses, right parentheses,
    non-punctuation character, possessive,
    colon or dash, abbreviation,
    sentence-ending punctuation, others
    '''
    noun = 'noun'
    verb = 'verb'
    article = 'article'
    modifier = 'modifier'
    conjunction = 'conjunction'
    pronoun = 'pronoun'
    preposition = 'preposition'
    proper_noun = 'proper noun'
    number = 'number'
    comma_semicolon = 'comma or semicolon'
    left_paren = 'left parentheses'
    right_paren = 'right parentheses'
    non_punctuation_char = 'non punctuation character'
    possessive = 'possessive'
    colon_dash = 'color or dash'
    abbrev = 'abbreviation'
    ending_punc = 'sentence ending punctuation'
    others = 'others'

    mapper = dict()
    # mapper["$"] = ''
    # mapper["''"] = ''
    # mapper["("] = ''
    # mapper[")"] = ''
    # mapper[","] = ''
    # mapper["."] = ''
    # mapper[":"] = ''
    # mapper["ABR"] = ''
    # mapper["CC"] = ''
    # mapper["CD"] = ''
    # mapper["DT"] = ''
    # mapper["EX"] = ''
    # mapper["FW"] = ''
    # mapper["IN"] = ''
    # mapper["JJ"] = ''
    # mapper["JJR"] = ''
    # mapper["JJS"] = ''
    # mapper["LS"] = ''
    # mapper["MD"] = ''
    # mapper["NN"] = ''
    # mapper["NNP"] = ''
    # mapper["NNPS"] = ''
    # mapper["NNS"] = ''
    # mapper["PDT"] = ''
    # mapper["POS"] = ''
    # mapper["PRP"] = ''
    # mapper["PRP$"] = ''
    # mapper["RB"] = ''
    # mapper["RBR"] = ''
    # mapper["RBS"] = ''
    # mapper["RP"] = ''
    # mapper["SYM"] = ''
    # mapper["TO"] = ''
    # mapper["UH"] = ''
    # mapper["UHW"] = ''
    # mapper["VB"] = ''
    # mapper["VBD"] = ''
    # mapper["VBG"] = ''
    # mapper["VBN"] = ''
    # mapper["VBP"] = ''
    # mapper["VBZ"] = ''
    # mapper["WDT"] = ''
    # mapper["WP"] = ''
    # mapper["WP$"] = ''
    # mapper["WRB"] = ''
    # mapper["``"] = ''

    mapper["."] = ending_punc
    mapper["("] = left_paren
    mapper[")"] = right_paren
    mapper["*"] = conjunction
    mapper["--"] = colon_dash       # dash
    mapper[","] = comma_semicolon   # comma
    mapper[":"] = colon_dash        # colon
    mapper["ABL"] = modifier        # pre-qualifier 
    mapper["ABN"] = modifier        # pre-quantifier
    mapper["ABX"] = modifier        # pre-quantifier
    mapper["AP"] = modifier         # post-determiner ?? i guess
    mapper["AT"] = article
    mapper["BE"] = ''
    mapper["BED"] = ''
    mapper["BEDZ"] = ''
    mapper["BEG"] = ''
    mapper["BEM"] = ''
    mapper["BEN"] = ''
    mapper["BER"] = ''
    mapper["BEZ"] = ''
    mapper["CC"] = ''
    mapper["CD"] = ''
    mapper["CS"] = ''
    mapper["DO"] = ''
    mapper["DOD"] = ''
    mapper["DOZ"] = ''
    mapper["DT"] = ''
    mapper["DTI"] = ''
    mapper["DTS"] = ''
    mapper["DTX"] = ''
    mapper["EX"] = ''
    mapper["FW"] = ''
    mapper["HL"] = ''
    mapper["HV"] = ''
    mapper["HVD"] = ''
    mapper["HVG"] = ''
    mapper["HVN"] = ''
    mapper["HVZ"] = ''
    mapper["IN"] = ''
    mapper["JJ"] = ''
    mapper["JJR"] = ''
    mapper["JJS"] = ''
    mapper["JJT"] = ''
    mapper["MD"] = ''
    mapper["NC"] = ''
    mapper["NN"] = ''
    mapper["NN$"] = ''
    mapper["NNS"] = ''
    mapper["NNS$"] = ''
    mapper["NP"] = ''
    mapper["NP$"] = ''
    mapper["NPS"] = ''
    mapper["NPS$"] = ''
    mapper["NR"] = ''
    mapper["NRS"] = ''
    mapper["OD"] = ''
    mapper["PN"] = ''
    mapper["PN$"] = ''
    mapper["PP$"] = ''
    mapper["PP$$"] = ''
    mapper["PPL"] = ''
    mapper["PPLS"] = ''
    mapper["PPO"] = ''
    mapper["PPS"] = ''
    mapper["PPSS"] = ''
    mapper["QL"] = ''
    mapper["QLP"] = ''
    mapper["RB"] = ''
    mapper["RBR"] = ''
    mapper["RBT"] = ''
    mapper["RN"] = ''
    mapper["RP"] = ''
    mapper["TL"] = ''
    mapper["regular"] = ''
    mapper["TO"] = ''
    mapper["UH"] = ''
    mapper["VB"] = ''
    mapper["VBD"] = ''
    mapper["VBG"] = ''
    mapper["VBN"] = ''
    mapper["VBZ"] = ''
    mapper["WDT"] = ''
    mapper["WP$"] = ''
    mapper["WPO"] = ''
    mapper["WPS"] = ''
    mapper["WQL"] = ''
    mapper["WRB"] = ''

    f = lambda x: mapper[x] if x in mapper else None
    return f

'''
Neural network setup to map tokens into sentence
or non-sentence endings (binary classification)

'''


def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """
    Build a list of (weights, biases) tuples,
    one for each layer in the net.

    """
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]


def neural_net_predict(params, inputs):
    """
    Implements a deep neural network for classification.
    params is a list of (weights, bias) tuples.
    inputs is an (N x D) matrix.
    returns normalized class log-probabilities.

     """
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = np.tanh(outputs)
    return outputs - logsumexp(outputs, axis=1, keepdims=True)


def l2_norm(params):
    """
    Computes l2 norm of params by flattening them into a vector.

    """
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)


def log_posterior(params, inputs, targets, L2_reg):
    log_prior = -L2_reg * l2_norm(params)
    log_lik = np.sum(neural_net_predict(params, inputs) * targets)
    return log_prior + log_lik


def accuracy(params, inputs, targets):
    target_class = np.argmax(targets, axis=1)
    predicted_class = np.argmax(neural_net_predict(params, inputs), axis=1)
    return np.mean(predicted_class == target_class)


def train_nn(
        tr_obs_set, tr_out_set, num_hiddens,
        batch_size=256, param_scale=0.1,
        num_epochs=5, step_size=0.001, L2_reg=1.0):

    num_input_dims = tr_obs_set.shape[1]
    layer_sizes = [num_input_dims, num_hiddens, 1]
    init_params = init_random_params(param_scale, layer_sizes)
    num_batches = int(np.ceil(tr_obs_set.shape[0] / batch_size))

    def batch_indices(iter):
        idx = iter % num_batches
        return slice(idx * batch_size, (idx+1) * batch_size)

    # Define training objective
    def objective(params, iter):
        idx = batch_indices(iter)
        return -log_posterior(
            params, tr_obs_set[idx], tr_out_set[idx], L2_reg)

    # Get gradient of objective using autograd.
    objective_grad = grad(objective)

    print("     Epoch     |    Train accuracy  |       Test accuracy  ")

    def print_perf(params, iter, gradient):
        if iter % num_batches == 0:
            train_acc = accuracy(params, tr_obs_set, tr_out_set)
            test_acc = accuracy(params, test_images, test_labels)
            print("{:15}|{:20}|{:20}".format(
                iter//num_batches, train_acc, test_acc))

    # The optimizers provided can optimize lists, tuples, or dicts of
    # parameters.
    optimized_params = adam(
        objective_grad, init_params, step_size=step_size,
        num_iters=num_epochs * num_batches, callback=print_perf)

    return optimized_params
