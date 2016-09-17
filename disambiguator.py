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
import cPickle

from constants import *
import create_toy_data as ctd
from collections import defaultdict

# load in libraries for NN
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad
from autograd.util import flatten
from optimizers import adam


def get_loc_in_array(value, array):
    find = np.where(array == value)[0]
    # print("value: {}, array: {}, find: {}".format(value, array, find))
    if find.shape[0] > 0:
        return find[0]
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
    if lexicon is None:
        lexicon = ctd.load_data(
            ctd.brown_generator(), return_tags=True)

    descriptor_array = np.array([
        NOUN,
        VERB,
        ARTICLE,
        MODIFIER,
        CONJUNCTION,
        PRONOUN,
        PREPOSITION,
        PROPER_NOUN,
        NUMBER,
        COMMA_SEMICOLON,
        LEFT_PAREN,
        RIGHT_PAREN,
        NON_PUNC_CHAR,
        POSSESSIVE,
        COLON_DASH,
        ABBREV,
        EOS_PUNC,
        OTHERS,
        IS_UPPER,
        FOLLOWS_EOS_PUNC
    ])

    cat_lookup = group_categories()
    num_tags = descriptor_array.shape[0]
    tag_counts = defaultdict(lambda: np.zeros(num_tags))
    tag_counts = defaultdict(lambda: np.zeros(num_tags))

    # loop through words and fill out freq
    for word, tag_bunch in lexicon:
        if tag_bunch == '--':
            tag_idx = get_loc_in_array(COLON_DASH, descriptor_array)
            tag_counts[word][tag_idx] += 1
            continue

        if '*' in tag_bunch:
            tag_idx = get_loc_in_array(CONJUNCTION, descriptor_array)
            tag_counts[word][tag_idx] += 1
            tag_bunch = tag_bunch.replace('*', '')

        for tt in tag_bunch.split('-'):
            for single_tag in tt.split('+'):
                # print("tag_bunch: {} || single_tag: {}".format(tag_bunch, single_tag))

                reduced_tags = cat_lookup(single_tag)

                if reduced_tags is None and single_tag in ['``', "''", '', "'", 'HL', 'FW', 'NC', 'TL']:
                    # tags known to be weird, but also okay
                    continue
                elif reduced_tags is None and \
                    single_tag in ['RB$', 'DT$', 'NR$', 'AP$', 'JJ$', 'CD$', 'N', 'T', 'PP', 'NIL']:
                    # odd tags, probably should really figure them out
                    # datasets/brown/raw/cj31 -> PPSS+BER-N
                    # datasets/brown/raw/cj37 -> FW-IN+AT-T
                    # datasets/brown/raw/ce01 -> WDT+BER+PP, PP is probably personal pronoun
                    continue
                elif reduced_tags is None:
                    # not a "known exception", print
                    print("word: {}, tag_bunch not found: {}".format(word, tag_bunch))
                    continue

                for reduced_tag in reduced_tags:
                    tag_idx = get_loc_in_array(reduced_tag, descriptor_array)
                    tag_counts[word][tag_idx] += 1

    with open('storage/brown_tag_distribution.pkl', 'wb') as f:
        cPickle.dump(dict(tag_counts), f)
    with open('storage/brown_tag_order.pkl', 'wb') as f:
        cPickle.dump(descriptor_array, f)

    return tag_counts


def get_descriptor_arrays(
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

        tag_order : list of strings
                    order of P-O-S in frequency array

        Returns
        -------
        probas : 2D array
                 probabilities per token per P-O-S
    '''

    if tag_counts is None:
        tag_counts = cPickle.load(
            open('storage/brown_tag_distribution.pkl', 'rb'))

    if tag_order is None:
        tag_order = cPickle.load(
            open('storage/brown_tag_order.pkl', 'rb'))

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

    desc_arrays = []
    prev_token = None
    for token_cnt, token in enumerate(tokens):
        if token_cnt % 10000 == 0:
            print('----------------------')
            print('processing: token ({}/{})'.format(token_cnt, len(tokens)))
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
                cur_tag_count[get_loc_in_array(NUMBER, tag_order)] += 1
            elif has_eos_punc(token):
                cur_tag_count[get_loc_in_array(EOS_PUNC, tag_order)] += 1
            elif is_abbrev(token):
                cur_tag_count[get_loc_in_array(ABBREV, tag_order)] += 1
            elif has_hyphen(token):
                cur_tag_count[get_loc_in_array(OTHERS, tag_order)] += 1
                # TODO: don't know how to lookup this...
            else:
                cur_tag_count = np.ones(num_tags)


        # zero out IS_UPPER and FOLLOWS_EOS_PUNC before normalize
        cur_tag_count[get_loc_in_array(IS_UPPER, tag_order)] = 0
        cur_tag_count[get_loc_in_array(FOLLOWS_EOS_PUNC, tag_order)] = 0

        # divide counts to get probabilities
        cur_tag_distrib = cur_tag_count / np.sum(cur_tag_count)

        ''' capitalized words in lexicon but not
            registered as proper nouns can still be
            proper nouns with 0.5 probability. If not
            in lexicon, use 0.9 probability.
        '''
        if is_upper(token):
            proper_pr = 0.5 if token_in_lexicon else 0.9
            cur_tag_distrib *= (1 - proper_pr)
            cur_tag_distrib[get_loc_in_array(
                PROPER_NOUN, tag_order)] = proper_pr

            cur_tag_distrib[get_loc_in_array(IS_UPPER, tag_order)] = 1

        if prev_token and has_eos_punc(prev_token):
            cur_tag_distrib[get_loc_in_array(FOLLOWS_EOS_PUNC, tag_order)] = 1

        prev_token = token
        desc_arrays.append(cur_tag_distrib)

    return np.array(desc_arrays)


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
    mapper = dict()
    mapper["."] = [EOS_PUNC]                      # ending punctuation
    mapper["("] = [LEFT_PAREN]                    # left parentheses
    mapper[")"] = [RIGHT_PAREN]                   # right parentheses
    mapper["*"] = [CONJUNCTION]                   # conjunction
    mapper["--"] = [COLON_DASH]                   # dash
    mapper[","] = [COMMA_SEMICOLON]               # comma
    mapper[":"] = [COLON_DASH]                    # colon
    mapper["ABL"] = [MODIFIER]                    # pre-qualifier [quite, rather]
    mapper["ABN"] = [MODIFIER]                    # pre-quantifier [half, all]
    mapper["ABR"] = [ABBREV]                      # abbreviation CUSTOM TAG
    mapper["ABX"] = [MODIFIER]                    # pre-quantifier [both]
    mapper["AP"] = [OTHERS]                       # post-determiner [many, several, next]
    mapper["AT"] = [ARTICLE]                      # article
    mapper["BE"] = [VERB]                         # be
    mapper["BED"] = [VERB]                        # were
    mapper["BEDZ"] = [VERB]                       # was
    mapper["BEG"] = [VERB]                        # being
    mapper["BEM"] = [VERB]                        # am
    mapper["BEN"] = [VERB]                        # been
    mapper["BER"] = [VERB]                        # are, art
    mapper["BEZ"] = [VERB]                        # is
    mapper["CC"] = [CONJUNCTION]                  # coordinating conjunction
    mapper["CD"] = [NUMBER]                       # cardinal numeral
    mapper["CS"] = [CONJUNCTION]                  # subordinating conjunction
    mapper["DO"] = [VERB]                         # do
    mapper["DOD"] = [VERB]                        # did
    mapper["DOZ"] = [VERB]                        # does
    mapper["DT"] = [OTHERS]                       # plural determiner [these, those]
    mapper["DTI"] = [OTHERS]                      # singular/plural determiner/quantifier [some, any]
    mapper["DTS"] = [OTHERS]                      # plural determiner
    mapper["DTX"] = [CONJUNCTION]                 # determiner/double conjunction [either]
    mapper["EX"] = [OTHERS]                       # existentil there
    mapper["FW"] = [OTHERS]                       # foreign word (hyphenated before regular tag)
    mapper["HL"] = [OTHERS]                       # word occurring in headline (hyphenated after regular tag)
    mapper["HV"] = [VERB]                         # have
    mapper["HVD"] = [VERB]                        # had (past tense)
    mapper["HVG"] = [VERB]                        # having
    mapper["HVN"] = [VERB]                        # had (past participle)
    mapper["HVZ"] = [VERB]                        # has
    mapper["IN"] = [PREPOSITION]                  # preposition
    mapper["JJ"] = [MODIFIER]                     # adjective
    mapper["JJR"] = [MODIFIER]                    # comparative adjective
    mapper["JJS"] = [MODIFIER]                    # semantically superlative adjective [chief, top]
    mapper["JJT"] = [MODIFIER]                    # morphologically superlative adjective [ biggest ]
    mapper["MD"] = [OTHERS]                       # modal auxiliary [can, should, will]
    mapper["NC"] = [ARTICLE]                      # cited word (hypenated after regular tag)
    mapper["NN"] = [NOUN]                         # singular or mass noun
    mapper["NN$"] = [NOUN, POSSESSIVE]            # possessive singular noun
    mapper["NNS"] = [NOUN]                        # plural noun
    mapper["NNS$"] = [NOUN, POSSESSIVE]           # possessive plural noun
    mapper["NP"] = [PROPER_NOUN]                  # proper noun or part of name phrase
    mapper["NP$"] = [PROPER_NOUN, POSSESSIVE]     # possessive proper noun
    mapper["NPS"] = [PROPER_NOUN]                 # proper plural noun
    mapper["NPS$"] = [PROPER_NOUN, POSSESSIVE]    # possessive plural proper noun
    mapper["NR"] = [NOUN, MODIFIER]               # adverbial noun [home, today, west]
    mapper["NRS"] = [NOUN, MODIFIER]              # plural adverbial noun
    mapper["OD"] = [NUMBER]                       # ordinal numeral [first, 2nd]
    mapper["PN"] = [PRONOUN]                      # nominal pronoun [everybody, nothing]
    mapper["PN$"] = [PRONOUN, POSSESSIVE]         # possessive nominal pronoun
    mapper["PP$"] = [PRONOUN, POSSESSIVE]         # possessive personal pronoun [my, our]
    mapper["PP$$"] = [PRONOUN, POSSESSIVE]        # second (nominal) possessive pronoun [mine, ours]
    mapper["PPL"] = [PRONOUN]                     # singular reflexive/intensive personal pronoun [myself]
    mapper["PPLS"] = [PRONOUN]                    # plural reflexive/intensive personal pronoun [ourselves]
    mapper["PPO"] = [PRONOUN]                     # objective personal pronoun [me, him, it, them]
    mapper["PPS"] = [PRONOUN]                     # 3rd. singular nominative pronoun  [he, she, it, one]
    mapper["PPSS"] = [PRONOUN]                    # other nominative personal pronoun [I, we, they, you]
    mapper["QL"] = [MODIFIER]                     # qualifier [very, fairly]
    mapper["QLP"] = [MODIFIER]                    # post-qualifier [enough, indeed]
    mapper["RB"] = [MODIFIER]                     # adverb
    mapper["RBR"] = [MODIFIER]                    # comparative adverb
    mapper["RBT"] = [MODIFIER]                    # superlative adverb
    mapper["RN"] = [MODIFIER]                     # nominal adverb [here then, inddors]
    mapper["RP"] = [MODIFIER]                     # adverb/participle [about, off, up]
    mapper["TL"] = [OTHERS]                       # word occuring in title (hyphenated after regular tag)
    mapper["TO"] = [OTHERS]                       # infinitive marker to
    mapper["UH"] = [OTHERS]                       # interjection, exclamation
    mapper["UNK"] = [OTHERS]                      # unknown CUSTOM TAG
    mapper["VB"] = [VERB]                         # verb, base form
    mapper["VBD"] = [VERB]                        # verb, past tense
    mapper["VBG"] = [VERB]                        # verb, present participle/gerund
    mapper["VBN"] = [VERB]                        # verb, past participle
    mapper["VBZ"] = [VERB]                        # verb, 3rd. singular present
    mapper["WDT"] = [OTHERS]                      # wh- determiner [what, which]
    mapper["WP$"] = [PRONOUN, POSSESSIVE]         # possessive wh- pronoun [whose]
    mapper["WPO"] = [PRONOUN]                     # objective wh- pronoun [whom, which, that]
    mapper["WPS"] = [PRONOUN]                     # nominative wh- pronoun [who, which, that]
    mapper["WQL"] = [MODIFIER]                    # wh- qualifier [how]
    mapper["WRB"] = [MODIFIER]                    # wh- adverb [how, where, when]

    f = lambda x: mapper[x] if x in mapper else None
    return f


def safe_index(A, s, e, pad=False):
    # safe indexing
    num_col = A.shape[1]
    pre = np.zeros((0, num_col))
    post = np.zeros((0, num_col))

    if s < 0:
        pre = np.zeros((-s, num_col))
        s = 0

    if e > A.shape[0]:
        post = np.zeros((e - A.shape[0], num_col))
        e = A.shape[0]

    if pad:
        return np.concatenate((pre, A[s:e], post))
    return A[s:e]


def make_grams(darrays, labels, num_grams, target_tag=EOS_PUNC, tag_order=None):
    ''' Given darrays, only take the ones that have a certain
        label. Then take k/2 neighbors from either side.

        Args
        ----
        darrays : np array
                 descriptor array

        labels : np array
                 output array

        num_grams : integer
                    num of surrounding grams
                    if num_grams = 3, 3 on each side
                    will be taken.

        target_tag : string
                     what tag to index to look for

        tag_order : array
                    list of tags

        Returns
        -------
        new_darrays : np array
                     gram'd descriptor array

        new_labels : np array
                     gram'd labels
    '''

    if tag_order is None:
        tag_order = cPickle.load(
            open('storage/brown_tag_order.pkl', 'rb'))

    if not target_tag in tag_order:
        raise ValueError('Target tag not found in tag order.')

    target_idx = np.where(tag_order == target_tag)[0][0]
    eos_idx = np.where(darrays[:,target_idx] > 0)[0]

    new_darrays = np.zeros((eos_idx.shape[0],darrays.shape[1]*(2*num_grams + 1)))
    new_labels = np.zeros(eos_idx.shape[0], 1)

    for i, idx in enumerate(eos_idx):
        if i % 5000 == 0:
            print('---------------------')
            print('finding: end-of-sentence ({}/{})'.format(i, eos_idx.shape[0]))

        sliced_darrays = safe_index(
            darrays, idx-num_grams, idx+num_grams+1, pad=True).flatten()
        new_darrays[i, :] = sliced_darrays
        new_labels[i, 0] = labels[idx]

    return (new_darrays, new_labels)


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


def main():
    ''' Paper found parameters to be efficient:
        6 word contexts --> 3 num_grams
        2 hidden_units
    '''
    lexicon = ctd.load_data(
        ctd.brown_generator(), return_sent_labels=True)

    # not efficient --> change me
    data = np.array([[t, l] for t, l in lexicon])
    tokens = data[:, 0]
    labels = data[:, 1].astype(int)

    # get features
    darrays = get_descriptor_arrays(tokens)

    # put into grams (give context)
    darrays, labels = make_grams(
        darrays, labels, 3, target_tag=EOS_PUNC)

    (tr_inputs, te_inputs), (tr_outputs, te_outputs) = ctd.split_data(
        darrays, out_data=labels, frac=0.80)

    trained_weights = train_nn(
        tr_inputs, tr_outputs, 2,
        batch_size=256, param_scale=0.1,
        num_epochs=5, step_size=0.001, L2_reg=1.0)

    # save the weights
    np.save('storage/trained_weights.np', trained_weights)

    te_preds = neural_net_predict(trained_weights, te_inputs)
    print("auc: {}".format(ctd.get_auc(te_outputs, te_preds)))
