''' Averaged Perceptron for POS tagging.
    Interesting approach that works as an ensemble of simpler
    neural networks instead of a complex model like (BLSTM).

    This tagger is very FAST and pretty much on par with
    state-of-the-art stuff.

    We should end up using a voting average system with all
    our different POS tagging methods.

    Perceptron (not even a NN yet) learning:

    - Given a training data pt (feature , POS-tag)
    - Get value of POS tag using current weights
    - if wrong, +1 to weight for right class and -1 for not

    But this doesn't generalize well. So instead of returning
    the final weights, which overfit a lot, we return the
    average of the weights.

    Preprocessing (regularization):

    - disambiguation + tokenization
    - all words are lower cased
    - digits in range 1800-2100 --> !YEAR
    - other digit strings --> !DIGITS

    This is also an autoregressive model. The predictions for one
    word become the features for the next.

    Possible problem:

    - Beam-search would be better than just looking at your
    previous neighbors. But not worth it.
    - This model is also bad at multi-tagging
'''

