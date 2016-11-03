''' Name Entity Recognition

    Model: Convolutional Neural Network
        - Conv --> Pool --> Conv --> Pool --> FNN
        - input: n-gram*300
        - conv(1): 4 kernels (2*41 size)
        - pool(1): horiz. and vert pooling dim = 2
        - conv(2): 8 kernels (1*21 size)
        - pool(2): horiz = 2, vert = 1
        - 256 HU w/ 0.5 proba dropout

    Heuristics:
    1. Encoding: Use Word2Vec/GloVe
    2. N-gram
        Train 3,5,7,9 and use a mixture of
        experts when doing predictions

        Bayesian Hierarchical Mixtures of Experts
    3. Possible data augmentation:
        Replace words with synonyms

    Training dataset: look in the drive for a
    news_tagged_data.txt.
'''

