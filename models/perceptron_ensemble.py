""" Averaged perceptron classifier """

import cPickle
import random
from collections import defaultdict


class PerceptronEnsemble(object):
    """ See paper here:
        - http://www.ciml.info/dl/v0_8/ciml-v0_8-ch03.pdf
        - https://msdn.microsoft.com/en-us/library/azure/dn906036.aspx
        - https://svn.spraakdata.gu.se/repos/richard/pub/ml2014_web/m7.pdf

    """

    self.weights = {}  # dict more efficient than np array
    self.n_classes = set()  # n classes
    self._acc_totals = defaultdict(int)  # store avg values
    self._acc_tstamps = defaultdict(int)  # store when last updated
    self._num_seen = 0  # count

    def predict(features):
        ''' grab label as a simple mult between features and weights

            see perceptron definition:
            - https://en.wikipedia.org/wiki/Perceptron

        '''
        scores = defaultdict(float)
        for feat, value in features.items():
            if feat not in self.weights or value == 0:
                continue

            pred_weights = self.weights[feat]
            for label, weight in pred_weights.items():
                scores[label] += value * weight

        return max(self.n_classes,
                   key=lambda label:(scores[label],label))

    def update(out, pred, features):
        ''' simplified gradient descent '''

        def update_feature(c, f, w, v):
            param = (f, c)
            self._acc_totals[param] += (self._num_seen - self._acc_tstamps[param]) * w
            self._acc_tstamps[param] = self._num_seen
            self.weights[f][c] = w + v

        self._num_seen += 1

        # if correct, don't change weights
        if out == pred:
            return None

        # if incorrect, change weights to make it so
        for f in features:
            weights = self.weights.setdefault(f, {})
            update_feature(out, f, weights.get(out, 0.0), 1.0)
            update_feature(pred, f, weights.get(pred, 0.0), -1.0)

    def average_weights(self):
        ''' get average from ensemble using the accumulated
            data structures
        '''

        for feat, weights in self.weights.items():
            new_feat_weights = {}

            for clas, weight in weights.items():
                param = (feat, clas)
                total = self._acc_totals[param]
                total += (self._num_seen - self._acc_tstamps[param]) * weight
                averaged = round(total / float(self._num_seen), 3)

                if averaged:
                    new_feat_weights[clas] = averaged

            self.weights[feat] = new_feat_weights
        return None

    def save_weights(self, out_path):
        with open(path, 'w') as fp:
            pickle.dump(dict(self.weights), fp)

    def load_weights(self, in_path):
        with open(path) as fp:
            self.weights = pickle.load(fp)


def train_perceptron_ensemble(data, num_iters):
    ''' Example of usage '''

    model = PerceptronEnsemble()

    for i in range(num_iter):
        random.shuffle(data)

        for features, class_ in examples:
            scores = model.predict(features)
            pred, score = max(scores.items(), key=lambda i: i[1])
            if pred != class_:
                model.update(class_, pred, features)

    model.average_weights()
    return model

