'''
Functions for evaluating the performance of machine learning
models, like AUC/ROC.

'''

from sklearn.metrics import roc_curve, auc

def get_auc(outputs, probas):
    ''' AUC is a common metric for binary classification
        methods by comparing true & false positive rates

        Args
        ----
        outputs : numpy array
                 true outcomes (OxTxN)

        probas : numpy array
                 predicted probabilities (OxTxN)

        Returns
        -------
        auc : integer
    '''

    fpr, tpr, _ = roc_curve(outputs, probas[:, 1])
    return auc(fpr, tpr)
