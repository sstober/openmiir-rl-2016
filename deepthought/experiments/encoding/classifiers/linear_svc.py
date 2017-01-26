import numpy as np
from sklearn.svm import SVC

from base import ClassifierFactory


def get_fold_score(X, Y, fold, c):
    train_idx, valid_idx = fold
    model = SVC(kernel='linear', C=c).fit(X[train_idx], Y[train_idx])

    if len(valid_idx) > 0:
        return model.score(X[valid_idx], Y[valid_idx])
    else:
        return model.score(X[train_idx], Y[train_idx])

def get_best_c(X, Y, folds, c_values, n_jobs=10):                           
    c_scores = []
    for c in c_values:
        # parallel version
        from joblib import Parallel, delayed

        fold_scores = Parallel(n_jobs=n_jobs)(delayed(get_fold_score)
                                              (X, Y, fold, c) for fold in folds)

        fold_scores = np.asarray(fold_scores)
        print 'fold_scores for c={:.4f}: {} mean={:.4f}'.format(c, fold_scores, fold_scores.mean())
        c_scores.append(np.mean(fold_scores))

    c_best = c_values[np.argmax(c_scores)]
    print 'best c={} with score {}'.format(c_best, np.max(c_scores))
    return c_best


class LinearSVCClassifierFactory(ClassifierFactory):

    def __init__(self, c_values=None):             
        super(LinearSVCClassifierFactory, self).__init__()
        if c_values is None:
            c_values = [0.01, 0.05, 0.1, 0.005, 0.001, 0.0005, 0.0001, 0.5, 1, 2]
        self.c_values = c_values
    
    def train(self, X, Y, idx_folds, hyper_params, model_prefix, verbose=False):
        # get best c through another round of cross-validation        
        best_c = get_best_c(X, Y, idx_folds, self.c_values)
        classifier = SVC(kernel='linear', C=best_c).fit(X, Y)
        return classifier, classifier.predict

    
class UntunedLinearSVCClassifierFactory(ClassifierFactory):
    """
    just for quick testing without tuning the C parameter
    """
    
    def __init__(self, c=0.01):             
        super(UntunedLinearSVCClassifierFactory, self).__init__()
        self.c = c

    def train(self, X, Y, idx_folds, hyper_params, model_prefix, verbose=False):
        classifier = SVC(kernel='linear', C=self.c).fit(X, Y)
        return classifier, classifier.predict