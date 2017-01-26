import numpy as np


def generate_folds(selector_list, test=None):
    folds = []
    for elem in selector_list:
        lst = [v for v in selector_list]
        lst.remove(elem)
        fold = dict(train=lst, valid=[elem])
        if test is not None:
            fold['test'] = test
        # print fold
        folds.append(fold)
    return folds


class ClassificationResult(object):

    def __init__(self, name):
        self.name = name
        empty = np.array([])
        self.train_Y_real = empty
        self.train_Y_pred = empty
        self.test_Y_real = empty
        self.test_Y_pred = empty
        self.fold_scores = []
        
    def append_train(self, y_real, y_pred):
        self.train_Y_real = np.r_[self.train_Y_real, y_real]
        self.train_Y_pred = np.r_[self.train_Y_pred, y_pred]
        
    def append_test(self, y_real, y_pred):
        self.test_Y_real = np.r_[self.test_Y_real, y_real]
        self.test_Y_pred = np.r_[self.test_Y_pred, y_pred]

    def train_error(self):
        return self.error(self.train_Y_real, self.train_Y_pred)
        
    def test_error(self):
        return self.error(self.test_Y_real, self.test_Y_pred)
    
    @staticmethod
    def error(y_real, y_pred):
        return np.mean(y_real != y_pred)
