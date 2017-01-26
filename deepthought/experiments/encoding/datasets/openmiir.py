from deepthought.experiments.encoding.experiment_templates.base import NestedCVFoldGenerator
from deepthought.util.crossvalidation_util import generate_folds


class OpenMIIRNestedCVFoldGenerator(NestedCVFoldGenerator):
    def __init__(self, subjects=None, **kwargs):
        self.subjects = ['P01', 'P04', 'P06', 'P07', 'P09', 'P11', 'P12', 'P13', 'P14']
        self.subjects = self.subjects if subjects is None else subjects
        super(OpenMIIRNestedCVFoldGenerator, self).__init__(**kwargs)

    def get_outer_cv_folds(self):
        return generate_folds(self.subjects)

    def get_inner_cv_folds(self, outer_fold):
        return generate_folds(xrange(5))

    def get_fold_selectors(self, outer_fold=None, inner_fold=None, base_selectors=None):
        selectors = self.base_selectors.copy() if base_selectors is None else base_selectors.copy()

        if outer_fold is not None:
            selectors['subject'] = outer_fold
        if inner_fold is not None:
            selectors['trial_no'] = inner_fold

        return selectors
