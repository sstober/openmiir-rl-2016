import numpy as np

from deepthought.experiments.encoding.experiment_templates.base import NestedCVExperimentTemplate


class End2EndBaseline(NestedCVExperimentTemplate):
    
    def __init__(self,
                 job_id,
                 hdf5name,
                 fold_generator,
                 pipeline_factory,
                 **kwargs):

        self.pipeline_factory = pipeline_factory
        super(End2EndBaseline, self).__init__(job_id, hdf5name, fold_generator, **kwargs)

    def pretrain_encoder(self, *args, **kwargs):

        def dummy_encoder_fn(indices):
            if type(indices) == np.ndarray:
                indices = indices.tolist()  # ndarray is not supported as indices

            # read the chunk of data for the given indices
            state = self.full_hdf5.open()
            data = self.full_hdf5.get_data(request=indices, state=state)
            self.full_hdf5.close(state)

            # get only the features source
            source_idx = self.full_hdf5.sources.index('features')
            data = np.ascontiguousarray(data[source_idx])

            return data

        return dummy_encoder_fn

    def run(self, verbose=False):
        from deepthought.experiments.encoding.classifiers.simple_nn import SimpleNNClassifierFactory
        cls_factory = SimpleNNClassifierFactory(self.pipeline_factory)
        super(End2EndBaseline, self).run(classifiers=(('mlp', cls_factory),), verbose=verbose)
