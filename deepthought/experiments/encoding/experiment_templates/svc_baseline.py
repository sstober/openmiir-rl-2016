import numpy as np

from deepthought.experiments.encoding.experiment_templates.base import NestedCVExperimentTemplate


class SVCBaseline(NestedCVExperimentTemplate):

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

            # apply optional channel mean
            if self.hyper_params['ch_mean'] is True:
                data = data.mean(axis=1)  # bc01 format -> will result in b01 format

            return data

        return dummy_encoder_fn

    def run(self, verbose=False):
        from deepthought.experiments.encoding.classifiers.linear_svc import LinearSVCClassifierFactory
        super(SVCBaseline, self).run(classifiers=(('linear_svc', LinearSVCClassifierFactory()),), verbose=verbose)
