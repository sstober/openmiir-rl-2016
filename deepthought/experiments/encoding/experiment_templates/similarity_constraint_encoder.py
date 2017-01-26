import logging

from deepthought.experiments.encoding.experiment_templates.base import GenericNNEncoderExperiment, NestedCVExperimentTemplate

log = logging.getLogger('deepthought.experiments')


class SimilarityConstraintEncoderExperiment(GenericNNEncoderExperiment):

    _default_params = NestedCVExperimentTemplate._default_params.copy()
    _default_params['group_attribute'] = 'subject'

    def __init__(self,
                 job_id,
                 hdf5name,
                 fold_generator,
                 encoder_pipeline_factory,
                 **kwargs):

        super(SimilarityConstraintEncoderExperiment, self).__init__(
            job_id, hdf5name, fold_generator, encoder_pipeline_factory, **kwargs)

    def build_pretrain_model(self, data_dict, hyper_params):
        """
        pretrain-method specific;
        constucts an SCE net;
        works with any network structure of the pipeline
        :param data_dict:
        :param hyper_params:
        :return:
        """
        from theano import tensor
        from blocks.model import Model

        # Note: this has to match the sources defined in the dataset
        indices = [tensor.ivector('{}_indices'.format(i)) for i in range(3)]

        pipeline = self.encoder_pipeline_factory.build_pipeline(
            input_shape=data_dict.get_value().shape, params=hyper_params)

        # compute feature represenation
        rep = [pipeline.apply(data_dict[indices[i]]) for i in range(3)]
        # for r in rep: print r.type

        # flatten representations
        rep = [r.flatten(ndim=2) for r in rep]
        # for r in rep: print r.type

        # compute similarities
        rval = []
        for i in range(1, 3):
            r = (rep[0] * rep[i]).sum(axis=1) # element-wise multiplication and row sum
            r = tensor.reshape(r, (r.shape[0], 1))
            rval.append(r)
        rval = tensor.concatenate(rval, axis=1)
        # print rval.type
        
        # optional softmax layer (normalization to sum = 1)
        if 'apply_softmax' in hyper_params and hyper_params['apply_softmax']:  # default=False
            from blocks.bricks import Softmax
            rval = Softmax().apply(rval)
        
        # optional argmax (int output instead of scores
        if 'return_probs' in hyper_params and hyper_params['return_probs'] is False:  # default=True
            rval = rval.argmax(axis=1)

        return Model(rval)

    @staticmethod
    def pretrain(model, hyper_params, full_hdf5, full_meta, train_selectors, valid_selectors=None):
        """
        generic training method for siamese networks;
        works with any network structure
        :return:
        """
        from theano import tensor        
        from blocks.bricks.cost import MisclassificationRate
        from deepthought.datasets.triplet import TripletsIndexDataset
        from deepthought.bricks.cost import HingeLoss

        train_data = TripletsIndexDataset(full_hdf5, full_meta,
                                        train_selectors,
                                        targets_source=hyper_params['pretrain_target_source'],
                                        group_attribute=hyper_params['group_attribute'])

        if valid_selectors is not None:
            if hyper_params['use_ext_dataset_for_validation']:
                ext_selectors = train_selectors
            else:
                ext_selectors = None

            valid_data = TripletsIndexDataset(full_hdf5, full_meta,
                                            valid_selectors,
                                            ext_selectors=ext_selectors,
                                            targets_source=hyper_params['pretrain_target_source'],
                                            group_attribute=hyper_params['group_attribute'])
        else:
            valid_data = None

        # Note: this has to match the sources defined in the dataset
        #y = tensor.lvector('targets')
        y = tensor.lmatrix('targets')
        
        # Note: this requires a one-hot encoding of the targets
        probs = model.outputs[0]
        cost = HingeLoss().apply(y, probs)
        # Note: this requires just the class labels, not in a one-hot encoding
        error_rate = MisclassificationRate().apply(y.argmax(axis=1), probs) 
        error_rate.name = 'error_rate'

        return GenericNNEncoderExperiment.run_pretrain(model, hyper_params, cost, 
                                                       train_data, valid_data, 
                                                       [error_rate])
