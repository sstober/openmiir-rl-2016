import logging

from deepthought.experiments.encoding.experiment_templates.base import GenericNNEncoderExperiment, NestedCVExperimentTemplate

log = logging.getLogger('deepthought.experiments')


class SiameseEncoderExperiment(GenericNNEncoderExperiment):

    _default_params = NestedCVExperimentTemplate._default_params.copy()
    _default_params['group_attribute'] = 'subject'

    def __init__(self,
                 job_id,
                 hdf5name,
                 fold_generator,
                 encoder_pipeline_factory,
                 **kwargs):

        super(SiameseEncoderExperiment, self).__init__(
            job_id, hdf5name, fold_generator, encoder_pipeline_factory, **kwargs)

    def build_pretrain_model(self, data_dict, hyper_params):
        """
        pretrain-method specific;
        constucts a siamese net;
        works with any network structure of the pipeline
        :param data_dict:
        :param hyper_params:
        :return:
        """
        from theano import tensor
        from blocks.model import Model

        # Note: this has to match the sources defined in the dataset
        indices = [tensor.ivector('{}_indices'.format(i)) for i in range(2)]

        pipeline = self.encoder_pipeline_factory.build_pipeline(
            input_shape=data_dict.get_value().shape, params=hyper_params)

        # compute feature represenation
        rep = [pipeline.apply(data_dict[indices[i]]) for i in range(2)]
        # for r in rep: print r.type

        # flatten representations
        rep = [r.flatten(ndim=2) for r in rep]
        # for r in rep: print r.type

        # TODO: choose based on hyper-params
        if hyper_params['siamese_distance'] == 'L1-norm':
            rval = abs(rep[0] - rep[1]).sum(axis=1)
        elif hyper_params['siamese_distance'] == 'euclidean':  # L2-norm
            # compute Euclidean distance
            rval = tensor.sqrt(((rep[0] - rep[1])**2).sum(axis=1))
        elif hyper_params['siamese_distance'] == 'dot-product':
            # alternative: negative dot-product distance
            rval = -(rep[0] * rep[1]).sum(axis=1)  # element-wise multiplication and row sum
        else:
            raise ValueError('Unsupported value for siamese_distance: {}'.format(hyper_params['siamese_distance']))
        rval = tensor.reshape(rval, (rval.shape[0], 1))

        return Model(rval)

    @staticmethod
    def pretrain(model, hyper_params, full_hdf5, full_meta, train_selectors, valid_selectors=None):
        """
        generic training method for siamese networks;
        works with any network structure
        :return:
        """
        from theano import tensor
        from deepthought.datasets.siamese import PairsIndexDataset

        train_pairs = PairsIndexDataset(full_hdf5, full_meta,
                                        train_selectors,
                                        targets_source=hyper_params['pretrain_target_source'],
                                        group_attribute=hyper_params['group_attribute'])

        if valid_selectors is not None:
            if hyper_params['use_ext_dataset_for_validation']:
                ext_selectors = train_selectors
            else:
                ext_selectors = None

            valid_pairs = PairsIndexDataset(full_hdf5, full_meta,
                                            valid_selectors,
                                            ext_selectors=ext_selectors,
                                            targets_source=hyper_params['pretrain_target_source'],
                                            group_attribute=hyper_params['group_attribute'])            
        else:
            valid_pairs = None            

        # Note: this has to match the sources defined in the dataset
        y = tensor.lvector('targets')

        margin = hyper_params['siamese_margin']  # 0  # -0.5 # FIXME: hyper-param
        d = model.outputs[0]
        cost = 0.5 * (y * d + (1 - y) * tensor.maximum(margin - d, 0))
        cost = tensor.mean(cost)
        
        return GenericNNEncoderExperiment.run_pretrain(model, hyper_params, cost, train_pairs, valid_pairs)
