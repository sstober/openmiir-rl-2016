import logging

from deepthought.experiments.encoding.experiment_templates.base import GenericNNEncoderExperiment, NestedCVExperimentTemplate

log = logging.getLogger('deepthought.experiments')


class TripletNetworkExperiment(GenericNNEncoderExperiment):

    _default_params = NestedCVExperimentTemplate._default_params.copy()
    _default_params['group_attribute'] = 'subject'

    def __init__(self,
                 job_id,
                 hdf5name,
                 fold_generator,
                 encoder_pipeline_factory,
                 **kwargs):

        super(TripletNetworkExperiment, self).__init__(
            job_id, hdf5name, fold_generator, encoder_pipeline_factory, **kwargs)

    def initialize(self):
        from deepthought.bricks.data_dict import generate_data_dict
        data_dict = generate_data_dict(self.full_hdf5, 'features', verbose=True)

        # this model will be used during pre-training
        self.pretrain_model = self.build_pretrain_model(data_dict, self.hyper_params)

        # this model will be used after pre-training to encode the input (no training)
        self.encoder_model = self.build_encoder_model(data_dict, self.hyper_params)

        self.init_param_values = self.pretrain_model.get_parameter_values()

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
        from blocks.bricks import Softmax
        
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

        # compute L2-norm - Triplet Network distance
        rval = []
        for i in range(1, 3):
            r = tensor.sqrt(((rep[0] - rep[i])**2).sum(axis=1))
            r = tensor.reshape(r, (r.shape[0], 1))
            rval.append(r)
        rval = tensor.concatenate(rval, axis=1)
        # print rval.type
        
        rval = Softmax().apply(rval)  # part of Triplet Network

        return Model(rval)

    @staticmethod
    def pretrain(model, hyper_params, full_hdf5, full_meta, train_selectors, valid_selectors=None):
        """
        generic training method for siamese networks;
        works with any network structure
        :return:
        """
        from theano import tensor
        from deepthought.datasets.triplet import TripletsIndexDataset
        from blocks.bricks.cost import MisclassificationRate

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
        # cost = HingeLoss().apply(y, probs)  # cost for SCE
        cost = (probs[:,0]**2).sum()  # triplet network cost = const * d+**2
        
        # Note: this requires just the class labels, not in a one-hot encoding
        # error_rate = MisclassificationRate().apply(y.argmax(axis=1), probs)  # SCE version
        error_rate = 1. - MisclassificationRate().apply(y.argmax(axis=1), probs)  # flipped for triplet net
        error_rate.name = 'error_rate'

        return GenericNNEncoderExperiment.run_pretrain(model, hyper_params, cost, 
                                                       train_data, valid_data, 
                                                       [error_rate])
