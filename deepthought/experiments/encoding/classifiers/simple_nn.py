from base import ClassifierFactory

class SimpleNNClassifierFactory(ClassifierFactory):

    def __init__(self, pipeline_factory=None):
        self.pipeline_factory = pipeline_factory
        super(SimpleNNClassifierFactory, self).__init__()

    def train(self, X, Y, idx_folds, hyper_params, model_prefix, verbose=False):

        import os
        from collections import OrderedDict
        from fuel.datasets import IndexableDataset
        from blocks.model import Model
        from blocks.bricks import Linear, Softmax
        from blocks.bricks.conv import MaxPooling
        from blocks.initialization import Uniform
        from deepthought.bricks.cost import HingeLoss
        import numpy as np
        import theano
        from theano import tensor

        assert model_prefix is not None

        fold_weights_filename = '{}_weights.npy'.format(model_prefix)

        # convert Y to one-hot encoding
        n_classes = len(set(Y))
        Y = np.eye(n_classes, dtype=int)[Y]

        features = tensor.matrix('features', dtype=theano.config.floatX)
        targets = tensor.lmatrix('targets')

        input_ = features

        dim = X.shape[-1]
        
        # optional additional layers
        if self.pipeline_factory is not None:
            # need to re-shape flattened input to restore bc01 format
            input_shape = (input_.shape[0],) + hyper_params['classifier_input_shape']  # tuple, uses actual batch size
            input_ = input_.reshape(input_shape)

            pipeline = self.pipeline_factory.build_pipeline(input_shape, hyper_params)
            input_ = pipeline.apply(input_)                        
            input_ = input_.flatten(ndim=2)
            
            # this is very hacky, but there seems to be no elegant way to obtain a value for dim
            dummy_fn = theano.function(inputs=[features], outputs=input_)
            dummy_out = dummy_fn(X[:1])
            dim = dummy_out.shape[-1]
            
            
        if hyper_params['classifier_pool_width'] > 1:
            # FIXME: this is probably broken!
            
    #        c = hyper_params['num_components']
    #        input_ = input_.reshape((input_.shape[0], c, input_.shape[-1] // c, 1))  # restore bc01
            # need to re-shape flattened input to restore bc01 format
            input_shape = hyper_params['classifier_pool_input_shape']  # tuple
            input_ = input_.reshape(input_shape)

            pool = MaxPooling(name='pool',
                              input_dim=input_shape[1:],  # (c, X.shape[-1] // c, 1),
                              pooling_size=(hyper_params['classifier_pool_width'], 1),
                              step=(hyper_params['classifier_pool_stride'], 1))
            input_ = pool.apply(input_)
            input_ = input_.reshape((input_.shape[0], tensor.prod(input_.shape[1:])))

            dim = np.prod(pool.get_dim('output'))


        linear = Linear(name='linear',
                        input_dim=dim,
                        output_dim=n_classes,
                        weights_init=Uniform(mean=0, std=0.01),
                        use_bias=False)
        linear.initialize()

        softmax = Softmax('softmax')

        probs = softmax.apply(linear.apply(input_))
        prediction = tensor.argmax(probs, axis=1)

        model = Model(probs)  # classifier with raw probability outputs
        predict = theano.function([features], prediction)  # ready-to-use predict function

        if os.path.isfile(fold_weights_filename):
            # load filter weights from existing file
            fold_weights = np.load(fold_weights_filename)
            print 'loaded filter weights from', fold_weights_filename
        else:
            # train model

            from blocks.bricks.cost import MisclassificationRate
            from blocks.filter import VariableFilter
            from blocks.graph import ComputationGraph
            from blocks.roles import WEIGHT
            from blocks.bricks import Softmax
            from blocks.model import Model
            from blocks.algorithms import GradientDescent, Adam
            from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
            from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
            from blocks.extensions.predicates import OnLogRecord
            from fuel.streams import DataStream
            from fuel.schemes import SequentialScheme, ShuffledScheme
            from blocks.monitoring import aggregation
            from blocks.main_loop import MainLoop
            from blocks.extensions.training import TrackTheBest
            from deepthought.extensions.parameters import BestParams
            # from deepthought.datasets.selection import DatasetMetaDB

            init_param_values = model.get_parameter_values()

            cost = HingeLoss().apply(targets, probs)
            # Note: this requires just the class labels, not in a one-hot encoding
            error_rate = MisclassificationRate().apply(targets.argmax(axis=1), probs)
            error_rate.name = 'error_rate'

            cg = ComputationGraph([cost])

            # L1 regularization
            if hyper_params['classifier_l1wdecay'] > 0:
                weights = VariableFilter(roles=[WEIGHT])(cg.variables)
                cost = cost + hyper_params['classifier_l1wdecay'] * sum([abs(W).sum() for W in weights])

            cost.name = 'cost'

            # iterate over trial folds
            fold_weights = []
            fold_errors = []

            # for ifi, ifold in fold_generator.get_inner_cv_folds(outer_fold):
            #
            #     train_selectors = fold_generator.get_fold_selectors(outer_fold=outer_fold, inner_fold=ifold['train'])
            #     valid_selectors = fold_generator.get_fold_selectors(outer_fold=outer_fold, inner_fold=ifold['valid'])
            #
            #     metadb = DatasetMetaDB(meta, train_selectors.keys())
            #
            #     # get selected trial IDs
            #     train_idx = metadb.select(train_selectors)
            #     valid_idx = metadb.select(valid_selectors)

            for train_idx, valid_idx in idx_folds:

                # print train_idx
                # print valid_idx

                trainset = IndexableDataset(indexables=OrderedDict(
                    [('features', X[train_idx]), ('targets', Y[train_idx])]))

                validset = IndexableDataset(indexables=OrderedDict(
                    [('features', X[valid_idx]), ('targets', Y[valid_idx])]))

                model.set_parameter_values(init_param_values)

                best_params = BestParams()
                best_params.add_condition(['after_epoch'],
                                          predicate=OnLogRecord('error_rate_valid_best_so_far'))

                algorithm = GradientDescent(cost=cost, parameters=cg.parameters, step_rule=Adam())

                extensions = [Timing(),
                              FinishAfter(after_n_epochs=hyper_params['classifier_max_epochs']),
                              DataStreamMonitoring(
                                  [cost, error_rate],
                                  DataStream.default_stream(
                                      validset,
                                      iteration_scheme=SequentialScheme(
                                          validset.num_examples, hyper_params['classifier_batch_size'])),
                                  suffix="valid"),
                              TrainingDataMonitoring(
                                  [cost, error_rate,
                                   aggregation.mean(algorithm.total_gradient_norm)],
                                  suffix="train",
                                  after_epoch=True),
                              TrackTheBest('error_rate_valid'),
                              best_params  # after TrackTheBest!
                              ]

                if verbose:
                    extensions.append(Printing())  # optional
                    extensions.append(ProgressBar())

                main_loop = MainLoop(
                    algorithm,
                    DataStream.default_stream(
                        trainset,
                        iteration_scheme=ShuffledScheme(trainset.num_examples, hyper_params['classifier_batch_size'])),
                    model=model,
                    extensions=extensions)

                main_loop.run()

                fold_weights.append(best_params.values['/linear.W'])
                fold_errors.append(main_loop.status['best_error_rate_valid'])
                # break # FIXME

            fold_errors = np.asarray(fold_errors).squeeze()
            print 'simple NN fold classification errors:', fold_errors

            fold_weights = np.asarray(fold_weights)

            # store filter weights for later analysis
            np.save(fold_weights_filename, fold_weights)

        weights = fold_weights.mean(axis=0)

        linear.parameters[0].set_value(weights)

        return model, predict
