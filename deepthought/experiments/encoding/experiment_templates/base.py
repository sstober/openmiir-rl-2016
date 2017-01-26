import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report  # precision_recall_fscore_support
from deepthought.util.crossvalidation_util import ClassificationResult
import logging

log = logging.getLogger('deepthought.experiments')


class NestedCVFoldGenerator(object):
    def __init__(self, base_selectors=None):
        self.base_selectors = dict() if base_selectors is None else base_selectors.copy()

    def get_outer_cv_folds(self):
        pass

    def get_inner_cv_folds(self, outer_fold):
        pass

    def get_fold_selectors(self, outer_fold=None, inner_fold=None, base_selectors=None):
        pass


class NestedCVExperimentTemplate(object):
    _default_params = dict(
        pretrain_target_source = 'targets',
        classification_target_source = 'targets',
    )

    def __init__(self,
                 job_id,
                 hdf5name,
                 fold_generator,
                 base_output_path='results',
                 hyper_params=None,  # override params
                 default_params=None,  # optional default values for unset hyper params
                 base_selectors=None,
                 ):
        self.job_id = job_id
        self.output_path = os.path.join(base_output_path, '{}'.format(job_id))
        self.hdf5name = hdf5name
        self.fold_generator = fold_generator
        # self.base_output_path = base_output_path
        self.base_selectors = dict() if base_selectors is None else base_selectors.copy()

        self.hyper_params = self._default_params.copy() if default_params is None else default_params.copy()

        # merge params (override defaults)
        if hyper_params is not None:
            for key, value in hyper_params.items():
                self.hyper_params[key] = value

        # # unwrap (for spearmint) and print the hyper parameter values
        # for key, value in self.hyper_params.items():
        #     if isinstance(value, list):
        #         self.hyper_params[key] = value[0]
        #     print('{} = {}'.format(key, self.hyper_params[key]))

        super(NestedCVExperimentTemplate, self).__init__()

    def initialize(self):
        """
        pre-outer loop hook
        :return:
        """
        pass

    def pretrain_encoder(self, outer_fold_index, outer_fold):
        pass

    def get_encoded_dataset(self, encoder_fn, selectors):
        """
        This version is intended for use with a data dict / indices.
        :return:
        """
        from deepthought.datasets.selection import DatasetMetaDB
        from deepthought.util.function_util import process_dataset
        import theano

        # build lookup structure
        metadb = DatasetMetaDB(self.full_meta, selectors.keys())

        # get selected trial IDs
        selected_trial_ids = metadb.select(selectors)

        X, Y = process_dataset(self.full_hdf5, encoder_fn,
                               indices=selected_trial_ids,
                               input_sources=['indices'],
                               target_source=self.hyper_params['classification_target_source'])
        meta = [self.full_meta[i] for i in selected_trial_ids]

        # flatten X (2d) and Y (1d)
        X = np.asarray(X, dtype=theano.config.floatX)
        X = X.reshape(X.shape[0], np.prod(X.shape[1:]))
        Y = Y.argmax(axis=1)

        return X, Y, meta

    def run(self, classifiers=(), verbose=False, debug=False):
        print 'running job #{}'.format(self.job_id)

        import deepthought.util.fs_util as fs_util
        fs_util.ensure_dir_exists(self.output_path)
        print 'output path: ', self.output_path

        # prepare result objects
        results = {k: ClassificationResult(k) for (k, _) in classifiers}

        # load full dataset with all sources only once!
        from deepthought.datasets.hdf5 import get_dataset
        self.full_hdf5, self.full_meta = get_dataset(self.hdf5name, selectors=self.base_selectors, sources=None)

        self.initialize()

        # main loop ###

        # outer cross-validation
        outer_folds = self.fold_generator.get_outer_cv_folds()
        for ofi, ofold in enumerate(outer_folds):
            print 'processing outer fold', ofold

            # phase I : pre-train features ###
            encoder_fn = self.pretrain_encoder(ofi, ofold)  # FIXME: add params

            # phase II : classify ###

            train_selectors = self.fold_generator.get_fold_selectors(outer_fold=ofold['train'])
            X_train, Y_train, meta_train = self.get_encoded_dataset(encoder_fn, train_selectors)

            test_selectors = self.fold_generator.get_fold_selectors(outer_fold=ofold['valid'])
            X_test, Y_test, _ = self.get_encoded_dataset(encoder_fn, test_selectors)

            for (classifier_name, classifier_factory) in classifiers:
                result = results[classifier_name]

                model_prefix = os.path.join(self.output_path, '{}_fold_{}'.format(classifier_name, ofi))

                # generate index folds
                idx_folds = []
                from deepthought.datasets.selection import DatasetMetaDB
                for ifold in self.fold_generator.get_inner_cv_folds(ofold):
                    train_selectors = self.fold_generator.get_fold_selectors(outer_fold=ofold['train'],
                                                                             inner_fold=ifold['train'])
                    metadb = DatasetMetaDB(meta_train, train_selectors.keys())

                    
                    if 'valid' in ifold.keys():
                        valid_selectors = self.fold_generator.get_fold_selectors(outer_fold=ofold['train'],
                                                                                 inner_fold=ifold['valid'])
                    else:
                        valid_selectors = None
                    
                    if debug:
                        print 'train_selectors:', train_selectors                                        
                        print 'valid_selectors:', valid_selectors

                    # get selected trial IDs
                    train_idx = metadb.select(train_selectors)
                    
                    if valid_selectors is not None:
                        valid_idx = metadb.select(valid_selectors)
                    else:
                        valid_idx = []

                    idx_folds.append((train_idx, valid_idx))

                if debug:
                    print idx_folds  # print the generated folds before running the classifier
                
                # train classifier
                classifier, predict_fn = classifier_factory.train(X_train, Y_train, idx_folds, self.hyper_params, model_prefix)

                # test classifier
                train_Y_pred = predict_fn(X_train)
                test_Y_pred = predict_fn(X_test)

                # append to result
                result.append_train(Y_train, train_Y_pred)
                result.append_test(Y_test, test_Y_pred)
                # result.fold_scores.append(classifier.score(X_test, Y_test))
                result.fold_scores.append(np.mean(Y_test == test_Y_pred))

                if verbose:
                    print '{} results for fold {}'.format(classifier_name, ofold)
                    print classification_report(Y_test, test_Y_pred)
                    print confusion_matrix(Y_test, test_Y_pred)
                    print 'overall test accuracy so far:', 1 - result.test_error()

        print 'all folds completed'

        for (classifier_name, _) in classifiers:
            result = results[classifier_name]
            fs_util.save(os.path.join(self.output_path, '{}_result.pklz'.format(classifier_name)), result)  # result

            print
            print 'SUMMARY for classifier', classifier_name
            print
            print 'fold scores: ', np.asarray(result.fold_scores)
            print
            print classification_report(result.test_Y_real, result.test_Y_pred)
            print confusion_matrix(result.test_Y_real, result.test_Y_pred)
            print
            print 'train accuracy:', 1 - result.train_error()
            print 'test accuracy :', 1 - result.test_error()

        return [results[classifier[0]].test_error() for classifier in classifiers]  # error for each classifier


class GenericNNEncoderExperiment(NestedCVExperimentTemplate):

    def __init__(self,
                 job_id,
                 hdf5name,
                 fold_generator,
                 encoder_pipeline_factory,
                 **kwargs):

        self.encoder_pipeline_factory = encoder_pipeline_factory
        super(GenericNNEncoderExperiment, self).__init__(job_id, hdf5name, fold_generator, **kwargs)

    def build_pretrain_model(self, data_dict, hyper_params):
        raise NotImplementedError()

    def pretrain(self, model, hyper_params, full_hdf5, full_meta, train_selectors, valid_selectors=None):
        raise NotImplementedError()

    def initialize(self):
        from deepthought.bricks.data_dict import generate_data_dict
        data_dict = generate_data_dict(self.full_hdf5, 'features', verbose=True)

        # this model will be used during pre-training
        self.pretrain_model = self.build_pretrain_model(data_dict, self.hyper_params)

        # this model will be used after pre-training to encode the input (no training)
        self.encoder_model = self.build_encoder_model(data_dict, self.hyper_params)

        self.init_param_values = self.pretrain_model.get_parameter_values()
                
        # print encoder model parameters
        print '--- Encoder model parameters ---'
        total_params = 0
        for k,v in self.encoder_model.get_parameter_values().items():
            nparams = np.prod(v.shape)
            total_params += nparams
            print k, v.shape, nparams
        print 'total number of params:', total_params

    def build_encoder_model(self, data_dict, hyper_params):
        from theano import tensor
        from blocks.model import Model

        indices = tensor.ivector('indices')
        pipeline = self.encoder_pipeline_factory.build_pipeline(
            input_shape=data_dict.get_value().shape, params=hyper_params)
        output = pipeline.apply(data_dict[indices])

        return Model(output)

    def pretrain_encoder(self, outer_fold_index, outer_fold):
        """
        generic template that works with any model structure
        :param outer_fold_index:
        :param outer_fold:
        :return:
        """
        import deepthought.util.fs_util as fs_util
        from deepthought.util.function_util import get_function

        fold_params_filename = os.path.join(self.output_path, 'fold_params_{}.pklz'.format(outer_fold_index))

        inner_folds = self.fold_generator.get_inner_cv_folds(outer_fold)

        if os.path.isfile(fold_params_filename):
            # load trained network parameters from existing file
            fold_param_values = fs_util.load(fold_params_filename)
            print 'loaded trained fold network parameters from', fold_params_filename
            #assert len(fold_param_values) == len(inner_folds)
        else:
            # compute trial fold models
            fold_param_values = []
            fold_errors = []
            for ifi, ifold in enumerate(inner_folds):
                log.info('processing fold {}.{}: {}'.format(outer_fold_index, ifi, ifold))

                train_selectors = self.fold_generator.get_fold_selectors(
                    outer_fold=outer_fold['train'], inner_fold=ifold['train'], base_selectors=self.base_selectors)
                
                if 'valid' in ifold.keys():
                    valid_selectors = self.fold_generator.get_fold_selectors(
                        outer_fold=outer_fold['train'], inner_fold=ifold['valid'], base_selectors=self.base_selectors)
                else:
                    valid_selectors = None

                self.pretrain_model.set_parameter_values(self.init_param_values)  # reset weights
                trained_model_param_values, best_error_valid = self.pretrain(
                    self.pretrain_model, self.hyper_params,
                    self.full_hdf5, self.full_meta,
                    train_selectors, valid_selectors)

                fold_param_values.append(trained_model_param_values)
                fold_errors.append(best_error_valid)
                
                if 'only_1_inner_fold' in self.hyper_params and self.hyper_params['only_1_inner_fold']:
                    print 'Stop after 1 inner fold requested (only_1_inner_fold=True).'
                    break

            fold_errors = np.asarray(fold_errors).squeeze()
            print 'fold errors:', fold_errors

            # store trained network parameters for later analysis
            fs_util.save(fold_params_filename, fold_param_values)
            print 'parameters saved to', fold_params_filename

        # build encoder
        encoder = self.encoder_pipeline_factory.set_pipeline_parameters(self.encoder_model, fold_param_values)

        # transform dataset (re-using data_dict and working with indices as input)
        encoder_fn = get_function(encoder, allow_input_downcast=True)

        return encoder_fn
    
    @staticmethod
    def run_pretrain(model, hyper_params, cost, train_data, valid_data=None, extra_costs=None):
        """
        generic training method for neural networks;
        works with any network structure
        :return:
        """
        from fuel.streams import DataStream
        from fuel.schemes import SequentialScheme, ShuffledScheme
        from blocks.filter import VariableFilter
        from blocks.graph import ComputationGraph
        from blocks.roles import WEIGHT
        from blocks.algorithms import GradientDescent, Adam, RMSProp, Scale
        from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
        from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
        from blocks.extensions.predicates import OnLogRecord
        from blocks.monitoring import aggregation
        from blocks.main_loop import MainLoop
        from blocks.extensions.training import TrackTheBest
        from deepthought.extensions.parameters import BestParams    

        if extra_costs is None:
            extra_costs = []
        
        cg = ComputationGraph([cost])

        # TODO: more hyper-params for regularization
        # L1 regularization
        if hyper_params['l1wdecay'] > 0:
            weights = VariableFilter(roles=[WEIGHT])(cg.variables)
            cost = cost + hyper_params['l1wdecay'] * sum([abs(W).sum() for W in weights])

        cost.name = 'cost'

        # set up step_rule
        if hyper_params['step_rule'] == 'Adam':
            step_rule = Adam(learning_rate=hyper_params['learning_rate'])
        elif hyper_params['step_rule'] == 'RMSProp':
            step_rule = RMSProp(learning_rate=hyper_params['learning_rate']) #, decay_rate=0.9, max_scaling=1e5)
        else:
            step_rule = Scale(learning_rate=hyper_params['learning_rate'])
        
        algorithm = GradientDescent(cost=cost, parameters=cg.parameters, step_rule=step_rule)

        if 'blocks_print_variable_names' in hyper_params and hyper_params['blocks_print_variable_names']:
            print 'cg.variables:', cg.variables

        train_monitoring_vars = [cost] + extra_costs + [aggregation.mean(algorithm.total_gradient_norm)]
        for var_name in hyper_params['blocks_extensions_train_monitoring_channels']:
            for v in cg.variables:
                if v.name == var_name:
                    print 'Monitoring variable:', v
                    train_monitoring_vars.append(v)

        # default extensions
        extensions = [Timing(),
                      FinishAfter(after_n_epochs=hyper_params['max_epochs']),
                      TrainingDataMonitoring(
                          train_monitoring_vars,
                          suffix="train",
                          after_epoch=True)
                      ]

        # additional stuff if validation set is used
        if valid_data is not None:
            valid_monitoring_vars = [cost] + extra_costs
            for var_name in hyper_params['blocks_extensions_valid_monitoring_channels']:
                for v in cg.variables:
                    if v.name == var_name:
                        print 'Monitoring variable:', v
                        valid_monitoring_vars.append(v)

            extensions.append(
                DataStreamMonitoring(
                    valid_monitoring_vars,
                    DataStream.default_stream(
                        valid_data,
                        iteration_scheme=SequentialScheme(
                            valid_data.num_examples, hyper_params['batch_size'])),
                    suffix="valid"))

            best_channel = 'cost_valid'
            print '#train:', train_data.num_examples, '#valid:', valid_data.num_examples
        else:
            best_channel = 'cost_train'
            print '#train:', train_data.num_examples

        # tracking of the best
        best_params = BestParams()
        best_params.add_condition(['after_epoch'],
                                  predicate=OnLogRecord(best_channel + '_best_so_far'))
        extensions.append(TrackTheBest(best_channel))
        extensions.append(best_params)  # after TrackTheBest!

        # printing and plotting
        if hyper_params['blocks_extensions_printing'] is True:
            extensions.append(Printing())  # optional
        if hyper_params['blocks_extensions_progressbar'] is True:
            extensions.append(ProgressBar())

        if hyper_params['blocks_extensions_bokeh'] is True:
            try:
                from blocks_extras.extensions.plot import Plot
                bokeh_available = True
            except:
                bokeh_available = False
            print 'bokeh available: ', bokeh_available

            if bokeh_available:
                extensions.append(Plot(
                    hyper_params['blocks_extensions_bokeh_plot_title'],
                    channels=hyper_params['blocks_extensions_bokeh_channels'],
                ))

        main_loop = MainLoop(
            algorithm,
            DataStream.default_stream(
                train_data,
                iteration_scheme=ShuffledScheme(
                    train_data.num_examples, hyper_params['batch_size'])),
            model=model,
            extensions=extensions)

        main_loop.run()

        return best_params.values, main_loop.status['best_' + best_channel]
