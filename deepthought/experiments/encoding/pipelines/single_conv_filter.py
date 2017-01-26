import numpy as np
from base import EncoderPipelineFactory


class SingleConvFilterPipelineFactory(EncoderPipelineFactory):

    def set_pipeline_parameters(self, encoder_model, fold_param_values):
        """
        model-structure specific;
        determines how an encoder model is derived from the fold model parameter values
        :param encoder_model:
        :param fold_param_values:
        :return:
        """
        fold_weights = [param_values['/pipeline/conv.W'] for param_values in fold_param_values]
        fold_weights = np.asarray(fold_weights)

        # FIXME: this only works for 64-channel layout
        # There might be in issue with flipped filter weights!
        # Using sign of channel T7 (left ear) for consistent polarity!
        fold_polarity = np.sign(fold_weights[:, 0, 14, 0, 0])
        fold_weights = np.asarray([p * w for p, w in zip(fold_polarity, fold_weights)])

        # build encoder with avg weights
        avg_values = dict()
        avg_values['/pipeline/conv.W'] = fold_weights.mean(axis=0)

        encoder_model.set_parameter_values(avg_values)

        return encoder_model

    def build_pipeline(self, input_shape, params):
        from blocks.bricks import Tanh, Sequence
        from blocks.bricks.conv import Convolutional, MaxPooling
        from blocks.initialization import Uniform
        
        _, num_channels, input_len, num_freqs = input_shape  # bc01
        
        # Note: this layer is linear
        conv = Convolutional(name='conv',
                             filter_size=(params['filter_width_time'], params['filter_width_freq']),
                             num_filters=params['num_components'],  # out
                             num_channels=num_channels,  # in
                             image_size=(input_len, num_freqs),
                             weights_init=Uniform(mean=0, std=0.01),
                             use_bias=params['use_bias'])

        tanh = Tanh()

        # optional pooling
        if params['pool_width_time'] > 1 or params['pool_width_freq'] > 1:
            pool = MaxPooling((params['pool_width_time'], params['pool_width_freq']), 
                              step=(params['pool_stride_time'], params['pool_stride_freq']))
            pipeline = Sequence([conv.apply, tanh.apply, pool.apply], name='pipeline')
        else:
            pipeline = Sequence([conv.apply, tanh.apply], name='pipeline')
        pipeline.initialize()

        return pipeline
