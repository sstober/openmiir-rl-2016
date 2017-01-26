class EncoderPipelineFactory(object):

    def build_pipeline(self, input_shape, params):
        pass

    def set_pipeline_parameters(self, encoder_model, fold_param_values):
        """
        generic average (works with any structure)
        :param encoder_model:
        :param fold_param_values:
        :return:
        """
        import numpy as np
        avg_param_values = dict()
        for key in fold_param_values[0].keys():
            param_values = [fpv[key] for fpv in fold_param_values]
            param_values = np.asarray(param_values)
            
            avg_param_values[key] = param_values.mean(axis=0)
        
        encoder_model.set_parameter_values(avg_param_values)

        return encoder_model
