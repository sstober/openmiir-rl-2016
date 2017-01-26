from blocks.extensions import SimpleExtension
import numpy as np


class BestParams(SimpleExtension):
    def __init__(self, **kwargs):
        kwargs.setdefault("after_training", False)
        super(BestParams, self).__init__(**kwargs)
        
    def do(self, callback_name, *args):
        self.values = self.main_loop.model.get_parameter_values()
        

class SaveParameters(SimpleExtension):
    def __init__(self, parameters, prefixes, **kwargs):
        kwargs.setdefault("after_epoch", True)
        super(SaveParameters, self).__init__(**kwargs)
        self.step = 0
        self.parameters = parameters
        self.prefixes = prefixes

    def do(self, callback_name, *args):
        for i in xrange(len(self.parameters)):
            filename = "%s_%d.npy" % (self.prefixes[i], self.step)
            np.save(filename, self.parameters[i].get_value())
        self.step += 1
