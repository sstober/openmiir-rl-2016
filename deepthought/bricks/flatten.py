from blocks.bricks import Brick
from blocks.bricks.base import application


class Flatten(Brick):
    def __init__(self, ndim, debug=False, **kwargs):
        super(Flatten, self).__init__(**kwargs)
        self.ndim = ndim
        self.debug = debug
        
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        output = input_.flatten(ndim=self.ndim)
        if self.debug:
            import theano
            output = theano.printing.Print('output:', attrs=('shape',))(output)
        return output
    