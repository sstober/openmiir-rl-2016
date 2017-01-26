from theano import tensor
from blocks.bricks import Brick
from blocks.bricks.base import application


class SwapAxes(Brick):
    def __init__(self, axis1, axis2, debug=False, **kwargs):
        super(SwapAxes, self).__init__(**kwargs)
        self.axis1 = axis1
        self.axis2 = axis2
        self.debug = debug
        
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        output = tensor.swapaxes(input_, self.axis1, self.axis2)
        if self.debug:
            import theano
            output = theano.printing.Print('output:', attrs=('shape',))(output)
        return output
    