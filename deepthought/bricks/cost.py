from blocks.bricks.cost import Cost
from blocks.bricks.base import application
from theano import tensor as T


class HingeLoss(Cost):

    @application(outputs=["cost"])
    def apply(self, y, y_hat):
        cost = T.sum(T.maximum(1 - y * y_hat, 0) ** 2., axis=1).mean()
        return cost
