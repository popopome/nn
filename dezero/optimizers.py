import numpy as np
from dezero.core import Parameter

class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self
    
    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]

        # 전처리(옵션)
        for f in self.hooks:
            f(params)

        # 매개변수 갱신
        for param in params:
            self.update_one(param)


    def update_one(self, param):
        raise NotImplementedError()
    
    def add_hook(self, f):
        self.hooks.append(f)

class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr
    
    def update_one(self, param):
        assert(isinstance(param, Parameter))
        param.data -= self.lr * param.grad.data


class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        assert(isinstance(param, Parameter))
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v
