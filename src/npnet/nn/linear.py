from npnet.tensor import Tensor
import numpy as np
from npnet.nn.module import Module

class Linear(Module):
    def __init__(self, in_size:int, out_size:int) -> None:
        super().__init__()
        weights_data: np.ndarray = np.random.uniform(size=in_size * out_size).reshape((in_size, out_size))
        self.weights = Variable(weights_data, requires_grad=True)
        self.b = Variable(np.random.uniform(size=out_size), requires_grad=True)

        self.add_parameter(self.weights)
        self.add_parameter(self.b)

    def reset_parameters(self):
        

    def forward(self, input: Tensor)->Tensor:
        tmp = input @ self.weights
        out = tmp + self.b
        return 