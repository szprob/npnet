from npnet.tensor import Parameter
import numpy as np
from npnet.nn.module import Module

class Linear(Module):
    def __init__(self, input_dim:int, output_dim:int) -> None:
        super().__init__()
        self.module_name = 'Linear'
        self.w = Parameter(shape = (input_dim,output_dim))
        self.b=  Parameter(shape = (output_dim))
        
    def forward(self, input: Tensor)->Tensor:
        tmp = input @ self.w
        out = tmp + self.b
        return out