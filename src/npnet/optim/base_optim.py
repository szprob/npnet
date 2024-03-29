from typing import List
from abc import ABC, abstractmethod
import numpy as np
from gradflow.grad_engine import Variable


class BaseOptimizer(ABC):
    """Torch-like optimizer."""

    def __init__(self, parameters: List[Variable], lr:float=0.0001) -> None:
        super().__init__()
        self._parameters = parameters
        self._lr = lr

    def zero_grad(self):
        for param in self._parameters:
            if param.requires_grad:
                


            if parameter.requires_grad == False:
                continue

            if isinstance(parameter.grad, np.ndarray):
                parameter.grad = np.zeros_like(parameter.grad)
            else:
                parameter.grad = np.array([0], dtype=np.float)

    @abstractmethod
    def step(self):
        raise NotImplementedError


class NaiveSGD(BaseOptimizer):
    def __init__(self, parameters: List[Variable], lr=0.001) -> None:
        super().__init__(parameters=parameters, lr=lr)

    def step(self):
        for parameter in self._parameters:
            clipped_grad = np.clip(parameter.grad, -1000, 1000)
            delta = -self._lr * clipped_grad
            delta = np.transpose(delta)
            # print(delta.shape)
            # print(parameter.data.shape)
            # print()
            parameter.data = parameter.data + delta