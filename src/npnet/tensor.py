from __future__ import annotations
from typing import List, Tuple, Dict, Union,Optional

import numpy as np

t1=Tensor(2.7)
t2=Tensor(5.3)
t3=t1@t2
t3
t3.backward()
t3.grad
t2.grad
t1.grad

class Tensor:
    """Basic torch-like tensor class.

    Attributes:
        data (Union[Tensor,np.ndarray]): 
            Data storing.
        parents (Optional[Tuple[Tensor]], optional):    
            Parents tensor. 
            Defaults to None.
        requires_grad (Optional[bool], optional): 
            Defaults to False.
    
    """    
    def __init__(
        self, 
        data: Union[Tensor,np.ndarray,List,float,int], 
        parents: Optional[Tuple[Tensor]] = None, 
        requires_grad:Optional[bool] = True,
    ) -> None:

        if isinstance(data, self.__class__):
            data = data.data
        elif isinstance(data, np.ndarray):
            data = data
        elif isinstance(data,List) or isinstance(data,int) or  isinstance(data,float)  :
            data=np.array([data])
        else:
            raise TypeError(f"The data type provided is not supported: {type(data)}")


        self.data = data
        self.grad: Union[int, None] = .0 if requires_grad else None
        self.parents = parents or ()
        self._requires_grad = requires_grad
        self._back_grad_fn = lambda: None

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad: bool):
        self.grad = .0 if requires_grad else None
        self._requires_grad = requires_grad

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, requires_grad={self.requires_grad})"

    def __add__(self, other: Tensor) -> Tensor:
        if not isinstance(other, Tensor):
            raise TypeError("The second operator must be a Tensor type")

        result = self.data + other.data
        child = Tensor(result, parents=(self, other))

        if any((parent.requires_grad for parent in child.parents)):
            child.requires_grad = True

            def _back_grad_fn():
                self.grad += child.grad
                other.grad += child.grad

            tensor._back_grad_fn = _back_grad_fn
        return tensor

    def backward(self, grad: Tensor | np.ndarray = None) -> None:
        if grad is None:
            grad = np.array([1.])
        
        if not isinstance(grad, (Tensor, np.ndarray)):
            raise ValueError("The backward gradient must be a numpy array")
        
        if isinstance(grad, Tensor):
            grad = grad.grad
        
        self.grad = grad
        _queue = [self]

        # TODO check if topological sort is needed
        while len(_queue):
            variable = _queue.pop(0)
            variable._back_grad_fn()
            _queue.extend(list(variable.parents))


    def __sub__(self, other: Tensor) -> Tensor:
        if not isinstance(other, Tensor):
            raise TypeError("The second operator must be a Tensor type")

        result = self.data - other.data
        child = Tensor(result, parents=(self, other))

        if any((parent.requires_grad for parent in child.parents)):
            child.requires_grad = True

            def _back_grad_fn():
                self.grad += child.grad
                other.grad -= child.grad

            child._back_grad_fn = _back_grad_fn
        return child


    def __matmul__(self, other: Tensor) -> Tensor:
        if not isinstance(other, Tensor):
            raise TypeError("The second operator must be a Tensor type")

        result = np.matmul(self.data, other.data)

        if not isinstance(result, np.ndarray):
            result = np.array([result])

        child = Tensor(result, parents=(self, other))

        if any((parent.requires_grad for parent in child.parents)):
            child.requires_grad = True

            def _back_grad_fn():
                self.grad += (other.data * child.grad)
                other.grad += (self.data * child.grad)

            child._back_grad_fn = _back_grad_fn
        return child

    def _accumulate_gradient(variable: Variable, grad: np.ndarray):
        if variable.requires_grad:
            variable.grad += grad
    def __pow__(self, exponent: int) -> Variable:
        if not isinstance(exponent, int):
            raise TypeError("For power operation the exponent must be a scalar integer value")


    def T(self) -> Variable:
        variable = Variable(np.transpose(self.data), parents=(self,), requires_grad=self.requires_grad)
        
        def _back_grad_fn():
            self.grad += variable.grad

        variable._back_grad_fn = _back_grad_fn
        return variable












if __name__ == "__main__":
    pass