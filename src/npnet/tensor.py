from __future__ import annotations
from typing import List, Tuple, Union,Optional

import numpy as np


def _add(tensor:Tensor, other: Tensor) -> Tensor:
    if not isinstance(other, Tensor) or not isinstance(tensor, Tensor):
        raise TypeError("The operator must be a Tensor type")

    result = tensor.data + other.data
    child = Tensor(result, parents=(tensor, other))

    if any((parent.requires_grad for parent in child.parents)):
        child.requires_grad = True

        def _back_grad_fn():
            tensor.grad += child.grad
            other.grad += child.grad

        child._back_grad_fn = _back_grad_fn
    return child

def _sub(tensor:Tensor, other: Tensor) -> Tensor:
    if not isinstance(other, Tensor) or not isinstance(tensor, Tensor):
        raise TypeError("The operator must be a Tensor type")

    result = tensor.data - other.data
    child = Tensor(result, parents=(tensor, other))

    if any((parent.requires_grad for parent in child.parents)):
        child.requires_grad = True

        def _back_grad_fn():
            tensor.grad += child.grad
            other.grad -= child.grad

        child._back_grad_fn = _back_grad_fn
    return child

def _mul(tensor:Tensor, other: Tensor) -> Tensor:
    if not isinstance(other, Tensor) or not isinstance(tensor, Tensor):
        raise TypeError("The operator must be a Tensor type")

    result = tensor.data * other.data
    child = Tensor(result, parents=(tensor, other))

    if any((parent.requires_grad for parent in child.parents)):
        child.requires_grad = True

        def _back_grad_fn():
            tensor.grad += (other.data * child.grad)
            other.grad += (tensor.data * child.grad)

        child._back_grad_fn = _back_grad_fn
    return child

def _pow(tensor:Tensor, exponent: float | int) -> Tensor:
    if not isinstance(tensor, Tensor):
        raise TypeError("The operator must be a Tensor type")

    result = np.power(tensor.data,exponent)
    child = Tensor(result, parents=(tensor,))

    if any((parent.requires_grad for parent in child.parents)):
        child.requires_grad = True

        def _back_grad_fn():
            grad = exponent * np.power(tensor.data,exponent-1)
            tensor.grad += (grad * child.grad)

        child._back_grad_fn = _back_grad_fn
    return child

def _matmul(tensor:Tensor, other: Tensor) -> Tensor:
    if not isinstance(other, Tensor) or not isinstance(tensor, Tensor):
        raise TypeError("The operator must be a Tensor type")

    result = np.matmul(tensor.data, other.data)

    if not isinstance(result, np.ndarray):
        result = np.array(result)

    child = Tensor(result, parents=(tensor, other))

    if any((parent.requires_grad for parent in child.parents)):
        child.requires_grad = True

        def _back_grad_fn():
            tensor.grad += np.matmul(child.grad,np.transpose(other.data))
            other.grad += np.matmul(np.transpose(tensor.data),child.grad)

        child._back_grad_fn = _back_grad_fn
    return child

def _transpose(tensor:Tensor) -> Tensor:
    if not isinstance(tensor, Tensor):
        raise TypeError("The operator must be a Tensor type")

    result = np.transpose(tensor.data)
    child = Tensor(result, parents=(tensor,))

    if any((parent.requires_grad for parent in child.parents)):
        child.requires_grad = True

        def _back_grad_fn():
            tensor.grad += np.transpose(child.grad)

        child._back_grad_fn = _back_grad_fn
    return child


class Tensor:
    """Basic torch-like tensor class.

    Attributes:
        data (Union[Tensor,np.ndarray,List,float,int]): 
            Data storing.
        parents (Optional[Tuple[Tensor]], optional):    
            Parents tensor. 
            Defaults to None.
        requires_grad (bool): 
            Defaults to False.
        dtype (Optional[str], optional): 
            Data type of tensor.
            If None,`dtype` will be equal to `data` dtype.
            Defaults to None.
    
    """    
    def __init__(
        self, 
        data: Union[Tensor,np.ndarray,List,float,int], 
        parents: Optional[Tuple[Tensor]] = None, 
        requires_grad:bool= True,
        dtype:Optional[str] = None,
    ) -> None:

        if isinstance(data, self.__class__):
            data = data.data
        elif isinstance(data, np.ndarray):
            data = data
        elif isinstance(data,List) or isinstance(data,int) or  isinstance(data,float)  :
            data=np.array(data)
        else:
            raise TypeError(f"The data type provided is not supported: {type(data)}")


        self.data = data
        if dtype is None :
            self.dtype=self.data.dtype
        else:
            self.dtype=dtype 
            self.data=self.data.astype(dtype)
        
        self._shape = self.data.shape

        self.parents = parents or ()
        self._requires_grad = requires_grad
        if requires_grad:
            self.grad= np.zeros_like(self.data,dtype=self.dtype) 
        else:
            self.grad = None


        self._back_grad_fn = lambda: None

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad: bool):
        if requires_grad:
            self.grad= np.zeros_like(self.data,dtype=self.dtype) 
        else:
            self.grad = None
        self._requires_grad = requires_grad

    @property
    def shape(self) -> Tuple:
        return self._shape

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, requires_grad={self.requires_grad})"

    def __add__(self, other: Tensor) -> Tensor:
        return _add(self,other)

    
    def __sub__(self, other: Tensor) -> Tensor:
        return _sub(self,other)

    def __mul__(self, other: Tensor) -> Tensor:
        return _mul(self,other)
    
    def __pow__(self,exponent: float|int)-> Tensor:
        return  _pow(self,exponent)

    def __matmul__(self, other: Tensor) -> Tensor:
        return  _matmul(self,other)

    def T(self)->Tensor:
        return _transpose(self)

    def backward(self) -> None:
        
        if not self.requires_grad:
            raise ValueError("The `requires_grad` should be True")

        self.grad =  np.ones(self.data.shape,dtype=self.dtype)

        tensor_queue = [self]
        while len(tensor_queue):
            tensor = tensor_queue.pop(0)
            tensor._back_grad_fn()
            tensor_queue.extend(list(tensor.parents))

class Parameter(Tensor):
    """Basic torch-like Parameter class.

    Attributes:
        shape (Union[Tuple,List]): 
            Data storing.
        requires_grad (bool): 
            Defaults to False.
        dtype (str ): 
            Data type of tensor.
            Defaults to 'float32'.
    
    """    
    def __init__(
        self, 
        shape: Tuple | List, 
        requires_grad:bool= True,
        dtype:str = 'float32',
    ) -> None:

        data= np.random.uniform(-1/2,1/2,size=shape).astype(dtype)
        Tensor.__init__(data=data,requires_grad=requires_grad,dtype=dtype)


if __name__ == "__main__":
    n1=np.array([[3,2,1],[4,3,2]])
    n2 = np.array([[5,6],[7,8],[9,1]])
    n1.shape
    n1
    n2
    n2.shape

    t1=Tensor(n1)
    t2=t1 + t1 
    t2=t2**2
    t2.backward()
    t2
    t1


    t2=Tensor(n2)
    t3=t1@t2
    t4=t3**2
    t5=t4.T()
    t5=t5**2
    t5.backward()
    t5
    t4
    t3
    t1
    t2
   