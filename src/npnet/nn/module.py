from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Any, Set
import os
import collections

from npnet.tensor import Tensor


class Module(ABC):
    """Torch-like module."""    

    def __init__(self) -> None:
        self._parameters: List[Tensor] = []
        self._modules: List[Module] = []
        self._training = True
        self._module_name = "Module"
        self.state_dict = collections.OrderedDict()

    @abstractmethod
    def forward(self, input: Tensor) -> Tensor:
        raise NotImplemented


    def add_parameter(self, parameter: Tensor) -> Tensor:
        self._parameters.append(parameter)
        return parameter


    def add_module(self, module: Module) -> Module:
        self._modules.append(module)
        return module


    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)


    @property
    def modules(self) -> List[Module]:
        return self._modules


    @property
    def parameters(self) -> List[Tensor]:
        modules_parameters = []
        modules = [self]
        visited_modules: Set[Module] = set([])

        while len(modules) != 0:
            module = modules.pop()

            if module in visited_modules:
                raise RecursionError("Module already visited, cycle detected.")

            modules_parameters.extend(module._parameters)
            modules.extend(module.modules)
            visited_modules.add(module)

        return modules_parameters


    def module_name(self) -> str:
        return self._module_name


    def __repr__(self) -> str:
        children_modules_description = ""
        if len(self.modules) != 0:
            children_modules_description = "Children modules:"
            modules_description_list = []

            for module in self.modules:
                modules_description_list.append(module.module_name)

            children_modules_description = f"Children modules: {os.linesep} {os.linesep.join(modules_description_list)}"

        return self.module_name + children_modules_description


if __name__ == "__main__":
    class Linear(Module):
        def __init__(self, in_size:int, out_size:int) -> None:
            super().__init__()
            weights_data: np.ndarray = np.random.uniform(size=in_size * out_size).reshape((in_size, out_size))
            self.weights = Tensor(weights_data, requires_grad=True)
            self.b = Tensor(np.random.uniform(size=out_size), requires_grad=True)

            self.add_parameter(self.weights)
            self.add_parameter(self.b)

        def forward(self, input: Tensor)->Tensor:
            tmp = input @ self.weights
            out = tmp + self.b
            return out

    tmp = Linear(10,2)


    

