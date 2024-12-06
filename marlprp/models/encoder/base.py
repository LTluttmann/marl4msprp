import abc
from typing import TypedDict
from torch import Tensor
import torch.nn as nn
from tensordict import TensorDict

class BaseEncoder(nn.Module, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> TensorDict:
        pass


class OperationsEncoderOutput(TypedDict):
    operations: Tensor

class MatNetEncoderOutput(TypedDict):
    operations: Tensor
    machines: Tensor