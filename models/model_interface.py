from abc import abstractmethod,ABC
import torch , numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union

class Abstract_model(ABC):
    @abstractmethod
    def __call__(self,x:Union[torch.tensor, np.array]) -> Union[torch.tensor, np.array]:
        pass
