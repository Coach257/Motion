import numpy as np
import math
import copy
import math
import torch
from torch import nn
import torch.nn.functional as F
L = torch.tensor([np.exp(-(x) ** 2 / 2) / (math.sqrt(2 * math.pi) ) for x in range (-3,4,1)])
print(L)
