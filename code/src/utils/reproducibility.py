import numpy as np
import torch

# ============================================================

# Reproducibiliy

# ============================================================


# Randomness
def set_seed(seed):
    """
    Sets the seed for all frameworks randomness and reproducibiliy.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
