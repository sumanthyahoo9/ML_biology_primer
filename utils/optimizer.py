"""
Optimizer Utils
"""
from torch.optim import AdamW

def build_optimizer(model, lr):
    """
    Return the optimizer with the model params and the learning rate
    """
    return AdamW(model.parameters(), lr=lr)