"""
Simple functions for checkpointing decisions
"""
def is_best(val_loss, best_so_far):
    return val_loss < best_so_far


def save_checkpoint(model, path):
    model.save_pretrained(path)