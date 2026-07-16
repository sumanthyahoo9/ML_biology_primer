import torch
from sklearn.metrics import roc_auc_score, f1_score


def softmax_probs(logits):
    return torch.softmax(logits, dim=-1)[..., 1]


def get_valid_mask(labels):
    return labels != -100


def flatten_masked(values, mask):
    return values[mask].detach().cpu().numpy()


def compute_auroc(probs, labels):
    return roc_auc_score(labels, probs)


def compute_f1(preds, labels):
    return f1_score(labels, preds)