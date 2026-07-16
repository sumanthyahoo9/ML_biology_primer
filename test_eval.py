"""
Test the trained model 
"""
import numpy as np
import torch
from transformers import RoFormerForTokenClassification
from model_loader import load_tokenizer
from paratope_model import NUM_LABELS
from paratope_dataset import build_split_dataset, build_dataloader
from utils.train import move_batch_to_device
from utils.metrics import softmax_probs, get_valid_mask, flatten_masked, compute_auroc, compute_f1
from config import MAX_LENGTH, BATCH_SIZE, CHECKPOINT_DIR


def load_best_model():
    model = RoFormerForTokenClassification.from_pretrained(CHECKPOINT_DIR, num_labels=NUM_LABELS)
    return model.to("cuda").eval()


def collect_batch(model, batch, device):
    batch = move_batch_to_device(batch, device)
    with torch.no_grad():
        outputs = model(**batch)
    mask = get_valid_mask(batch["labels"])
    probs = flatten_masked(softmax_probs(outputs.logits), mask)
    preds = flatten_masked(outputs.logits.argmax(dim=-1), mask)
    labels = flatten_masked(batch["labels"], mask)
    return probs, preds, labels


def collect_all(model, dataloader, device):
    results = [collect_batch(model, b, device) for b in dataloader]
    probs = np.concatenate([r[0] for r in results])
    preds = np.concatenate([r[1] for r in results])
    labels = np.concatenate([r[2] for r in results])
    return probs, preds, labels


def run_test_eval():
    tokenizer = load_tokenizer()
    model = load_best_model()
    test_ds = build_split_dataset(tokenizer, "test", MAX_LENGTH)
    test_dl = build_dataloader(test_ds, BATCH_SIZE, shuffle=False)
    probs, preds, labels = collect_all(model, test_dl, "cuda")
    auroc = compute_auroc(probs, labels)
    f1 = compute_f1(preds, labels)
    print(f"Test AUROC={auroc:.4f}  F1={f1:.4f}")
    return auroc, f1


if __name__ == "__main__":
    run_test_eval()