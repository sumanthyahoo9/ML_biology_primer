"""
Training utilities
"""
import torch


def move_batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def token_accuracy(logits, labels):
    preds = logits.argmax(dim=-1)
    mask = labels != -100
    correct = (preds[mask] == labels[mask]).sum().item()
    return correct / mask.sum().item()


def train_step(model, batch, optimizer, device):
    batch = move_batch_to_device(batch, device)
    outputs = model(**batch)
    outputs.loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs.loss.item()


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    losses = [train_step(model, b, optimizer, device) for b in dataloader]
    return sum(losses) / len(losses)


def eval_step(model, batch, device):
    batch = move_batch_to_device(batch, device)
    with torch.no_grad():
        outputs = model(**batch)
    acc = token_accuracy(outputs.logits, batch["labels"])
    return outputs.loss.item(), acc


def evaluate(model, dataloader, device):
    model.eval()
    results = [eval_step(model, b, device) for b in dataloader]
    losses, accs = zip(*results)
    return sum(losses) / len(losses), sum(accs) / len(accs)