"""
Primary training script
"""
from model_loader import load_all
from paratope_model import load_paratope_model
from paratope_dataset import build_split_dataset, build_dataloader
from utils.optimizer import build_optimizer
from utils.train import train_one_epoch, evaluate
from utils.checkpoint import is_best, save_checkpoint
from config import MAX_LENGTH, BATCH_SIZE, LR, EPOCHS, CHECKPOINT_DIR


def build_dataloaders(tokenizer):
    train_ds = build_split_dataset(tokenizer, "train", MAX_LENGTH)
    val_ds = build_split_dataset(tokenizer, "val", MAX_LENGTH)
    train_dl = build_dataloader(train_ds, BATCH_SIZE, shuffle=True)
    val_dl = build_dataloader(val_ds, BATCH_SIZE, shuffle=False)
    return train_dl, val_dl


def run_epoch(model, train_dl, val_dl, optimizer, device, epoch):
    train_loss = train_one_epoch(model, train_dl, optimizer, device)
    val_loss, val_acc = evaluate(model, val_dl, device)
    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
    return val_loss


def maybe_save(model, val_loss, best_so_far):
    if is_best(val_loss, best_so_far):
        save_checkpoint(model, CHECKPOINT_DIR)
        print(f"  -> new best val_loss={val_loss:.4f}, checkpoint saved to {CHECKPOINT_DIR}")
        return val_loss
    return best_so_far


def run_training():
    tokenizer, _ = load_all()
    model = load_paratope_model()
    train_dl, val_dl = build_dataloaders(tokenizer)
    optimizer = build_optimizer(model, LR)
    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        val_loss = run_epoch(model, train_dl, val_dl, optimizer, "cuda", epoch)
        best_val_loss = maybe_save(model, val_loss, best_val_loss)
    return model


if __name__ == "__main__":
    run_training()