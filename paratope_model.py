import torch
from transformers import RoFormerForTokenClassification
from config import MODEL_NAME, TRAIN_DTYPE

NUM_LABELS = 2


def get_train_dtype():
    return torch.float32 if TRAIN_DTYPE == "float32" else torch.float16


def load_paratope_model():
    dtype = get_train_dtype()
    model = RoFormerForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, torch_dtype=dtype
    )
    return model.to("cuda")


if __name__ == "__main__":
    model = load_paratope_model()
    print(model.classifier)