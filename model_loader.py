"""
Load a specific model
"""
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from config import MODEL_NAME, DTYPE


def get_torch_dtype():
    """
    Get the PyTorch datatype
    """
    return torch.float16 if DTYPE == "float16" else torch.bfloat16


def load_tokenizer():
    """
    Load the tokenizer from Hugging-face
    """
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def load_model():
    """
    Load a model
    """
    dtype = get_torch_dtype()
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    return model.to("cuda")


def load_all():
    """
    Load BOTH the tokenizer and the model
    """
    tokenizer = load_tokenizer()
    model = load_model()
    return tokenizer, model


if __name__ == "__main__":
    tok, mdl = load_all()
    print(f"Loaded {MODEL_NAME} in {get_torch_dtype()}")