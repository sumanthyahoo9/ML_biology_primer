"""
Verify the tokenization
"""
from model_loader import load_all
from data_loader import list_paratope_files
from dataset_reader import read_parquet_files
from utils.tokenizer import format_sequence


def check_token_count(tokenizer, seq):
    enc = tokenizer(format_sequence(seq))
    return len(enc["input_ids"]), len(seq) + 2


def verify_row(tokenizer, seq):
    actual, expected = check_token_count(tokenizer, seq)
    return actual == expected, actual, expected


def verify_sample(tokenizer, df, n=10):
    results = [verify_row(tokenizer, s) for s in df["sequence"].head(n)]
    passed = sum(r[0] for r in results)
    print(f"{passed}/{n} sequences aligned correctly")
    return results


def load_sample_df(n=10):
    path = list_paratope_files()[0]
    return read_parquet_files(path).head(n)


if __name__ == "__main__":
    tok, mdl = load_all()
    df = load_sample_df(10)
    results = verify_sample(tok, df, 10)
    for ok, actual, expected in results:
        if not ok:
            print(f"MISMATCH: got {actual} tokens, expected {expected}")