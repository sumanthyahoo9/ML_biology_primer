PAD_LABEL = -100
LABEL2ID = {"N": 0, "P": 1}


def format_sequence(seq):
    return " ".join(list(seq))


def encode_labels(labels, max_length):
    ids = [LABEL2ID[l] for l in labels][: max_length - 2]
    return [PAD_LABEL] + ids + [PAD_LABEL]


def tokenize_and_align(tokenizer, seq, labels, max_length):
    text = format_sequence(seq)
    enc = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")
    label_ids = encode_labels(labels, max_length)
    label_ids += [PAD_LABEL] * (max_length - len(label_ids))
    enc["labels"] = label_ids
    return enc


def build_dataset(tokenizer, df, max_length):
    return [
        tokenize_and_align(tokenizer, r.sequence, r.paratope_labels, max_length)
        for r in df.itertuples()
    ]