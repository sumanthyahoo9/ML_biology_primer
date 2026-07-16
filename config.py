MODEL_NAME = "alchemab/antiberta2"
MAX_LENGTH = 150
DTYPE = "float16"  # T4 = Turing arch, no bf16 tensor core support
TRAIN_DTYPE = "float32"
SEED = 42
BATCH_SIZE=16
LR = 2e-5
EPOCHS = 10
CHECKPOINT_DIR = "checkpoints/best_paratope_model"
