import datasets
import torch
from transformers import AutoConfig
from experiments.run_mntp import get_model_class


def get_device():
    if torch.backends.mps.is_available():
        return "mps"  # mac GPU
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def get_class_name():
    config = AutoConfig.from_pretrained(
        "dicta-il/dictalm2.0-instruct"
    )
    model_class = get_model_class(config)
    print(config.__class__.__name__)
    print(model_class)


def load_dataset_helper():
    from datasets import load_dataset
    from itertools import islice

    ds = load_dataset(
        "HeNLP/HeDC4", split="train", streaming=True)
    first_100 = list(islice(ds, 100))
    dataset = datasets.Dataset.from_list(first_100)
    print(dataset)

if __name__ == "__main__":
    load_dataset_helper()
