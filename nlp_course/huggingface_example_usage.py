import torch
from llm2vec import LLM2Vec

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"  # mac GPU
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


l2v = LLM2Vec.from_pretrained(
    base_model_name_or_path="omriel1/LLM2Vec-DictaLM2.0-mntp",
    peft_model_name_or_path="omriel1/LLM2Vec-DictaLM2.0-mntp-unsup-simcse",
    device_map=get_device(),
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

texts = [
    "היי מה קורה?",
    "הכל טוב איתך?"
]
results = l2v.encode(
    texts
)
print(results)
