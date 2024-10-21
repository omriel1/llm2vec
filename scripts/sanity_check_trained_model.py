import argparse
import torch
import torch.nn.functional as F
from llm2vec import LLM2Vec

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"  # mac GPU
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def main():
    # This loads the model onto the GPU in bfloat16 precision
    parser = argparse.ArgumentParser(description="Load model for text generation.")
    parser.add_argument('--path', type=str, required=True, help="Path to the model")
    args = parser.parse_args()

    assert args.path is not None, "You should pass --path variable. For example:\n" \
                                  "python scripts/sanity_check_trained_model.py --path ./output/mntp/dictalm2.0-instruct"

    l2v = LLM2Vec.from_pretrained(
        base_model_name_or_path="dicta-il/dictalm2.0-instruct",
        peft_model_name_or_path=args.path,
        device_map=get_device(),
        torch_dtype=torch.bfloat16,
    )

    documents = [
        "אני אוהב לאכול פסטה",
        "פיצה זה ממש טעים",
        "מלחמת העולם השנייה החלה בשנת 1939"
    ]

    vectors = l2v.encode(documents)
    print(vectors.shape)
    print(vectors)
    cos_sim_1_2 = F.cosine_similarity(vectors[0], vectors[1], dim=0)
    cos_sim_1_3 = F.cosine_similarity(vectors[0], vectors[2], dim=0)
    cos_sim_2_3 = F.cosine_similarity(vectors[1], vectors[2], dim=0)

    print(f"Cosine similarity between vector 1 and 2: {cos_sim_1_2.item()}")
    print(f"Cosine similarity between vector 1 and 3: {cos_sim_1_3.item()}")
    print(f"Cosine similarity between vector 2 and 3: {cos_sim_2_3.item()}")


if __name__ == "__main__":
    main()
