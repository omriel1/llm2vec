import torch


def clear_cuda_cache():
    if torch.cuda.is_available():
        print("CUDA is available.")

        # Get initial memory stats
        print(f"Initial allocated memory: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        print(f"Initial cached memory: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")

        # Empty the CUDA cache
        torch.cuda.empty_cache()
        print("Cleared the CUDA cache.")

        # Check memory stats after clearing cache
        print(f"Memory allocated after clearing: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        print(f"Memory cached after clearing: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
    else:
        print("CUDA is not available on this device.")


if __name__ == "__main__":
    clear_cuda_cache()
