import torch

# Enter your values here
seed = 1000
n = 10

# Create and shuffle

if __name__ == "__main__":
    torch.manual_seed(seed)
    shuffled = torch.randperm(n)

    print(f"Seed: {seed}, N: {n}")
    print(f"Shuffled indices: {shuffled.tolist()}")