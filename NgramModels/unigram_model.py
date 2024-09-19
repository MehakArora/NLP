def predict_word():
    return "the"

def predict_unigram():
    return {"the": 0.1, "cat": 0.2, "dog": 0.3}

def sample_from_distribution(dist: dict[str, float], k : int) -> str:
    """Sample a word from a distribution."""
    import random 
    population = list(dist.keys())
    weights = list(dist.values())
    print(population, weights)
    return random.choices(population, weights, k = k)

if __name__ == "__main__":
    print(predict_word())
    print(predict_unigram())
    print(sample_from_distribution(predict_unigram(), 5))