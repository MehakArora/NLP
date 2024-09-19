import numpy as np
import csv
import nltk
nltk.download('gutenberg')
nltk.download('punkt')

def sample_from_categorical_choice(p: dict) -> str:
    #Get the keys and values of the dictionary
    keys = list(p.keys())
    values = list(p.values())
    #Generate a random number
    random_number = np.random.choice( keys, 1, p=values)
    return random_number[0]


def finish_sentence(sentence: tuple, n: int, corpus: tuple, randomize: bool) -> tuple:
    stop_tokens = [".", "?", "!"]
    stop_length = 10
    stupid_backoff = 0.4
    sentence = list(sentence)

    corpus = list(corpus)

    #Create a dictionary of n-grams (from 1 to n)
    n_grams = {}
    for l in range(1, n+1):
        for i in range(len(corpus) - l):
            n_gram = tuple(corpus[i:i + l])
            if n_gram not in n_grams:
                n_grams[n_gram] = {}
            n_grams[n_gram][corpus[i + l]] = n_grams[n_gram].get(corpus[i + l], 0) + 1
        
        
 
    generated_sentence = sentence
    while generated_sentence[-1] not in stop_tokens and len(generated_sentence) < stop_length:
        if len(generated_sentence) < n:
            n_gram = tuple(['<s>'] * (n - len(generated_sentence)) + generated_sentence)
        else: 
            n_gram = tuple(generated_sentence[-n+1:])
        
        a = 0
        found= n_gram in n_grams.keys()
        while not found:
            a += 1
            n_gram = tuple(generated_sentence[-n+1+a:])
            found = n_gram in n_grams.keys()
            if a == n:
                return sentence + ['.']
                break

        if randomize:
            frequencies = list(n_grams[n_gram].values())
            probabilities = [((stupid_backoff)**a)*p for p in frequencies]
            probabilities = probabilities/np.sum(probabilities)
            next_token = sample_from_categorical_choice(dict(zip(list(n_grams[n_gram].keys()), probabilities)))
            generated_sentence.append(next_token)
        
        else:
            probabilities = list(n_grams[n_gram].values())
            keys = list(n_grams[n_gram].keys())
            possible_tokens = [keys[i] for i in range(len(keys)) if probabilities[i] == max(probabilities)]
            #sort possible tokens in alphabetical order
            possible_tokens.sort()
            next_token = possible_tokens[0]
            generated_sentence.append(next_token)

    return generated_sentence



def test_generator_private():
    """Test Markov text generator."""
    corpus = tuple(
        nltk.word_tokenize(nltk.corpus.gutenberg.raw("austen-sense.txt").lower())
    )
    #import pdb; pdb.set_trace()
    with open("test_examples.csv") as csvfile:
        csvreader = csv.DictReader(csvfile, delimiter=",")
        for row in csvreader:

            words = finish_sentence(
                row["input"].split(" "),
                int(row["n"]),
                corpus,
                randomize=False,
            )
            print(f"input: {row['input']} (n={row['n']})")
            print(f"output: {' '.join(words)}")
            assert words == row["output"].split(" ")

if __name__ == "__main__":
    test_generator_private()