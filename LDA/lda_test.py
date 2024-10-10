"""Latent Dirichlet Allocation

Patrick Wang, 2021
"""
from typing import List

from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import numpy as np


def lda_gen(vocabulary: List[str], alpha: np.ndarray, beta: np.ndarray, xi: int) -> List[str]:
    """Generate a document using LDA.
    """
    words = []
    n = np.random.poisson(xi)
    for _ in range(n):
        #Draw topic from direchlet distribution
        topic_distribution = np.random.dirichlet(alpha)
        topic = np.random.choice(len(topic_distribution), p=topic_distribution)
        word = np.random.choice(len(beta[0]), p=beta[topic])
        words.append(vocabulary[word])
        
    return words


def test():
    """Test the LDA generator."""
    vocabulary = [
        "bass", "pike", "deep", "tuba", "horn", "catapult",
    ]
    beta = np.array([
        [0.4, 0.4, 0.2, 0.0, 0.0, 0.0],
        [0.0, 0.3, 0.1, 0.0, 0.3, 0.3],
        [0.3, 0.0, 0.2, 0.3, 0.2, 0.0]
    ])
    alpha = np.array([0.2, 0.2, 0.2])
    xi = 50
    documents = [
        lda_gen(vocabulary, alpha, beta, xi)
        for _ in range(500)
    ]

    # Create a corpus from a list of texts
    dictionary = Dictionary(documents)
    corpus = [dictionary.doc2bow(text) for text in documents]
    model = LdaModel(
        corpus,
        id2word=dictionary,
        num_topics=3,
    )
    print(model.alpha)
    print(model.show_topics())

    #print the original beta wrt the vocabulary
    for i in range(len(beta)):
        print("Topic", i)
        for j in range(len(beta[0])):
            print(vocabulary[j], beta[i][j])
    
    #print the learned beta wrt the vocabulary
    for i in range(3):
        print("Topic", i)
        for word, prob in model.get_topic_terms(i):
            print(dictionary[word], prob)


if __name__ == "__main__":
    test()
