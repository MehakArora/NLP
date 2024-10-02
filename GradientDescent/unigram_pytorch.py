"""Pytorch."""

import nltk
import numpy as np
from numpy.typing import NDArray
import torch
from typing import List, Optional
from torch import nn
import matplotlib.pyplot as plt


FloatArray = NDArray[np.float64]


def onehot(vocabulary: List[Optional[str]], token: Optional[str]) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary), 1))
    try:
        idx = vocabulary.index(token)
    except ValueError:
        idx = len(vocabulary) - 1
    embedding[idx, 0] = 1
    return embedding


def loss_fn(logp: float) -> float:
    """Compute loss to maximize probability."""
    return -logp


class Unigram(nn.Module):
    def __init__(self, V: int):
        super().__init__()

        # construct uniform initial s
        s0 = np.ones((V, 1))
        self.s = nn.Parameter(torch.tensor(s0.astype(float)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # convert s to proper distribution p
        logp = torch.nn.LogSoftmax(0)(self.s)

        # compute log probability of input
        return torch.sum(input, 1, keepdim=True).T @ logp


def gradient_descent_example():
    """Demonstrate gradient descent."""
    # generate vocabulary
    vocabulary = [chr(i + ord("a")) for i in range(26)] + [" ", None]

    # generate training document
    text = nltk.corpus.gutenberg.raw("austen-sense.txt").lower()

    # tokenize - split the document into a list of little strings
    tokens = [char for char in text]

    # generate one-hot encodings - a V-by-T array
    encodings = np.hstack([onehot(vocabulary, token) for token in tokens])

    # convert training data to PyTorch tensor
    x = torch.tensor(encodings.astype(float))

    # define model
    model = Unigram(len(vocabulary))

    # set number of iterations and learning rate
    num_iterations =  250
    learning_rate =  0.01

    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_iter = []
    for _ in range(num_iterations):
        logp_pred = model(x)
        loss = loss_fn(logp_pred)
        loss.backward()
        loss_iter.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()

    # display results
    #Plot the loss curve 
    loss_curve = plt.figure(figsize=(10, 5))
    loss_curve.suptitle("Loss v/s iterations")
    plt.plot(range(num_iterations), loss_iter)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig("loss_curve.png")

    #Minimum loss
    print("Minimum loss:", min(loss_iter))
    
    #Optimal and model distribution
    tokens = np.array(tokens)
    optimal = np.array([sum(tokens == x)/len(tokens) for x in vocabulary])
    optimal_logsoftmax = np.log(np.exp(optimal[:-1]) / np.sum(np.exp(optimal[:-1])))
    print("Minimal possible loss: ", -np.sum(optimal_logsoftmax))
    print("Optimal distribution")
    print(list(zip(vocabulary, optimal)))


    # display final distribution
    s = model.s.detach().numpy()
    p = np.exp(s) / np.sum(np.exp(s))
    print("Model distribution")
    print(list(zip(vocabulary, p.flatten())))
    
    #Plot the optimal and model distribution
    distribution = plt.figure(figsize=(10, 5))
    distribution.suptitle("Optimal and Model distribution")
    vocabulary = vocabulary[:-1] + ['<unk>']
    plt.bar(vocabulary, optimal, label="Optimal")
    plt.bar(vocabulary, p.flatten(), label="Model", alpha=0.6)
    plt.xlabel("Tokens")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig("distribution.png")


if __name__ == "__main__":
    import time 
    start = time.time()
    gradient_descent_example()
    end = time.time()
    print("Time taken: ", end-start)