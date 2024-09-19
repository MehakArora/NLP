import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #observed data 
    p_obs = ["apple", "apple", "apple", "apple", "apple", "apple", "banana", "banana", "banana", "banana"]

    #unigram model 
    num_apple = p_obs.count("apple")
    num_banana = p_obs.count("banana")
    p_apple = num_apple / len(p_obs)
    p_banana = num_banana / len(p_obs)
    p_unigram = {"apple": p_apple, "banana": p_banana}

    #probability of the observed data given the unigram model
    p_obs_given_unigram = p_apple ** num_apple * p_banana ** num_banana
    print(p_obs_given_unigram)

    #plot the probability distribution as a function of p_apple
    p_apple_values = np.linspace(0, 1, 100)
    p_banana_values = 1 - p_apple_values   
    p_unigram_values =  p_apple_values ** num_apple * p_banana_values ** num_banana
    plt.plot(p_apple_values, p_unigram_values, label="apple")  

    #plot the probability of the observed data given the unigram model
    plt.axvline(x=p_apple, color="red", linestyle="--", label="p(apple) = 0.6")
    plt.axhline(y=p_obs_given_unigram, color="red", linestyle="--", label="p(obs|unigram)")
    plt.title("Unigram Model")  
    plt.xlabel("p(apple)")
    plt.ylabel("probability")
    plt.legend()
    plt.show()
    plt.savefig("unigram_model.pdf")
    print(p_unigram)