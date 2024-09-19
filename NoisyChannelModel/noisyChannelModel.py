#Build a spelling corrector that corrects corrupted words according to a noisy-channel model composed of
#a unigram model and weighted-Levenshtein-distance error model.

import sys
import re
import math
import numpy as np
from collections import Counter


# Function to read the unigram model
def readWordModel():
    # Read the unigram model
    wordModel = open('count_1w.txt', 'r')
    wordModel = wordModel.readlines()
    return wordModel

# Function to create a dictionary of the unigram model
def createWordCounter(wordModel):
    words = Counter()
    for line in wordModel:
        line = line.split('\t')
        words[line[0]] = int(line[1])
    return words

#Read the files with the noisy channel model
def readNoisyChannelFiles(filename):
    with open(filename, 'r') as file:
        model = file.readlines()
    return model 

#Create a dictionary of the noisy channel model
def createNoisyChannelModel(model):
    errorModel = Counter()
    for line in model[1:]:
        line = line.strip('\n').split(',')
        if len(line) == 2:
            errorModel[line[0]] = int(line[1])
        else:
            errorModel[(line[0], line[1])] = int(line[2])
    return errorModel

#Levenshtein distance
# This function is adapted from GeeksForGeeks: https://www.geeksforgeeks.org/introduction-to-levenshtein-distance/
def levenshteinRecursive(str1, str2, l1, l2):
    """
    Recursive implementation of the Levenshtein distance algorithm.
    str1: the query 
    str2: the suggestion
    """
    #print(l1, l2, str1, str2)
      # str1 is empty
    if l1 == 0:
        return l2
    # str2 is empty
    if l2 == 0:
        return l1
    if str1[l1 - 1] == str2[l2 - 1]:
        return levenshteinRecursive(str1, str2, l1 - 1, l2 - 1)
    return 1 + min(
          # Insert     
        levenshteinRecursive(str1, str2, l1, l2 - 1),min(
              # Remove
            levenshteinRecursive(str1, str2, l1 - 1, l2),
          # Replace
            levenshteinRecursive(str1, str2, l1 - 1, l2 - 1))
    )

#Find the edit between candidate and query
def findEdit(candidate, query):
    """
    Find the edit between the candidate and the query
    candidate: the suggestion
    query: the query
    """
    candidate = '#' + candidate
    query = '#' + query
    if len(candidate) == len(query):
        for i in range(len(candidate)):
            if candidate[i] != query[i]:
                return (candidate[i], query[i]), 'Substitution'
    elif len(candidate) > len(query): #Case of Deletion 
        for i in range(len(query)):
            if candidate[i] != query[i]:
                return (candidate[i-1], candidate[i]), 'Deletion'
        return (candidate[-2], candidate[-1]), 'Deletion'
    elif len(candidate) < len(query): #Case of Insertion
        for i in range(len(candidate)):
            if candidate[i] != query[i]:
                return (candidate[i-1], query[i]), 'Insertion'
        return (candidate[-1], query[-1]), 'Insertion'


def levensteinDP2(str1, str2):
    """
    Dynamic programming implementation of the Levenshtein distance algorithm.
    str1: the query (represented as the rows of the dp matrix) 
    str2: the suggestion (represented as the columns of the dp matrix)
    """
    m = len(str1)
    n = len(str2)
    dp = np.zeros((m+1, n+1)) #Add one because we start creating the matrix by comparing empty strings at pos (0,0)
    
    #Filling the first column with the distance between the empty string and the query
    for i in range(m+1):
        dp[i][0] = i 
    
    #Filling the first row with the distance between the empty string and the suggestion
    for j in range(n+1):
        dp[0][j] = j
    
    #Filling the rest of the matrix
    for i in range(1, m+1):
        for j in range(1, n+1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]  #No operation needed
            else:
                dp[i][j] = 1 + min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1]) #Min edit way to use Insertion, Deletion or Substitution
    return dp[m][n]
    

# Function to calculate the probability of a word
def calculateWordProbability(word, wordCounter):
    if word in wordCounter:
        return wordCounter[word]/sum(wordCounter.values())
    else:
        return 0

def log(x):
    if x == 0:
        return -1000
    return np.log(x)
    
# Function to calculate the probability of a word given a noisy channel model
def calculateNoisyChannelProbability(candidate, word, insertion, deletion, substitution, bigrams, unigrams, wordCounter):
    probability = 0
    p_w = calculateWordProbability(candidate, wordCounter)
    if p_w == 0:
        return 0
    edit, editType = findEdit(candidate, word)
    logprobability = log(p_w)
    if editType == 'Insertion':
        if edit[0] == '#':
            logprobability += log(insertion[edit]) - log(unigrams[edit[1]] )
        else:
            logprobability += log(insertion[edit] ) - log(unigrams[edit[0]] )
    elif editType == 'Deletion':
        if edit[0] == '#':
            logprobability += log(deletion[edit]) - log(unigrams[edit[1]])
        else:
            edit = edit[0] + edit[1]
            logprobability += log(deletion[edit]) - log(bigrams[edit] )
    elif editType == 'Substitution':
        logprobability += log(substitution[edit] ) - log(unigrams[edit[1]] )
    return logprobability


# Function to find the best suggestion
def best_suggestion(query, suggestions, insertion, deletion, substitution, bigrams, unigrams, wordCounter): 
    probabilities = [calculateNoisyChannelProbability(suggestion, query, insertion, deletion, 
                     substitution, bigrams, unigrams, wordCounter) for suggestion in suggestions]
    
    max_prob = np.max(probabilities)
    suggestions = np.array(suggestions)
    probabilities = np.array(probabilities)
    best_suggestions = suggestions[probabilities == max_prob]
    if len(best_suggestions) == 1:
        return best_suggestions[0], max_prob
    else:
        word_probs = [calculateWordProbability(word, wordCounter) for word in best_suggestions]
        return best_suggestions[np.argmax(word_probs)], max_prob
    
def correct(query: str) -> str:
    if query in wordCounter:
        return query
    suggestions = [word for word in wordCounter.keys() if levensteinDP2(word, query) == 1 ]
    suggestion, probability = best_suggestion(query, suggestions, insertions, deletions, substitutions, bigrams, unigrams, wordCounter)
    return suggestion


wordModel = readWordModel()
# Create a dictionary of the unigram model
wordCounter = createWordCounter(wordModel)
# Read the noisy channel model
insertions = createNoisyChannelModel(readNoisyChannelFiles('additions.csv'))
deletions = createNoisyChannelModel(readNoisyChannelFiles('deletions.csv'))
substitutions = createNoisyChannelModel(readNoisyChannelFiles('substitutions.csv'))
bigrams = createNoisyChannelModel(readNoisyChannelFiles('bigrams.csv'))
unigrams = createNoisyChannelModel(readNoisyChannelFiles('unigrams.csv'))

if __name__ == '__main__':
    query = sys.argv[1]
    print("Query: ", query)
    #import pdb; pdb.set_trace()
    word = correct(query)
    print("Correct: ", word)