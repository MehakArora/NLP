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
                return (query[i], candidate[i]), 'Substitution'
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

def levensteinDP(str1, str2):
    """
    Dynamic programming implementation of the Levenshtein distance algorithm.
    str1: the query 
    str2: the suggestion
    """
    m = len(str1)
    n = len(str2)
    dp = [[0 for x in range(n+1)] for x in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1])
    return dp[m][n]

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
            
    
# Function to calculate the probability of a word given a noisy channel model
def calculateNoisyChannelProbability(candidate, word, insertion, deletion, substitution, bigrams, unigrams, wordCounter):
    probability = 0
    p_w = calculateWordProbability(candidate, wordCounter)
    if p_w == 0:
        return 0
    edit, editType = findEdit(candidate, word)
    logprobability = np.log(p_w)
    if editType == 'Insertion':
        logprobability += np.log(insertion[edit]) - np.log(unigrams[edit[0]])
    elif editType == 'Deletion':
        if edit[0] == '#':
            logprobability += np.log(deletion[edit]) - np.log(unigrams[edit[1]])
        else:
            logprobability += np.log(deletion[edit]) - np.log(bigrams[edit])
    elif editType == 'Substitution':
        logprobability += np.log(substitution[edit]) - np.log(unigrams[edit[1]])
    return probability


# Function to find the best suggestion
def best_suggestion(query, suggestions, insertion, deletion, substitution, bigrams, unigrams, wordCounter):
    best_suggestion = ''
    best_probability = 0    
    probabilities = [calculateNoisyChannelProbability(suggestion, query, insertion, deletion, 
                     substitution, bigrams, unigrams, wordCounter) for suggestion in suggestions]
    return suggestions[np.argmax(probabilities)], np.max(probabilities)

def correct(query: str) -> str:
    suggestions = [word for word in wordCounter.keys() if levensteinDP2(word, query) == 1 ]
    suggestion, probability = best_suggestion(query, suggestions, insertions, deletions, substitutions, bigrams, unigrams, wordCounter)
    query_probability = calculateWordProbability(query, wordCounter)
    if probability > query_probability:
        return suggestion
    else:
        return query 


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
    word = correct(query)
    print("Correct: ", word)