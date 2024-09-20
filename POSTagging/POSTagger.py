import numpy as np
import nltk 
from viterbi import viterbi
from collections import Counter

nltk.download('universal_tagset')
nltk.download('brown')
train = nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]
test = nltk.corpus.brown.tagged_sents(tagset='universal')[10150:10153]

# Probably would have been more efficient to combine all these functions into a single one to avoid all these loops

# Get all possible words and tags
def getWordTagCounts(taggedSentences):
    wordCounter = Counter()
    tagCounter = Counter()
    for sentence in taggedSentences:
        for word, tag in sentence:
            wordCounter[word] += 1
            tagCounter[tag] += 1
    words_array = np.array(list(wordCounter.keys()))
    tags_array = np.array(list(tagCounter.keys()))
    return words_array, tags_array, wordCounter, tagCounter

# Get initial probabilities
def initialProbabilities(taggedSentences, tags):
    initialCount = np.zeros(len(tags))
    for sentence in taggedSentences:
        initialCount[tags == sentence[0][1]] += 1
    return initialCount/len(taggedSentences)


# Get transition probabilities and emission probabilities
def transObsProbabilities(taggedSentences, tags, tagCount, words, wordCount):
    tpMatrix = np.zeros((len(tags), len(tags)))
    obMatrix = np.zeros((len(words), len(tags)))
    for sentence in taggedSentences:
        for i in range(len(sentence)):
            if i < len(sentence) - 1:
                tpMatrix[tags == sentence[i][1], tags == sentence[i+1][1]] += 1
            obMatrix[words == sentence[i][0], tags == sentence[i][1]] += 1
    
    
    tpMatrix = tpMatrix/np.array([[tagCount[x] for x in tags]]).T
    obMatrix = obMatrix/np.array([[wordCount[x] for x in words]]).T
    obMatrix = np.append(obMatrix, np.zeros((1, len(tags))), axis=0)
    obMatrix[-1, :] += 1/len(tags) # Add one for unknown words

    return tpMatrix, obMatrix

# Get the most likely tag sequence for a sentence
def mostLikelyTagSequence(sentence, pi, tpMatrix, obMatrix, tags, words):
    obs = [list(words).index(word) if word in list(words) else len(words) for word, tag in sentence]
    return [list(tags)[tag] for tag in viterbi(obs, pi, tpMatrix, obMatrix)[0]]

# Calculate precision, recall, and accuracy
def calculate_metrics(confusion_matrix):
    # Initialize arrays to store precision and recall for each class
    precision = np.zeros(len(tags))
    recall = np.zeros(len(tags))
    
    # Calculate precision and recall for each class
    for i in range(len(tags)):
        true_positives = confusion_matrix[i, i]
        false_positives = np.sum(confusion_matrix[:, i]) - true_positives
        false_negatives = np.sum(confusion_matrix[i, :]) - true_positives
        
        precision[i] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall[i] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Calculate overall accuracy
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    
    return precision, recall, accuracy

if __name__ == "__main__":
    words, tags, wordCount, tagCount = getWordTagCounts(train)
    pi = initialProbabilities(train, tags)
    tpMatrix, obMatrix = transObsProbabilities(train, tags, tagCount, words, wordCount)
    outputs = []
    gts = []
    for sentence in test:
        output = mostLikelyTagSequence(sentence, pi, tpMatrix, obMatrix.T, tags, words)
        gt = [tag for word, tag in sentence]
        outputs = outputs + output
        gts = gts + gt
        print("Sentence: ", [word for word, tag in sentence])
        print("Output: ", output)
        print("Ground Truth: ", gt)
    
    #confusion matrix and metrics 
    confusion_matrix = np.zeros((len(tags), len(tags)))
    for i in range(len(outputs)):
        confusion_matrix[tags == gts[i], tags == outputs[i]] += 1
    print(confusion_matrix)

    precision, recall, accuracy = calculate_metrics(confusion_matrix)
    print("Average Precision for all tags: ", np.mean(precision))
    print("Average Recall for all tags: ", np.mean(recall))
    print("Accuracy: ", accuracy)

    print("Precision for each tag: \n", list(zip(tags, precision)))
    print("Recall for each tag: \n", list(zip(tags, recall)))
    