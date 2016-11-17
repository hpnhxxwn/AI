#!/usr/bin/python

import random
import collections
import math
import sys
import numpy as np
from collections import Counter
from util import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    return Counter(x.split())
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

def incrementSparseVector(v1, scale, v2):
    """
        Given two sparse vectors |v1| and |v2|, perform v1 += scale * v2.
        This function will be useful later for linear classifiers.
        """
    for k in v2.keys():
        v1[k] += scale*v2[k]
    
    return v1

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    #extract features from both training samples and test sample
    XTrain = []; YTrain = []
    for (x,y) in trainExamples:
        XTrain.append(featureExtractor(x))
        YTrain.append(y)
    #initialize w
        for z in XTrain[-1]:
            if z not in weights:
                weights[z] = 0
    #stochastic gradient decent using traning sets
    for i in range(numIters):
        for k in range(len(YTrain)):
            if dotProduct(weights, XTrain[k])*YTrain[k] < 1:
                incrementSparseVector(weights, eta*YTrain[k], XTrain[k])
                #trainError = evaluatePredictor(trainExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
                #testError = evaluatePredictor(testExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
                #print "Train error = %s, test error = %s" %(trainError, testError)
    # END_YOUR_CODE
    return weights

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        phi = Counter()
        for x in weights:
            if random.randint(0,1) == 1:
                phi[x] = 1

        y = 1 if dotProduct(phi, weights) >= 0 else -1
        # END_YOUR_CODE
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)       
        s = x.replace(" ", "")
        k = 0
        features = Counter()
        while k <= len(s)-n:
            features[s[k:(k+n)]] += 1
            k += 1

        return features
        # END_YOUR_CODE
    return extract


############################################################
# Problem 4: k-means
############################################################
def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 32 lines of code, but don't worry if you deviate from this)
    n = len(examples)
    #cluster centroid
    l = random.sample(range(0, n), K)
    mu = [examples[k] for k in l]
    mu2 = [0]*K
    #cluster assignments
    z_old = [1] * n
    z = [0] * n
    t = 1
    dataProduct = [0]*n
    for i in range(n):
        dataProduct[i] = dotProduct(examples[i], examples[i])

    while t < maxIters and not z_old == z :
        #assign clusters 
        totalcost = 0
        z_old = z[:]
        for k in range(K):
            mu2[k] = dotProduct(mu[k], mu[k])
        #because mu could be non-sparse (mu - x)^2 = mu^2 +x^2 - 2mu*x
        for i in range(n):
            dmin = float('inf')
            for j in range(K):
                dcur = mu2[j] + dataProduct[i] - 2*dotProduct(examples[i], mu[j])
                if(dcur < dmin):
                    dmin = dcur
                    z[i] = j
            totalcost += float(dmin) #+float(productMatrix[i,i])         
        #update mu
        t += 1
        for k in range(K):
            myPoints = [examples[i] for i in range(n) if z[i] == k]
            mu[k] = collections.defaultdict(float)
            if(len(myPoints) > 0):
                for point in myPoints:
                    for key in point:
                        mu[k][key] += point[key]/float(len(myPoints))
    return mu, z, totalcost

    # END_YOUR_CODE
