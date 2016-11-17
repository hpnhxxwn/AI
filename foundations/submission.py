import collections

############################################################
# Problem 3a

def computeMaxWordLength(text):
    """
    Given a string |text|, return the longest word in |text|.  If there are
    ties, choose the word that comes latest in the alphabet.
    A word is defined by a maximal sequence of characters without whitespaces.
    You might find max() and list comprehensions handy here.
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return(max(sorted(text.split(), reverse = True), key = len))
    # END_YOUR_CODE

############################################################
# Problem 3b

def manhattanDistance(loc1, loc2):
    """
    Return the Manhattan distance between two locations, where the locations
    are pairs of numbers (e.g., (3, 5)).
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return(sum(tuple(abs(i-j) for i,j in zip(loc1,loc2))))
    # END_YOUR_CODE

############################################################
# Problem 3c
#a helper recursive function
def search_sentence(cur_word, cur_sentence, sentence_len, D, collection):
    if sentence_len == len(cur_sentence) + 1:  
        collection.add(' '.join(cur_sentence + [cur_word]))
    elif sentence_len > len(cur_sentence)+1 and cur_word in D:
        for next_word in D[cur_word]:
            search_sentence(next_word, cur_sentence + [cur_word], sentence_len, D, collection)    


def mutateSentences(sentence):
    """
    High-level idea: generate sentences similar to a given sentence.
    Given a sentence (sequence of words), return a list of all possible
    alternative sentences of the same length, where each pair of adjacent words
    also occurs in the original sentence. (The words within each pair should appear 
    in the same order in the output sentence as they did in the orignal sentence.)
    Notes:
    - The order of the sentences you output doesn't matter.
    - You must not output duplicates.
    - Your generated sentence can use a word in the original sentence more than
      once.
    """
    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
    words = sentence.split()
    D = {}
    collection = set()
    #create a dictionary with the key being the previous words and content be its next possibility
    for i in range(len(words)-1):
        if words[i] in D:
            D[words[i]].append(words[i+1])
        else:
            D[words[i]] = [words[i+1]]
    ##exhaustive search
    for word in D:
        D[word] = list(set(D[word]))

    for word in D:
        search_sentence(word, [], len(words), D, collection)

    return(collection)
    # END_YOUR_CODE

############################################################
# Problem 3d

def sparseVectorDotProduct(v1, v2):
    """
    Given two sparse vectors |v1| and |v2|, each represented as collection.defaultdict(float), return
    their dot product.
    You might find it useful to use sum() and a list comprehension.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    return sum(v1[k]*v2[k] for k in v1 and v2)
    # END_YOUR_CODE

############################################################
# Problem 3e

def incrementSparseVector(v1, scale, v2):
    """
    Given two sparse vectors |v1| and |v2|, perform v1 += scale * v2.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    for k in v2.keys():
        v1[k] += scale*v2[k]

    return v1
    # END_YOUR_CODE

############################################################
# Problem 3f

def computeMostFrequentWord(text):
    """
    Splits the string |text| by whitespace and returns two things as a pair: 
        the set of words that occur the maximum number of times, and
    their count, i.e.
    (set of words that occur the most number of times, that maximum number/count)
    You might find it useful to use collections.defaultdict(float).
    """
    # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
    cnt = collections.Counter(text.split())
    max_cnt = max(cnt.values())
    output = []
    output.append([]); output.append(max_cnt)
    for item, value in cnt.items():
        if value == max_cnt:
            output[0].append(item)

    output[0] = set(output[0])
    return(output)
    # END_YOUR_CODE

############################################################
# Problem 3g

def computeLongestPalindrome(text):
    """
    A palindrome is a string that is equal to its reverse (e.g., 'ana').
    Compute the length of the longest palindrome that can be obtained by deleting
    letters from |text|.
    For example: the longest palindrome in 'animal' is 'ama'.
    Your algorithm should run in O(len(text)^2) time.
    You should first define a recurrence before you start coding.
    """
    # BEGIN_YOUR_CODE (our solution is 19 lines of code, but don't worry if you deviate from this)
    cache = {}
    def recurse(m,n, text):
        if (m,n) in cache:
            result = cache[(m,n)]
        elif m==n:
            result = 1
        elif m > n:
            result = 0
        elif text[m] == text[n]:
            result = 2 + recurse(m+1, n-1, text)
        else:
            result1 = recurse(m+1,n, text)
            result2 = recurse(m,n-1, text)
            result = max(result1, result2)

        
        cache[(m,n)] = result
        return result

    return recurse(0, len(text)-1, text)



    # END_YOUR_CODE
