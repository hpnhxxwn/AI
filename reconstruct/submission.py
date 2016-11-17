import shell
import util
import wordsegUtil

############################################################
# Problem 1b: Solve the segmentation problem under a unigram model

class SegmentationProblem(util.SearchProblem):
    def __init__(self, query, unigramCost):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 0  
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        #reach the end of the query
        return len(self.query) == state
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
        # Return a list of possible choices start from state(where to cut, next state, cost)-(action, newState, cost)
        succ_choices = []
        for i in range(state+1, len(self.query)+1):
            succ_choices.append((i-state, i, self.unigramCost(self.query[state:i])))
        return succ_choices
        # END_YOUR_CODE

def segmentWords(query, unigramCost):
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(SegmentationProblem(query, unigramCost))

    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    words = []; state = 0
    for i in ucs.actions:
        words.append(query[state:(state + i)])
        state = state + i
    return ' '.join(words)
    # END_YOUR_CODE

############################################################
# Problem 2b: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords, bigramCost, possibleFills):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        # (location, word at the location)
        return (0, wordsegUtil.SENTENCE_BEGIN)
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        return len(self.queryWords) == state[0]
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
        # return all possible combinations (action, newState, cost) -in our problem, 
        #actions is how you choose a pair at state (state, state+1)
        #total cost is the bigram cost of the action
        succ_choices = []
        succ, pre_w = state
        possibleSucc = self.possibleFills(self.queryWords[succ])
        #if no fills 
        if len(possibleSucc) == 0:
            possibleSucc = set([self.queryWords[succ]])
        for succ_w in possibleSucc:
            succ_choices.append((succ_w, (succ+1, succ_w), self.bigramCost(pre_w, succ_w)))
        return succ_choices

        # END_YOUR_CODE

def insertVowels(queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    if len(queryWords) == 0:
        return ''
    ucs = util.UniformCostSearch(verbose = 0)
    ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))
    #setence = []
    #for word in ucs.actions:
    #    setence.append(word)
    #return ' '.join(setence)
    return ' '.join(ucs.actions)
    # END_YOUR_CODE

############################################################
# Problem 3b: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query, bigramCost, possibleFills):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        return (0, wordsegUtil.SENTENCE_BEGIN)
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        return len(self.query) == state[0]
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 15 lines of code, but don't worry if you deviate from this)
        #we need to consider first all cut possibility and then calculate the bigram cost
        #(action, newState, cost) = (new word = where to cut and what to fill, (cur location, previous word), bigramCost)
        cur_loc, pre_w = state
        choices = []
        for i in range(cur_loc+1, len(self.query)+1):
            succPiece = self.query[cur_loc:i]
            possibleSucc = self.possibleFills(succPiece)
            if len(possibleSucc) == 0:
                possibleSucc = set([succPiece])
            for word in possibleSucc:
                choices.append((word, (i, word), self.bigramCost(pre_w, word)))
        return choices



        # END_YOUR_CODE

def segmentAndInsert(query, bigramCost, possibleFills):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    ucs =  util.UniformCostSearch(verbose = 0)
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills))
    return ' '.join(ucs.actions)
    # END_YOUR_CODE

############################################################

if __name__ == '__main__':
    shell.main()
