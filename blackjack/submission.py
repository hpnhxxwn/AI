import util, math, random
from collections import defaultdict
from util import ValueIteration

############################################################
# Problem 2a

# If you decide 2a is true, prove it in blackjack.pdf and put "return None" for
# the code blocks below.  If you decide that 2a is false, construct a counterexample.
class CounterexampleMDP(util.MDP):
    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 1
        # END_YOUR_CODE

    # Return set of actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 0
        # END_YOUR_CODE

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        if state == 0 or state == 2:
            return list()
        else:
            return [(0, 0.1, 100),(2, 0.9, 5)]
        # END_YOUR_CODE

    def discount(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 1
        # END_YOUR_CODE

############################################################
# Problem 3a

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.
    # The second element is the index (not the value) of the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.
    # The final element is the current deck.
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be done in succAndProbReward
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to None. 
    # When the probability is 0 for a particular transition, don't include that 
    # in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 53 lines of code, but don't worry if you deviate from this)
        reward = 0; outcomes = []
        total, nextCard, deckCount = state
        newState = [0]*3
        newState[1] = nextCard
        newState[2] = deckCount
        if deckCount is None or sum(deckCount) == 0:
            return []
        elif nextCard != None:
            newState[1] = None
            if action == 'Peek':
                return []
            elif action == 'Take':
                newState[0] = total + self.cardValues[nextCard]  
                newState[2] = list(newState[2])           
                newState[2][nextCard] -= 1
                if newState[0] > self.threshold:
                    newState[2] = None
                    outcomes.append((tuple(newState), 1, 0))
                elif sum(newState[2]) <= 0:
                    newState[2] = None
                    outcomes.append((tuple(newState), 1, newState[0]))
                else:
                    newState[2] = tuple(newState[2])
                    reward = newState[0]
                    outcomes.append((tuple(newState), 1, 0))
            else:
                outcomes.append(((total,None, None), 1, total))
        else:
            if action == 'Peek':
                s = sum(deckCount)
                for i in range(0, len(deckCount)):
                    if deckCount[i] != 0:
                        prob = float(deckCount[i])/s
                        outcomes.append(((total, i, deckCount), prob, -self.peekCost))
            elif action == 'Take':
                s = sum(deckCount)
                for i in range(0, len(deckCount)):
                    if deckCount[i] != 0:
                        newState = [total, None, list(deckCount)]
                        prob = float(deckCount[i])/s
                        newState[0] = total + self.cardValues[i]
                        newState[2][i] -= 1
                        newState[2] = tuple(newState[2])
                        if newState[0] > self.threshold:
                            newState[2] = None
                            outcomes.append((tuple(newState), prob, 0))
                        elif sum(newState[2]) <= 0:
                            newState[2] = None
                            outcomes.append((tuple(newState), prob, newState[0]))
                        else:
                            outcomes.append((tuple(newState), prob, 0))
            else:
                outcomes.append(((total, None, None), 1, total))
        #print outcomes
        return outcomes

        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Problem 3b

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the optimal action at
    least 10% of the time.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    return BlackjackMDP([21, 4], 100, 20, 1)
    # END_YOUR_CODE

############################################################
# Problem 4a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
        if newState == None: return
        succOpti = max(self.getQ(newState, action) for action in self.actions(newState))
        step_size = self.getStepSize()   
        curScore = self.getQ(state, action)
        res =  curScore  - reward - self.discount*succOpti
        for f, v in self.featureExtractor(state, action):
            self.weights[f] -=  v * step_size * res
         # END_YOUR_CODE

# Return a singleton list containing indicator feature for the (state, action)
# pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 4b: convergence of Q-learning
# Small test case
mdp1 = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)
rl = QLearningAlgorithm(mdp1.actions,  mdp1.discount(), identityFeatureExtractor, 0.2)

mdp1 = BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                                   threshold=10, peekCost=1)


random.seed(1)
startState = mdp1.startState()
algo = ValueIteration()
algo.solve(mdp1)
print "pi of Value iteration is:"
#print algo.pi
states = algo.pi.keys()

util.simulate(mdp1, rl, 30000)
rl.explorationProb = 0
pi_rl = {}
for state in states:
    pi_rl[state] = rl.getAction(state)

print "small test case"
#print "pi of reinforcement learning is:"
#print pi_rl

for key in pi_rl.keys():
    if set(pi_rl[key]) != set(algo.pi[key]):
        print key


# Large test case
mdp1 = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)

random.seed(1)
startState = mdp1.startState()
algo = ValueIteration()
algo.solve(mdp1)
#print "pi of Value iteration is:"
#print algo.pi
states = algo.pi.keys()

rl = QLearningAlgorithm(mdp1.actions,  mdp1.discount(), identityFeatureExtractor, 0.2)
util.simulate(mdp1, rl, 30000)
rl.explorationProb = 0
pi_rl = {}
for state in states:
    pi_rl[state] = rl.getAction(state)

print "large test case"
#print "pi of reinforcement learning is:"
#print pi_rl

diff = 0
print "difference"
for key in pi_rl.keys():
    if set(pi_rl[key]) != set(algo.pi[key]):
        diff += 1
        #print key, pi_rl[key], algo.pi[key]

print diff



############################################################
# Problem 4c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs (see
# identityFeatureExtractor()).
# Implement the following features:
# - indicator on the total and the action (1 feature).
# - indicator on the presence/absence of each card and the action (1 feature).
#       Example: if the deck is (3, 4, 0 , 2), then your indicator on the presence of each card is (1,1,0,1)
#       Only add this feature if the deck != None
# - indicator on the number of cards for each card type and the action (len(counts) features).  Only add these features if the deck != None
def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state
    # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
    features = []
    key1 = (total, action)
    features.append((key1, 1))
    if (counts != None) and (sum(counts) != 0):
        key2 = ['presence']
        for i in counts:
            if i == 0:
                key2.append(1)
            else:
                key2.append(0)
        key2.append(action)
        features.append((tuple(key2), 1))
        for i in range(len(counts)):
            key3 = (i, counts[i], action)
            features.append((key3, 1))

    return features
    # END_YOUR_CODE

############################################################
# Problem 4d: What happens when the MDP changes underneath you?!

# Original mdp
originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)
random.seed(1)
algo = ValueIteration()
algo.solve(originalMDP)
print "pi of Value iteration is:"
print algo.pi
states = algo.pi.keys()
# New threshold
newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)
s = util.simulate(newThresholdMDP ,util.FixedRLAlgorithm(algo.pi), 30000)
s = sum(s)
print s
s = util.simulate(newThresholdMDP ,QLearningAlgorithm(newThresholdMDP.actions, newThresholdMDP.discount(), identityFeatureExtractor, 0.2), 30000)
s = sum(s)
print s
