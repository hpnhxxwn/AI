import math, random

class problem_simple():
	def __init__(self, value):
		self.value = value
		self.policy = [0]*5
	def succAndReward(self, state, action):
		if state == 0 or state == 4:
			return 0
		if action == 1:
			p = 0.3
			if state  == 3:
				return (1-p)* (self.value[state-1] - 5) + p*100
			if state == 1:
				return (1-p)*20 + p*(self.value[state+1] - 5)
			else:
				return (1-p)*self.value[state-1] + p*self.value[state+1] - 5
		if action == -1:
			p = 0.2
			if state + 1 == 4:
				return (1-p)* (self.value[state-1] - 5) + p*100
			if state - 1 == 0:
				return (1-p)*20 + p*(self.value[state+1] - 5)
			else:
				return (1-p)*self.value[state-1] + p*self.value[state+1] - 5
	def returnValue(self):
		return self.value
	def returnPolicy(self):
		return self.policy
	def setValue(self, state, new):
		self.value[state] = new
	def setPolicy(self, state, new):
		self.policy[state] = new

prob = problem_simple([float(0)]*5)
iter = 2
k = 0
q = [0]*2
while k < iter:
	for i in range(1,4):
		q[1] = prob.succAndReward(i,1)
		q[0] = prob.succAndReward(i,-1)
		a = q.index(max(q))
		prob.setValue(i, max(q))
		prob.setPolicy(i, (a-0.5)*2)
	print prob.returnValue()
	print prob.returnPolicy()
	k += 1

prob.returnValue()
prob.returnPolicy()


