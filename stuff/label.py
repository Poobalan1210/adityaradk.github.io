import audll
import random
import numpy as np
import pickle as pkl

audll.createNetwork(2, 4, [50, 50, 2])
audll.weight = None
audll.bias = None

with open('weights.pkl', 'rb') as f:
   audll.weight = pkl.load(f)
with open('biases.pkl', 'rb') as f:
   audll.bias = pkl.load(f)

x = [1, 1, 0, 0]

audll.feedForward(x)
print(audll.output)
