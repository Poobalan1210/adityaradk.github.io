import audll
import pickle as pkl
from mnist_web import mnist

loops = 100

audll.createNetwork(2, 4, [50, 50, 2])

data = [None] * 4
labels = [None] * 4

# left-right-down
data[0] = [1, 1, 0, 0]
data[1] = [0, 0, 1, 1]
data[2] = [1, 0, 1, 0]
data[3] = [0, 1, 0, 1]

# horizontal, vertical
labels[0] = [1, 0]
labels[1] = [1, 0]
labels[2] = [0, 1]
labels[3] = [0, 1]

audll.printProgress(0, loops)

x = 0
while x < loops:
    audll.backPropagate(labels, True, 4, data)
    audll.simpleGradientDecent(0.01, 4)
    x = x + 1
    audll.printProgress(x, loops)

with open('weights.pkl', 'wb') as f:
    pkl.dump(audll.weight, f)
with open('biases.pkl', 'wb') as f:
    pkl.dump(audll.bias, f)
