import sys
import time
import math
import numpy as np
import pickle as pkl
from operator import add
from random import choice

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def sigprime (x):
    df = x * (1 - x)
    return df

def createNetwork(layer, di, dlo):
    global layervalues
    global output
    global bias
    global weight
    global layers
    global gradient
    global layerd
    global indexes
    indexes = dlo
    gradient = [None] * 2
    layervalues = [None] * layer
    layerd = [None] * layer
    layers = layer + 1
    gradient[1] = [None] * (layer + 1)
    bias = [None] * (layer + 1)
    a = 0
    while a < layer:
        layervalues[a] = [0] * dlo[a]
        layerd[a] = [0] * dlo[a]
        gradient[1][a] = [0] * dlo[a]
        bias[a] = [((choice([i for i in range(0,9) if i not in [0, 5]]))/10)] * dlo[a]
        a = a + 1
    bias[-1] = [((choice([i for i in range(0,9) if i not in [0, 5]]))/10)] * dlo[-1]
    gradient[1][-1] = [0] * dlo[-1]
    weight = [None] * (layer + 1)
    gradient[0] = [None] * (layer + 1)
    output = [0] * dlo[-1]
    x = 0
    while x < layer:
        y = 0
        weight[x] = [None] * dlo[x]
        gradient[0][x] = [None] * dlo[x]
        while y < dlo[x]:
            if x == 0:
                weight[0][y]= [((choice([i for i in range(0,9) if i not in [0, 5]]))/10)] * di
                gradient[0][0][y]= [0] * di
            else:
                weight[x][y]= [((choice([i for i in range(0,9) if i not in [0, 5]]))/10)] * dlo[(x - 1)]
                gradient[0][x][y] = [0] * dlo[(x - 1)]
            y = y + 1
        x = x + 1
    z = 0
    weight[-1] = [None] * dlo[-1]
    gradient[0][-1] = [None] * dlo[-1]
    while z < dlo[-1]:
        weight[-1][z] = [((choice([i for i in range(0,9) if i not in [0, 5]]))/10)] * dlo[-2]
        gradient[0][-1][z] = [0] * dlo[-2]
        z = z + 1

def feedForward(input):
    global layervalues
    global output
    global weight
    global layers
    global ipg
    ipg = None
    ipg = input
    x = 0 # layer
    while x < layers: # loop through layers
        if x == 0: # input
            y = 0
            while y < len(layervalues[x]): # loop through neurons
                layervalues[x][y] = sigmoid(np.sum(np.multiply(weight[x][y], input)) + bias[x][y])
                #print(layervalues[x][y])
                y = y + 1
        elif x == (layers - 1):
            y = 0
            while y < len(output): # loop through neurons
                output[y] = sigmoid(np.sum(np.multiply(weight[x][y], layervalues[(x - 1)])) + bias[x][y])
                #print(layervalues[(x - 1)])
                y = y + 1
        else:
            y = 0
            while y < len(layervalues[x]): # loop through neurons
                layervalues[x][y] = sigmoid(np.sum(np.multiply(weight[x][y], layervalues[(x - 1)])) + bias[x][y])
                y = y + 1
        x = x + 1

def backPropagate(itlab, yc=False, loop=0, data=None, start=0, boo=False):
    global layervalues
    global output
    global weight
    global layers
    global layerd
    global gradient
    global ipg
    n = 0
    while n < loop:
        if yc:
            feedForward(data[n + start])
            #print(data[n + start])
            tlab = None
            if boo:
                tlab = (itlab[n + start] * 1)
            else:
                tlab = itlab[n + start]
        else:
            cleanGradients()
            tlab = None
            tlab = itlab
        x = 0
        cost = 0
        while x < len(output):
            cost = cost + ((output[x] - tlab[x]) ** 2)
            y = 0
            while y < len(layervalues[-1]):
                gradient[0][-1][x][y] = (sigprime(output[x]) * ( 2 * (output[x] - tlab[x]) ) * layervalues[-1][y]) + gradient[0][-1][x][y]
                #print(gradient[0][-1][x][y])
                y = y + 1
            gradient[1][-1][x] = (sigprime(output[x]) * ( 2 * (output[x] - tlab[x]) )) + gradient[1][-1][x]
            #print("")
            #print(gradient[1][-1][x])
            #print("")
            x = x + 1
        #time.sleep(1)
        #print("---------------------------")
        #print("")
        cost = cost / len(output)
        x = 0
        while x < (layers - 1): # finds hidden layer ∂C/∂h, i.e, the derivative of the cost with respect to the hidden layer
            y = 0
            while y < len(layervalues[-(x + 1)]):
                if x == 0: #used to be y == 0
                    z = 0
                    while z < len(output):
                        layerd[-1][y] = layerd[-1][y] + ( weight[-1][z][y] * sigprime(output[z]) * ( 2 * (output[z] - tlab[z]) ) )
                        z = z + 1
                else:
                    z = 0
                    while z < len(layervalues[-x]):
                        layerd[-(x + 1)][y] = layerd[-(x + 1)][y] + ( weight[-(x + 1)][z][y] * sigprime(layervalues[-x][z]) * layerd[-x][z] )
                        z = z + 1
                y = y + 1
            x = x + 1
        a = 0
        while a < (len(layervalues) - 1):
            x = 0
            while x < len(layervalues[-(a + 1)]):
                y = 0
                while y < len(layervalues[-(a + 2)]):
                    gradient[0][-(a + 2)][x][y] = (sigprime(layervalues[-(a + 1)][x]) * layerd[-(a + 1)][x] * layervalues[-(a + 2)][y]) + gradient[0][-(a + 2)][x][y]
                    y = y + 1
                gradient[1][-(a + 2)][x] = (sigprime(layervalues[-(a + 1)][x]) * layerd[-(a + 1)][x]) + gradient[1][-(a + 2)][x]
                x = x + 1
            a = a + 1
        x = 0
        while x < len(layervalues[0]):
            y = 0
            while y < len(ipg):
                gradient[0][0][x][y] = (sigprime(layervalues[0][x]) * layerd[0][x] * ipg[y]) + gradient[0][0][x][y]
                y = y + 1
            gradient[1][0][x] = (sigprime(layervalues[0][x]) * layerd[0][x]) + gradient[1][0][x]
            x = x + 1
        n = n + 1

def decendGradient(lr):
    global gradient
    global weight
    global bias
    global layers
    global indexes
    dlo = indexes
    layer = layers - 1
    learnr8(lr)
    a = 0
    while a < layer:
        bias[a] = np.subtract(bias[a], gradient[1][a])
        a = a + 1
    bias[-1] = np.subtract(bias[-1], gradient[-1][-1])
    x = 0
    while x < layer:
        y = 0
        while y < dlo[x]:
            if x == 0:
                weight[0][y]= np.subtract(weight[0][y], gradient[0][0][y])
            else:
                weight[x][y]= np.subtract(weight[x][y], gradient[0][x][y])
            y = y + 1
        x = x + 1
    z = 0
    while z < dlo[-1]:
        weight[-1][z] = np.subtract(weight[-1][z], gradient[0][-1][z])
        z = z + 1

def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    prefix = 'Training:'
    suffix = 'Complete'
    length = 50
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    if iteration == total:
        print()

def simpleGradientDecent(lr, c=1):
    global layervalues
    global output
    global gradient
    x = 0
    while x < len(output):
        y = 0
        while y < len(layervalues[-1]):
            #print(gradient[0][-1][x][y])
            gradient[0][-1][x][y] = gradient[0][-1][x][y] / c
            #print(gradient[0][-1][x][y])
            y = y + 1
        #time.sleep(60)
        gradient[1][-1][x] = gradient[1][-1][x] / c
        x = x + 1
    a = 0
    while a < (len(layervalues) - 1):
        x = 0
        while x < len(layervalues[-(a + 1)]):
            y = 0
            while y < len(layervalues[-(a + 2)]):
                gradient[0][-(a + 2)][x][y] = gradient[0][-(a + 2)][x][y] / c
                y = y + 1
            gradient[1][-(a + 2)][x] = gradient[1][-(a + 2)][x] / c
            x = x + 1
        a = a + 1
    x = 0
    while x < len(layervalues[0]):
        y = 0
        while y < len(ipg):
            gradient[0][0][x][y] = gradient[0][0][x][y] / c
            y = y + 1
        gradient[1][0][x] = gradient[1][0][x] / c
        x = x + 1
    decendGradient(lr)
    cleanGradients()

def cleanGradients():
    global layervalues
    global output
    global gradient
    x = 0
    while x < len(output):
        y = 0
        while y < len(layervalues[-1]):
            gradient[0][-1][x][y] = 0
            y = y + 1
        gradient[1][-1][x] = 0
        x = x + 1
    a = 0
    while a < (len(layervalues) - 1):
        x = 0
        while x < len(layervalues[-(a + 1)]):
            y = 0
            while y < len(layervalues[-(a + 2)]):
                gradient[0][-(a + 2)][x][y] = 0
                y = y + 1
            gradient[1][-(a + 2)][x] = 0
            x = x + 1
        a = a + 1
    x = 0
    while x < len(layervalues[0]):
        y = 0
        while y < len(ipg):
            gradient[0][0][x][y] = 0
            y = y + 1
        gradient[1][0][x] = 0
        x = x + 1

def learnr8(lr):
    global layervalues
    global output
    global gradient
    x = 0
    while x < len(output):
        y = 0
        while y < len(layervalues[-1]):
            gradient[0][-1][x][y] = gradient[0][-1][x][y] * lr
            y = y + 1
        gradient[1][-1][x] = gradient[1][-1][x] * lr
        x = x + 1
    a = 0
    while a < (len(layervalues) - 1):
        x = 0
        while x < len(layervalues[-(a + 1)]):
            y = 0
            while y < len(layervalues[-(a + 2)]):
                gradient[0][-(a + 2)][x][y] = gradient[0][-(a + 2)][x][y] * lr
                y = y + 1
            gradient[1][-(a + 2)][x] = gradient[1][-(a + 2)][x] * lr
            x = x + 1
        a = a + 1
    x = 0
    while x < len(layervalues[0]):
        y = 0
        while y < len(ipg):
            gradient[0][0][x][y] = gradient[0][0][x][y] * lr
            y = y + 1
        gradient[1][0][x] = gradient[1][0][x] * lr
        x = x + 1
