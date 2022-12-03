import random
import numpy as np
#import Tkinter
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math


def sign(signal):
    if signal > 0.0: 
        return 1
    else:
        return -1

def theta(signal,user_input):
    ans = signal
    for i in range(len(signal)):
        if user_input == 1:
            break
        elif user_input == 3:
            ans[i] = sign(signal[i])
        else:
            return np.tanh(signal)
    return ans

def derrivitive_of_theta(signal,user_input):
    if user_input == 1 or user_input == 3:
        return 1
    else:
        return 1 - np.multiply(signal,signal)

def deaugment(weights):
    num_rows, num_cols = weights.shape
    deaugmented_weights = np.zeros(shape = (num_rows -1,1))
    i = num_rows - 1
    while i > 0: #will ignore the 0th elment aka the augmented element
        deaugmented_weights[i-1] = weights[i]
        i -= 1
    return deaugmented_weights
#----------------------------------MAIN----------------------------------
m = int(input("specify m: "))
user_input = int(input("final layer transformations:\n1. identity\n2. tanh\n3. sign\nenter selection: "))

#specifying the architecture of the network
DL = [2,m,1]

#setting the number of layers 
L = len(DL) - 1

#initializing the first layer 
x_mat = np.ones(shape = (2,1))
x_mat[0][0] = 1.0
x_mat[1][0] = 2.0

#initializing the y values for the traning set
y_mat = np.ones(shape = (1,1))
y_mat[0][0] = 1.0

#initlizlize the weights for each layer
weights = []
i = 0
while i < L:
    w0 = np.ones(shape = (DL[i]+1,DL[i+1]))
    for j in range(DL[i]+1): #for every column
        for k in range(DL[i+1]): #for every Row
            w0[j][k] = 0.25 # initlaize it with all weights == 0.25
    weights.append(w0)
    i += 1

#initialize outputs of each layer
outputs = []
#initalize outputs with the inputs to the first layer.
outputs.append(x_mat)

#initlizize the array of signals generated for each layer
signals = []
learning_rate = 0.1
t = 0
while t < 1:
    print("---------forward propogation---------")
    print("layer 0:")
    print(outputs[0])
    for l in range (0,L):
        print()
        print("layer ",l+1,":")
        signal = np.matmul(np.transpose(weights[l]), np.insert(outputs[l], 0, 1,axis = 0) )
        print("Input:")
        print(signal)
        signals.append(signal)
        if l != L-1: #hidden layer
            signal = theta(signal, 0)
        else: # final layer swtich to using user inputed transformation
            signal = theta(signal, user_input) # retruns a numpyarray of values  
        outputs.append(signal)  # do not add a augmented one on the final output
        print()
        print("output:")
        #add an augmented one for costmetic purposes

        tmp = np.ones(shape = (len(signal)+1,1))
        for i in range(len(signal)+1):
            if i == 0:
                tmp[i] = np.ones(shape=(1,))
            else:
                tmp[i] = signal[i-1]
        print(tmp)
        
    print()
    print("---------backward propogation---------")
    deltas = []
    for i in range(L):
        deltas.append(0) 
    if user_input == 1 or user_input == 3:
        deltas[L-1] = 2 * (outputs[L] - y_mat) * 1
    else:
        deltas[L-1] = 2 * (outputs[L] - y_mat) * (1 - math.pow(outputs[L],2))
    #print()
    #print("deltas")
    #print(deltas)
    #print()
    gradients = []
    gradients.append( np.dot(0.25,np.matmul(np.insert(outputs[L-1],0,1,axis = 0), np.transpose(deltas[L-1])) ))
    l = L - 2
    while l > -1:
        deltas[l] = np.multiply(1 - np.multiply(outputs[l+1],outputs[l+1]),np.matmul(deaugment(weights[l+1]),deltas[l+1]))
        gradients.insert(0,np.dot(0.25,np.matmul(np.insert(outputs[l],0,1,axis = 0), np.transpose(deltas[l]))) )
        l -= 1
    
    print ("gradients:")
    for g in gradients:
        print(g)
        print("")
    t += 1
