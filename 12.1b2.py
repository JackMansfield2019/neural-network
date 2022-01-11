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
    #print(signal)
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
m = input("specify m: ")
user_input = input("final layer transformations:\n1. identity\n2. tanh\n3. sign\nenter selection: ")
DL = [2,m,1]
print("length of DL")
print(len(DL))
L = len(DL) - 1
x_mat = np.ones(shape = (2,1))
x_mat[0][0] = 1.0
x_mat[1][0] = 2.0

print(x_mat)
y_mat = np.ones(shape = (1,1))
y_mat[0][0] = 1.0
#going from 0 .. L -1 L elements
weights1 = []
i = 0
while i < L:
    w0 = np.ones(shape = (DL[i]+1,DL[i+1]))
    for j in range(DL[i]+1): #for every colum
        for k in range(DL[i+1]): #for every Row
            w0[j][k] = 0.2501
    weights1.append(w0)
    i += 1

weights2 = []
i = 0
while i < L:
    w0 = np.ones(shape = (DL[i]+1,DL[i+1]))
    for j in range(DL[i]+1): #for every colum
        for k in range(DL[i+1]): #for every Row
            w0[j][k] = 0.2499
    weights2.append(w0)
    i += 1

outputs = []
outputs.append(x_mat)
signals = []

for l in range (0,L):# - exlusive on the last one
    #print("layer: {}".format(l))
    signal = np.matmul(np.transpose(weights1[l]), np.insert(outputs[l], 0, 1,axis = 0) )#dont need to transpose since i store my matrix in a transposed state.
    signals.append(signal)
    if l != L: #hidden layer
        signal = theta(signal, 0)
    else: # final layer swtich to using user inputed transformation
        signal = theta(signal, user_input) # retruns a numpyarray of values  
    outputs.append(signal)  # do not add a augmented one on the final output
    #outputs.append(np.insert(signal, 0, 1)) 
xl1 = outputs[2]

for l in range (0,L):# - exlusive on the last one
    #print("layer: {}".format(l))
    signal = np.matmul(np.transpose(weights2[l]), np.insert(outputs[l], 0, 1,axis = 0) )#dont need to transpose since i store my matrix in a transposed state.
    signals.append(signal)
    if l != L: #hidden layer
        signal = theta(signal, 0)
    else: # final layer swtich to using user inputed transformation
        signal = theta(signal, user_input) # retruns a numpyarray of values  
    outputs.append(signal)  # do not add a augmented one on the final output
    #outputs.append(np.insert(signal, 0, 1)) 
xl2 = outputs[2]


print("xl1:")
print xl1
ein1 = xl1 - y_mat
ein1 = 0.25 * (ein1 * ein1)
print "ein1:"
print ein1
print ""     

print("xl2")
print(xl2)
ein2 = xl2 - y_mat
ein2 = 0.25 * (ein2 * ein2)
print "ein2:"
print ein2
print "" 

print "gradient"
print  (ein1-ein2)/0.0002
print ""


'''
learning_rate = 0.1
t = 0
#dont save with the augmented one!!!!
while t < 1:
    print("---------forward propogation---------")
    for l in range (0,L):# - exlusive on the last one
        print("layer: {}".format(l))
        signal = np.matmul(np.transpose(weights[l]), np.insert(outputs[l], 0, 1,axis = 0) )#dont need to transpose since i store my matrix in a transposed state.
        print signal
        signals.append(signal)
        if l != L: #hidden layer
            signal = theta(signal, 0)
        else: # final layer swtich to using user inputed transformation
            signal = theta(signal, user_input) # retruns a numpyarray of values  
        outputs.append(signal)  # do not add a augmented one on the final output
        #outputs.append(np.insert(signal, 0, 1))   

    
    print("---------backward propogation---------")
    print("outputs:")
    print(outputs)
    print("signals:")
    print signals
    #deltas = np.zeros(shape = (L,1))#0,1,2 ... L-1
    h = 0.0001
    gradients = []
    for w in range(len(weights)):
        h_mat = np.ones(shape = weights[w].shape)
        h_mat = np.multiply(h_mat,0.0001)
        #print(h_mat_pos)
        if w != L: #hidden layer
            xl = np.insert(theta( np.matmul( np.transpose( weights[w] + h_mat  ) , np.insert(outputs[w], 0, 1,axis = 0) ), 0 ),0,1,axis = 0)
        else:
            xl = np.insert(theta( np.matmul( np.transpose( weights[w] + h_mat  ) , np.insert(outputs[w], 0, 1,axis = 0) ), user_input),0,1,axis= 0)
        print("xl1:")
        print xl
        ein1 = xl - y_mat
        ein1 = (ein1 * ein1)/4
        print "ein1:"
        print ein1
        
        if w != L: #hidden layer
            xl = np.insert(theta( np.matmul( np.transpose( weights[w] - h_mat ) , np.insert(outputs[w], 0, 1,axis = 0) ), 0 ),0,1,axis = 0)
        else:
            xl = np.insert(theta( np.matmul( np.transpose( weights[w] - h_mat ) , np.insert(outputs[w], 0, 1,axis = 0) ), user_input),0,1,axis= 0)
        print("xl2")
        print(xl)
        ein2 = xl - y_mat
        ein2 = ein2 * ein2
        ein2 = 0.25 * ein2
        print "ein2:"
        print ein2
        print "gradient"
        print  (ein1-ein2)/0.0002
        print ""
        #gradients.append()
    t += 1
'''