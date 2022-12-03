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
#print("length of DL")
#print(len(DL))
L = len(DL) - 1
x_mat = np.ones(shape = (2,1))
x_mat[0][0] = 1.0
x_mat[1][0] = 2.0


y_mat = np.ones(shape = (1,1))
y_mat[0][0] = 1.0
#going from 0 .. L -1 L elements
weights = []
i = 0
while i < L:
    w0 = np.ones(shape = (DL[i]+1,DL[i+1]))
    for j in range(DL[i]+1): #for every colum
        for k in range(DL[i+1]): #for every Row
            w0[j][k] = 0.25
    weights.append(w0)
    i += 1
#print weights
'''
calulaitng gradient numerically 

create a weights array all inilized to 0.25
for each layer
    create a gradient numpy array same dimentions as the  
    create 
    for each element in the weight array
        add + h to the elment
        run forward propagation
        produce a xL
        plug xl into ein funciton
        get a ein for this value

        add a -h to the element 
        run a forward propagation
        produce a xl
        plug that xl into ein function from hw
        get a ein for this value

        use the 2 ein's to calulate the value for the formula from piazza

        plug that into corresponding spot in gradients 
'''
gradients = []
for i in range(L):
    g0 = np.zeros(shape = (DL[i]+1,DL[i+1]))
    for j in range(DL[i]+1): #for every colum
        for k in range(DL[i+1]): #for every Row
            weights[i][j][k] = 0.2501
            outputs = []
            outputs.append(x_mat)
            
            for l in range (0,L):# - exlusive on the last one
                #print("layer: {}".format(l))
                signal = np.matmul(np.transpose(weights[l]), np.insert(outputs[l], 0, 1,axis = 0) )#dont need to transpose since i store my matrix in a transposed state.
                if l != L-1: #hidden layer
                    signal = theta(signal, 0)
                else: # final layer swtich to using user inputed transformation
                    signal = theta(signal, user_input) # retruns a numpyarray of values  
                outputs.append(signal)  # do not add a augmented one on the final output
                #outputs.append(np.insert(signal, 0, 1))
            #print outputs
            xl1 = outputs[L]

            weights[i][j][k] = 0.2499
            outputs = []
            outputs.append(x_mat)
            #print("---------forward propogation---------")
            for l in range (0,L):# - exlusive on the last one
                #print("layer: {}".format(l))
                signal = np.matmul(np.transpose(weights[l]), np.insert(outputs[l], 0, 1,axis = 0) )#dont need to transpose since i store my matrix in a transposed state.
                if l != L: #hidden layer
                    signal = theta(signal, 0)
                else: # final layer swtich to using user inputed transformation
                    signal = theta(signal, user_input) # retruns a numpyarray of values  
                outputs.append(signal)  # do not add a augmented one on the final output
                #outputs.append(np.insert(signal, 0, 1)) 
            xl2 = outputs[L]

            #print("xl1:")
            #print xl1
            ein1 = xl1 - y_mat
            ein1 = (ein1 * ein1)/4
            #print "ein1:"
            #print ein1

            #print("xl2")
            #print(xl2)
            ein2 = xl2 - y_mat
            ein2 = ein2 * ein2
            ein2 = 0.25 * ein2
            #print "ein2:"
            #print ein2

            #print "gradient"
            g0[j][k] = (ein1-ein2)/0.0002
            #print  (ein1-ein2)/0.0002
            #print ""

            weights[i][j][k] = 0.25
    gradients.append(g0)
            
print "gradients:"
for g in gradients:
    print g
    print ""