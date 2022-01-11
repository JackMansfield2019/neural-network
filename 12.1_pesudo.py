'''
For the output node
transformation in the last layer, you should allow the user to pick between identity, θ(s) = s, tanh,
θ(s) = tanh(s) and sign, θ(s) = sign(x).

Set the architecture to [2, m, 1]: 2 inputs (d(0) = 2), m hidden units (d(1) = m), and 1 output node (L = 2).

1. promt user for m
2. promt user to pick options 1 - 3
    1. theta(s) = s
    2. theta(s) = tanh(s)
    3. theta(s) = sign(s)
3. create data d(0): 
    d(0) = 2
    x_1 = [1,2]
4. create DL:
    DL(0) = 2
    DL(1) = m
    DL(2) = 1
5. create array of weights
    L= 1
        rows   colums
    1. for L =1 ... L = L  therefore there should be L: (D(1) x D(L-1) +1) matricies
        [(D(1) x 2 + 1)
         (3 x 1)]
    2. make default value = 0.25
6. create array of outputs
    same dimentions as the weights
7. for t < threshold
    for l in range (1,L +1): - exlusive on the last one
        #---------forward propogation---------
        signal = np.matmul(weights[l],output[l-1]) #dont need to transpose since i store my matrix in a transposed state.
        if l != L+1: #hidden layer
            signal = theta(signal, 0)
        else: # final layer swtich to using user inputed transformation
            signal = theta(signal, user_input) # retruns a numpyarray of values    
        output[l] = np.insert(signal, 0, 1)            
        #signal is a vector of the signals produced for each perceptron
        #we add one to that vector.
        #---------backward propogation---------
        gradients  = 
        weights[l] = weights[l] + np.dot(learning_rate, )
'''