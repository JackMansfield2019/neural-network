import random
import numpy as np
#import Tkinter
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
import time

def format(data):
    image = np.arange(256).reshape(16, 16)
    image = image.astype('float')
    for i in range(16):
        for j in range(16):
            image[i][j] = data[(16*i)+j]
    return image

def plot_image(data):
    image = np.arange(256).reshape(16, 16)
    image = image.astype('float')
    for i in range(16):
        for j in range(16):
            image[i][j] = ((data[(16*i)+j] + 1)/2)
    plt.imshow(image)
    plt.savefig("hello.pdf")
    plt.show()

def get_intensity(data):
    intensity = 0.0
    for i in data:
        intensity += i + 1
    return intensity/256

def get_symmetry(data):
    ans = 0 
    for i in range(8):
        for j in range(16):
            data[j*16 + i] - data[(j+1)*16 - (i + 1)]
            ans += abs(data[j*16 + i] - data[(j+1)*16 - (i + 1)])
    for i in range(16):
        for j in range(8):

            ans += abs(data[j*16 + i] - data[(15 - j)*16 - i])
    return ans

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

def create_data(filename1,filename2):
    #open file
    with open(filename1, 'r') as f1:
        lines = f1.readlines()
    with open(filename2, 'r') as f2:
        lines += f2.readlines()

    #suffle the file so everything is random
    random.shuffle(lines)

    ones_x_test = []
    ones_y_test = []
    other_x_test = []
    other_y_test = []
    # READ FILE
    # FILTER INTO 1'S AND 5'S
    for line in lines:
        
        if line[0] == '1':
            x = [float(i) for i in line.split()]
            x = x[1:]
            ones_x_test.append(get_symmetry(x))
            ones_y_test.append(get_intensity(x))
            
        else:
            x = [float(i) for i in line.split()]
            x = x[1:]
            other_x_test.append(get_symmetry(x))
            other_y_test.append(get_intensity(x))
    
    max_symmetry = max(max(ones_x_test),max(other_x_test))
    max_intensity =max(max(ones_y_test),max(other_y_test))
    for i in range(len(ones_x_test)):
        ones_x_test[i] =  (2*(ones_x_test[i] / max_symmetry))-1
        ones_y_test[i] = (2*(ones_y_test[i] / max_intensity))-1
    for i in range(len(other_x_test)):
        other_x_test[i] =  (2*(other_x_test[i] / max_symmetry))-1
        other_y_test[i] = (2*(other_y_test[i] / max_intensity))-1

    ones_x_train = []
    ones_y_train = []
    other_x_train = []
    other_y_train = []

    for i in range(300):
        rand = random.randint(0, len(ones_x_test) + len(other_x_test)-1)
        if rand < len(ones_x_test):
            ones_x_train.append(ones_x_test.pop(rand))
            ones_y_train.append(ones_y_test.pop(rand))
        else:
            other_x_train.append(other_x_test.pop(rand- len(ones_x_test)))
            other_y_train.append(other_y_test.pop(rand - len(ones_x_test)))

    x_mat_test = np.ones(shape = (len(ones_x_test)+len(other_x_test),2)) 
    y_mat_test = np.ones(shape = (len(ones_x_test)+len(other_x_test),1))  
    for i in range(len(ones_x_test)+len(other_x_test)):
        if i < len(ones_x_train):
            x_mat_test[i][0] = ones_x_test[i]
            x_mat_test[i][1] = ones_y_test[i]
            y_mat_test[i] = 1.0
        else:
            x_mat_test[i][0] = other_x_test[i - len(ones_x_test)]
            x_mat_test[i][1] = other_y_test[i - len(ones_x_test)]
            y_mat_test[i] = -1.0
    
    x_mat_train = np.ones(shape = (len(ones_x_train)+len(other_x_train),2)) 
    y_mat_train = np.ones(shape = (len(ones_x_train)+len(other_x_train),1))  
    for i in range(len(ones_x_train)+len(other_x_train)):
        if i < len(ones_x_train):
            x_mat_train[i][0] = ones_x_train[i]
            x_mat_train[i][1] = ones_y_train[i]
            y_mat_train[i] = 1.0
        else:
            x_mat_train[i][0] = other_x_train[i - len(ones_x_train)]
            x_mat_train[i][1] = other_y_train[i - len(ones_x_train)]
            y_mat_train[i] = -1.0
    
    return (x_mat_test,y_mat_test,x_mat_train,y_mat_train,ones_x_test,ones_y_test,other_x_test,other_y_test,ones_x_train,ones_y_train,other_x_train,other_y_train)

    
#----------------------------------MAIN----------------------------------
x_mat,y_mat,x_mat_train,y_mat_train,ones_x,ones_y,other_x,other_y,ones_x_train,ones_y_train,other_x_train,other_y_train = create_data('ZipDigits.train','ZipDigits.test')
print("finished create data")


'''
for every iteration
when the validatoin et error goes up stop traning.

for every iteration
    find error on the vaidaton set
        get gradients for the 250
        construct v(t) 
        forward propagate on the 50 
        get a xL
        calulate the ein_train
        if ein_train > old_ein_train - the past 5 have increased or if the increase is super large
            break
        if ein_train < old _ein_train
        store optimal weights wit hrespect to train_ein 

'''
m = input("specify m: ")
user_input = input("final layer transformations:\n1. identity\n2. tanh\n3. sign\nenter selection: ")
DL = [2,m,1]
#print("length of DL")
#print(len(DL))
L = len(DL) - 1


weights = []
i = 0
while i < L:
    w0 = np.ones(shape = (DL[i]+1,DL[i+1]))
    for j in range(DL[i]+1): #for every colum
        for k in range(DL[i+1]): #for every Row
            w0[j][k] = random.uniform(-1, 1)
    weights.append(w0)
    i += 1



R =  1.0
alpha = random.uniform(1.05, 1.1)
beta = 0.8#random.uniform(0.5, 0.8)
t = 0
og_g = []
for l in range(0,L):
        og_g.append( np.zeros(weights[l].shape) )

og_deltas = []
for i in range(L):
    og_deltas.append(0) 

eins = []
ts = []

total_time = 0.0
old_ein = 9999.9
while t < 10000:
    start_time = time.time()
    print("-------t = {}-------".format(t))
    
    ein = 0.0
    g = og_g
    for x in range(300):
        #print("---------forward propogation---------")
        outputs = [np.expand_dims(x_mat_train[x], axis=1)]
        current_y = y_mat_train[x]

        for l in range (0,L):
            #print("layer: {}".format(l))
            signal = np.dot(np.transpose(weights[l]), np.concatenate(([[1]] , outputs[l]), axis = 0) )
            #switch transformation for final layer
            if l != L -1: 
                signal = theta(signal, 0)
            else: 
                signal = theta(signal, user_input)  
            outputs.append(signal)
        


        #print("---------backward propogation---------")
        #switch transformation of final layer
        deltas = og_deltas
        if user_input == 1 or user_input == 3:
            deltas[L-1] = 2 * (outputs[L] - current_y) * 1
        else:
            deltas[L-1] = 2 * (outputs[L] - current_y) * (1 - math.pow(outputs[L],2))
        
        gradients = [ np.dot(0.25,np.dot(np.concatenate(([[1]] , outputs[L-1]), axis = 0), np.transpose(deltas[L-1]))) ]
        l = L - 2
        while l > -1:
            #print("l = {}".format(l))
            deltas[l] = np.multiply(1 - np.multiply(outputs[l+1],outputs[l+1]),np.dot(deaugment(weights[l+1]),deltas[l+1]))
            gradients.insert(0,np.dot(0.25,np.dot(np.concatenate(([[1]] , outputs[l]), axis = 0), np.transpose(deltas[l]))) )
            l -= 1

        #update gradients for this time step
        g[0] = g[0] + np.multiply(0.00333333333,gradients[0])
        g[1] = g[1] + np.multiply(0.00333333333,gradients[1])
        '''
        print "gradients:"
        for g in gradients:
            print g
            print ""
        '''

    for x in range(300):
        outputs2 = [np.expand_dims(x_mat_train[x], axis=1)]
        current_y = y_mat_train[x]
        Vt = [np.multiply(R,np.multiply(-1,g[0])),np.multiply(R,np.multiply(-1,g[1])) ]
        #print("---------forward propogation---------")
        l = 0
        for l in range (0,L):
            #print("layer: {}".format(l))
            signal = np.dot(np.transpose(weights[l] + Vt[l]), np.concatenate(([[1]] , outputs2[l]), axis = 0) )
            #switch transformation for final layer
            if l != L -1: #hidden layers
                signal = theta(signal, 0)
            else: #final layer
                signal = theta(signal, user_input) # retruns a numpyarray of values  
            outputs2.append(signal)
            
        ein += (0.00083333333) * math.pow((outputs2[L]-current_y),2)

    '''
    print R
    print("old: {}".format(old_ein))
    print("new: {}".format(ein))
    print ""
    '''
    
    #uncomment for part 2B
    
    sum1 = 0.0
    for l in range(len(weights)):
        for m in range(len(weights[l])):
            for n in range(len(weights[l][m])):
                sum1 += math.pow(weights[l][m][n],2)

    ein = ein + (0.01/300)*sum1
    
    if ein < old_ein:
        #weights = weights + np.multiply(R,np.multiply(-1,g))
        for i in range(0,L):
            weights[i] = weights[i] + np.multiply(R,np.multiply(-1,g[i]))
            #print weights[i]
        
        R = alpha * R
        old_ein = ein
    else:
        R = beta * R
    
    eins.append(math.log10(old_ein))
    ts.append(t)
    total_time += time.time() - start_time
    t += 1

print("--- %s seconds ---" % (total_time/1000))
print("ERROR: {}".format(old_ein))
'''
print ""
print "weights:"
for w in weights:
    print w
    print ""
'''
plt.xlabel('iterations')
plt.ylabel('Error')

plt.plot(ts, eins, 'bo', label='E_in',markersize=2.5)
plt.xlabel('iterations')
plt.ylabel('E_in')
plt.savefig("12.2.1.pdf")
plt.show()



'''
todo list

make it efficient enouguh for 2,000,000 iterations - idk man

figure out the log when printing ein vs iterations

A
for 2,000,000 iterations
print the ein vs iterations and the decsion boundry

B
uncomment the weight decay code
    for 2,000,000 iterations
    print the ein vs iterations and the decsion boundry

C
learn + write early stopping
before we begin
take 50 out of our traning set, just the last 50 since our traning set is randomized already
as we train


learn SVM's

begin wrtting SVM's using the cvx opt package
'''

p_x = []
p_y = []
n_x = []
n_y = []
x = -1
y = -1
data_point = np.ones(shape = (2,1))
while x < 1:
    while y < 1:
        data_point[0][0] = x
        data_point[1][0] = y

        outputs = [data_point]
        for l in range (0,L):
            #print("layer: {}".format(l))
            signal = np.dot(np.transpose(weights[l]), np.concatenate(([[1]] , outputs[l]), axis = 0) )
            #switch transformation for final layer
            if l != L -1: 
                signal = theta(signal, 0)
            else: 
                signal = theta(signal, user_input)  
            outputs.append(signal)
        #print outputs[L]

        if outputs[L] > 0:
            p_x.append(x)
            p_y.append(y)
        else:
            n_x.append(x)
            n_y.append(y)
        y+=0.02
    y = -1
    x+=0.02
blues, = plt.plot(p_x, p_y, 'bo', markersize=1)
reds, = plt.plot(n_x, n_y, 'ro', markersize=1)
ones, = plt.plot(ones_x_train,ones_y_train, 'bo', label='1', markersize=5)
other, = plt.plot(other_x_train, other_y_train, 'rx', label='other',markersize=5)
plt.xlabel('Symmetry')
plt.ylabel('Intensity')
plt.axis([-1, 1, -1, 1])
plt.savefig("12.2.1boundry.pdf")
plt.show()