# Program Overview
**Simple_Neural_Network.py:** Implementation of a simple neural network and gradient descent using backpropagation for a neural network architecture [2,m,1]

**Numerical_Gradient_Calulation.py:** Program to numerically obtain gradients from the simple nueral network, used to check that the Network is working correctly

**Handwriting_Neural_Network.py:** Program using a Neural Network Model to create a classifier to seperate handwritten "1" digits from "5" digits:

**Handwriting_Neural_Network_With_Validation.py:** Program Implmenting the previous Neural Network Model with a Validation set of size 50 to test the model with.

# Simple_Neural_Network

A program to implement neural networks and gradient descent using backpropagation for a neural network architecture [2,m,1] with 
2 input nodes, $m$ -hidden unit nodes, and 1 output sigmoidal node.  

• Allows user to specfify m hidden nodes in 2nd layer

• Allows the user to pick between identity: θ(s) = s, tanh: θ(s) = tanh(s), and sign: θ(s) = sign(x) for the output node transformation in the last layer.

• All the hidden node transformations are tanh, θ(s) = tanh(s) for hidden-layer nodes

• All inital weights set to 0.25:

<p align="center">
  <img src="https://user-images.githubusercontent.com/25088039/204698089-8312780a-10a7-4a8e-8db4-62e1aedb44b0.JPG?row=true" width="300" height="300">
</p>
<p align="center">
  Figure 1: 2-input, 2-hidden layer, 1-output Neural network 
</p>

### Usage 
    Python3 Simple_Neural_Network.py
    
    
# Results


the gradient of Ein(w) using the backpropagation algorithm using a network where m = 2, all the weights equal to 0.25 and a data set with 1 point: x1 =[1,2]; y = 1

Output with identity transformation: θ(s) = s:

<p align="center"> 
  
<img src="https://user-images.githubusercontent.com/25088039/204704254-3379137c-cc6a-43fd-b6d6-57e96c56f828.JPG" width="300" height="440">
</p>


Output with tanh transformation: θ(s) = tanh(s):

<p align="center">
<img src="https://user-images.githubusercontent.com/25088039/204704453-439c1dcb-ef90-420e-8607-208815b6358c.JPG?row=true" width="300" height="440">
</p>

# Numerical_Gradient_Calulation

Program to obtain previous gradients numerically by peturbing each weight in turn by 0.0001.

Output with identity transformation: θ(s) = s:
<p align="center">
<img src="https://user-images.githubusercontent.com/25088039/204705993-6327dbf6-f003-4c9a-8704-02aa81262433.JPG?row=true">
</p>
Output with tanh transformation: θ(s) = tanh(s):
<p align="center">
<img src="https://user-images.githubusercontent.com/25088039/204705994-1a6971df-61df-450d-a26b-9602ebe68fe9.JPG?row=true">
</p>

These results are almost identical to our previous results verifing my back proogation gradient calculation.
### Usage 
    Python3 Numerical_Gradient_Calulation.py

#  Neural Network for Digits
Using the Neural Network Model from the previous part create a classifier to seperate the 1 digits from the 5 digits as shown below:

<p align="center">
  <img src="https://user-images.githubusercontent.com/25088039/205457670-b0e80ee7-b0ee-46c3-aa6c-1b51779be126.JPG?row=true" width="617" height="134">
</p>
<p align="center">
  Figure 2: 1 and 5
</p>
I chose to use the features of intensity and symmetry, where symmetry in this case means the whether the image is verticaly symmetrical. Let $f(i,j)$ denotes the grayscale values from $-1$ to $1$ for pixel $(i,j)$ as given. And $i,j$ ranges from $1$ to $16$.

Then the intensity is defined as: $\displaystyle I_{avg} = \frac{1}{256}\sum_{i = 1}^{16} \sum_{i = 1}^{16} f(i,j)$

And the symmetry is defined as:  $\displaystyle I_{sym} = \frac{1}{256}\times\frac{1}{256}\sum_{i = 1}^{16} \sum_{i = 1}^{16} |f(i,j)-f(17-i,j)|$

Using the Neural Network Model developed in the previous part to create this classifier, with $m=10$, and all active functions being sigmoids. I use Intensity and Asymmetry as the $X1$ and $X2$ input, with $N=300$. I then compute $$E_{in}(w) = \frac{1}{300}\sum^{300}_{n=1}(h(x_n,w)-y_n)^2$$ using a max iteration of $1\times 10^{6}$ and $S(x) = x$ , the decision boundary at  our max iteration and the iteration errors along the way are shown as follows:

<p align="center">
  <img src="https://user-images.githubusercontent.com/25088039/205458150-bd4b759f-8690-4fed-8958-9522906394e4.JPG?row=true" width="346" height="300">
  <img src="https://user-images.githubusercontent.com/25088039/205458154-c5d8d462-91c0-43de-8aeb-07c99a99c840.JPG?row=true" width="375" height="300">
</p>
 <p align="center">
  Figure 3: $S(x) = x$, Max Iteration $= 1\times 10^6$ 
</p>

### Usage 
    Python Handwriting_Neural_Network.py

# Neural Network with Validation

I then separated my set of 300 data points into a validation set of size 50 and training set of size 250, so: $$E_{val}(w) = \frac{1}{50}\sum_{n=1}^{50}(\hat{y_n} \neq y_n)$$ I then counted the number of misclassified data within the 50 validation dataset. The minimum $E_{val}(w)$ occurs at iteration $423$ before it goes up again (shown in red curve $Figure$  $4(b)$ ), in which stopped the iteration, the resulting boundary and the Validation error are shown as follows:

<p align="center">
   <img src="https://user-images.githubusercontent.com/25088039/205403361-7f9a2a13-d45b-40ba-ae1d-3e4c95e2f186.JPG?row=true" width="358" height="300">
   <img src="https://user-images.githubusercontent.com/25088039/205403379-ffa21772-eb9a-474c-963a-9fab6de1ff91.JPG?row=true" width="400" height="300">
</p>
<p align="center">
  Figure 4: $S(x) = x$, Stopping Iteration at iteration $423$
</p>

### Usage 
    Python Handwriting_Neural_Network_With_Validation.py
