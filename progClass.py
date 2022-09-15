import numpy as np

np.random.seed(0)#the same random value will be generated every time
X=[[1, 2, 3, 2.51],
        [2.0,5.0,-1.0,2.0],
        [-1.5,2.7,3.3,-0.8]]


class Layer_Dense:
 def __init__(self,n_inputs,n_neurons):
  self.weights=0.10*np.random.randn(n_inputs,n_neurons)#no need to transpose as we have got it fixed here
  self.biases=np.zeros((1, n_neurons))

 def forward(self,inputs):
  self.output=np.dot(inputs, self.weights)+self.biases

class Activation_Relu:
  def forward(self,inputs):
   self.output=np.maximum(0,inputs)//will replace the elements with 0 which r -ve

layer1=Layer_Dense(4,5)
layer2=Layer_Dense(5,2)

layer1.forward(X)
print("LAYER 1 \n")
print(layer1.output)

print("\n\n\nLAYER 2 \n")
layer2.forward(layer1.output)
print(layer2.output)
