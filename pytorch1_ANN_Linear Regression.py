import torch
t1=torch.tensor(.4)
print("t1", t1)
print(type(t1))
print(t1.dtype)
#comments 
#Pytorch is an open-source library for deep learning.
#it is a framework for building deep learning models.
#we can get more information about the pytorch library from the official website:
#www.pytorch.org/
#huggingface.co/transformers/ is for learning the NLP

#Below code is for Pytorch for ANN   
#Implementation of Linear Regression using PyTorch
import numpy as np

#imput data defining
input=np.array([[12,55,77],
                [34,54,43],
                [34,66,21],
                [92,72,48],
                [12,99,12]],dtype="float32")
print(input)

#target data
targets=np.array([[55,77],
                    [34,63],
                    [23,67],
                    [89,45],
                    [67,87]],dtype="float32")


#convert input to tensor
inputs=torch.from_numpy(input)
targets=torch.from_numpy(targets)

print("inputs",inputs)
print("targets",targets)


#initialize weights
w=torch.randn(2,3,requires_grad=True) #initializing weights as 2 x 3 matrix based on number of target columns X number of input columns
                   #of input columns.
b=torch.randn(2,requires_grad=True) #initializing bias as 2 x 1 matrix based on number of target columns X number of input columns

print("w",w)
print("b",b)


def model(inputs,w,b):
    return torch.matmul(inputs,w.t()) + b

prediction=model(inputs,w,b)
print("prediction",prediction)
print("actual",targets)


def model2(inputs,w,b):
    return inputs @ w.t()+ b #@ is the matrix multiplication operator or dot product operator



prediction=model2(inputs,w,b)
print("prediction",prediction)
print("actual",targets)

#the predicted values are different from the actual target values
#so we need to calculate the loss function
def MSE(prediction,targets):
    dif=prediction-targets
    return torch.sum(dif**2)/dif.numel()

loss=MSE(prediction,targets)
print("loss",loss)

loss2=torch.mean((prediction-targets)**2)    #both loss and loss2 are the same

print("loss2",loss2) 

loss.backward()
print("w",w)
print("w-grad",w.grad)


with torch.no_grad():   #no_grad is a context manager indicating that the gradient should 
                            #not be calculated


    w-=w.grad* 1e-5  #1e-5 is a small value to avoid division by zero. 
                     # resetting the w value to slightly less than the previous
    b-=b.grad* 1e-5  # resetting the b value to slightly less than the previous

print("w",w)

#w(new)=w(old)-learning_rate(dx/dw)

preds=model(inputs,w,b)
loss=MSE(preds,targets)
print("loss",loss)
print("preds",preds)
print("actual",targets)
n=900
min_loss=500
for i in range(n):  #n is the number of iterations

    preds=model(inputs,w,b) # get the prediction y=w*inputs+b

    loss=MSE(preds,targets) #calculate the loss difference between the prediction and the 
                            #actual target values
    loss.backward()  #calculate the gradient of the loss function

    
    with torch.no_grad():
        w-=w.grad* 1e-5 #resetting the w value to slightly less than the previous
        b-=b.grad* 1e-5 #resetting the b value to slightly less than the previous

        
        w.grad.zero_() # initialize gradient to zero

        b.grad.zero_() # initialize gradient to zero

    if loss < min_loss:
        min_loss=loss
        iter=i 

    print(f"Epochs:{i}/{n}----Loss:{loss}") #what is epochs? epochs is the number of
    #                                         # interations that have been completed to find the
                                            # optimal value of w and b. So that the predicted
                                            # values become closer to the actual target values.
                                            # by getting minimum loss, we can get the optimal
                                            # value of w and b.
                                            # for each epoch or interation the loss is getting
                                            # reduced.


print("preds",preds) # Now the predicted values are close to the actual target values.

print("actual",targets)
print("min_loss",min_loss)
print("iter",iter)

#Using Torch Builtin Functions

import torch.nn as nn #for neural networks
#input(temperature,Humidity,Rainfall)
imputs=np.array([[12,55,77],
                 [24,55,77],
                 [32,43,90],
                 [34,54,43],
                 [74,90,25],
                 [72,54,79],
                 [13,97,65],
                 [97,74,64],
                 [79,13,34],
                 [97,54,43],
                 [25,65,43],
                 [65,54,79],
                 [43,72,74],
                 [64,25,90],
                 [24,64,97],
                 [75,13,72]],dtype="float32")

#target(Mango_crop,Apple_crop)

targets=np.array([[55,77],
                  [98,23],
                  [64,32],
                  [67,92],
                  [23,98],
                  [54,54],
                  [22,43],
                  [75,58],
                  [89,85],
                  [34,54] ,
                  [25,43],
                  [75,33],
                  [34,68],
                  [45,87],
                  [54,67],
                  [67,23] ],dtype="float32")
print("input",imputs)
print("target",targets)

#convert input to tensor
inputs=torch.from_numpy(imputs)

targets=torch.from_numpy(targets)

print("inputs",inputs)
print("targets",targets) 
print('TARGET_SHAPE',targets.shape)

#dataloaders
from torch.utils.data import TensorDataset,DataLoader
train_dataset=TensorDataset(inputs,targets)

print("train_dataset",train_dataset) #it creates the object of the class TensorDataset

print(train_dataset[0])
print(train_dataset[0:3])

from torch.utils.data import DataLoader

batch_size=5
train_dataset=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)   
                                    #The DataLoader class is used to create a
                                    # DataLoader object that can be used to iterate over
                                    # the data in the dataset.





for x, y in train_dataset: #Since we have givne batch_size=5,
                            # each time we will get 5 number of data for each
                            #  input and target dataset
    print("x",x)            # this way we can train the model on batch data wise 
                            #(here on 5 inputs and 5 targets)
                            # Its generates data randomly each time
                            # so that we can train the model more robustly.
                            # During data loader it is not recommended to use the seeding
                            # We want the model to be trained on new new data pointseach time
    print("y",y)
    break    


# Now build the Nural Network Linear model
#Define the model

model=nn.Linear(3,2) #3 inputs and 2 outputs
                     #the transpose happens automatically inside the model
print("model weight",model.weight) #Here the model is automatically initialized with random weights
 # Heare we get weight matrix with 3 columns and 2 rows as we have 2 target labels (y=wx+b)
print("model bias",model.bias) #Here the model is automatically initialized with bias weights

print(list(model.parameters())) #printing all the parameters of the model i.e weights and bias


#this parameters we need to pass it to the optimizer to find the optimal value of w and b

preds= model(inputs)

print("preds",preds)

import torch.nn.functional as F
loss_fn=F.mse_loss
loss=loss_fn(preds,targets)
print("loss",loss)

#Define the optimizer

opt=torch.optim.SGD(model.parameters(),lr=0.01) #SGD is a stochastic gradient descent optimizer
                                       #lr= learning rate.
                                       #here we are passing weights and bias as parameters
                                       #

#training loop
pred1=model(torch.Tensor([[72., 54., 79.]]))
pred2=model(torch.Tensor([[65., 54., 79.],
        [24., 55., 77.],
        [72., 54., 79.],
        [25., 65., 43.],
        [43., 72., 74.]]))
print("pred1**********************",pred1)
print("pred2**********************",pred2)
def fit(num_epochs,model,loss_fn,opt,train_dataset):
    for epoch in range(num_epochs):
        for x, y in train_dataset:
            pred=model(x)   #get the prediction for batch of training data
            loss=loss_fn(pred,y) #calculate the loss difference between the prediction
                                    # and the actual target values
            loss.backward() #calculate the gradient of the loss function

            opt.step() #this step is for updating the new weights and bias automatically
            opt.zero_grad() #initialize the gradient to zero

        if (epoch+1)%2==0: #every 10 epochs we print the loss

            print(f"Epochs:{epoch+1}/{num_epochs}----Loss:{loss}")
           
# Train the model for 2 number of epochs and get the optimal value of w and b


fit(2,model,loss_fn,opt,train_dataset)

#predicting thea values trhough trained model  
print(model(torch.Tensor([[72., 54., 79.]]))) #
            
#calculate the accuracy of the model
import sklearn.metrics as metrics

from sklearn.metrics import accuracy_score
print(accuracy_score(pred1,targets))





    



