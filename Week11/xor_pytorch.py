"""
Implementation des XOR-Beispiels mittels PyTorch
"""

# Installation: https://pytorch.org/get-started/locally/
# z.B. pip3 install torch torchvision
# Anaconda: conda install pytorch torchvision torchaudio cpuonly -c pytorch

#Quellen für den folgenden Code:
# https://weiliu2k.github.io/CITS4012/pytorch/nn_oop.html
# https://machinelearningmastery.com/how-to-evaluate-the-performance-of-pytorch-models/


import torch

# Jedes neuronale Netz wird als Python-Klasse modelliert, die von der allgemeinen Klasse torch.nn.Module erbt
class XOR(torch.nn.Module):
    """
    An XOR is similuated using neural network with
    two fully connected linear layers
    """

    def __init__(self, input_dim, output_dim):
        """
        Args:
            input_dim (int): size of the input features
            output_dim (int): size of the output
        """

        # Initialisierung der übergeordneten Klasse, von der geerbt wird
        super(XOR, self).__init__()

        # Anlegen der Layers; hier 1. Layer mit 2 Hidden Nodes
        self.fc1 = torch.nn.Linear(input_dim, 2)
        self.fc2 = torch.nn.Linear(2, output_dim)

    def forward(self, x_in):
        """The forward pass of the perceptron

        Args:
            x_in (torch.Tensor): an input data tensor
                x_in.shape should be (batch, num_features)
        Returns:
            the resulting tensor. tensor.shape should be (batch,).
        """
        hidden = torch.relu(self.fc1(x_in))
        yhat = torch.sigmoid(self.fc2(hidden))
        return yhat


# Create training data for XOR problem
# "Tensors are a specialized data structure that are very similar to arrays and matrices. In PyTorch, we use tensors to encode
# the inputs and outputs of a model, as well as the model’s parameters. Tensors are similar to NumPy’s ndarrays, except
# that tensors can run on GPUs or other specialized hardware to accelerate computing." (https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html)
x_train_tensor = torch.tensor([[0,0],[0,1],[1,1],[1,0]]).float()
y_train_tensor = torch.tensor([0,1,1,0]).view(4,1).float()


# Verify the shape of the output tensor
#print(y_train_tensor.shape)

# Hyperparameter setup
# We need to set up the learning rate and the number of epochs, and then select the three key components of a neural model:
# model, optimiser and loss function before training

# Sets learning rate - this is "eta"
lr = 0.01

# Step 0 - Initializes parameters "b" and "w" randomly
#torch.manual_seed(42)
# Now we can create a model
model = XOR(input_dim= 2,output_dim=1)


# Defines a stochastic gradient descent (SGD) optimizer to update the parameters
# (now retrieved directly from the model)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Defines a (binary) Cross Entropy loss function
loss_fn = torch.nn.BCELoss()

# Defines number of epochs
n_epochs = 10000

### Training
for epoch in range(n_epochs):

   #In PyTorch, models have a train() method which, somewhat disappointingly, does NOT perform a training step.
   # Its only purpose is to set the model to training mode. Why is this important? Some models may use mechanisms like
   # Dropout, for instance, which have distinct behaviors during training and evaluation phases.
    model.train()

    # Step 1 - Computes model's predicted output - forward pass
    yhat = model(x_train_tensor)

    # Step 2 - Computes the loss
    loss = loss_fn(yhat, y_train_tensor)

    # Step 3 - Computes gradients for both "b" and "w" parameters
    loss.backward()

    # Step 4 - Updates parameters using gradients and the learning rate
    optimizer.step()
    optimizer.zero_grad()

    #print intermediate results
    #if (epoch % 100 == 0):
    #    print("Epoch: {0}, Loss: {1}, ".format(epoch, loss.detach().numpy()))

# We can also inspect its parameters using its state_dict
#print(model.state_dict())

# Prediction (on training set)
# evaluate model
y_pred = model(x_train_tensor)
#print(y_pred)
acc = (y_pred.round() == y_train_tensor).float().mean()
acc = float(acc)
#print(f"accuracy: {acc}")

