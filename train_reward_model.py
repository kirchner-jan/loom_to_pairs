import itertools
import torch, wandb
from torch import nn
from torch.utils.data import DataLoader
from loom_dataset import LoomComparisonsDataset
import datetime

# Get CPU or GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self , input_size , hidden_size , output_size=1):
        """
        We create a neural network with a single hidden layer, and the hidden layer has a ReLU
        activation function
        
        :param input_size: The number of features in the input data
        :param hidden_size: The number of neurons in the hidden layer
        :param output_size: The number of outputs the network will produce, defaults to 1 (optional)
        """
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size , hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size , output_size)
        )
    def forward(self , x):
        yhat = self.linear_relu_stack(x)
        return yhat

def loss_fn(r0 , r1 , y0 , y1):
    """
    > The loss function is the logistic loss function, which is the log of the sum of the exponential of
    the difference between the predicted scores of the two items
    
    :param r0: the score of the first item
    :param r1: the score of the first item
    :param y0: the score of the first item
    :param y1: the label of the first image
    :return: The loss function is being returned.
    """
    return torch.log(1 + torch.exp(r0 - r1)) if y1 > y0 else torch.log(1 + torch.exp(r1 - r0))

def train(dataloader, model, loss_fn, optimizer):
    """
    > For each batch of embeddings, compute the predicted rewards for each embedding, compute the loss,
    and backpropagate
    
    :param dataloader: a dataloader object that returns a batch of embeddings
    :param model: the model we're training
    :param loss_fn: the loss function to use
    :param optimizer: the optimizer used to train the model
    """
    model.train()
    for embed_stubs in dataloader:
        # compute prediction error
        predicted_rewards = [model(stub[2].to(device))[0][0] for stub in embed_stubs]
        rewards = [stub[1] for stub in embed_stubs]
        reward_tuple = [a for a in zip(predicted_rewards , rewards)]
        # compute loss
        loss = torch.tensor([loss_fn(r0 , r1 , y0 , y1) for (r0,y0),(r1,y1) in itertools.combinations(reward_tuple , 2) ], requires_grad=True)
        loss = torch.mean(loss)
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        wandb.log({"loss": loss})



epochs = 500
pickle_file = 'data/Sample_ARCADIA.pkl'
# Set up logging
wandb.init(project='Utility Learning Loom', config={'pickle_file': pickle_file})
# Create model
model = NeuralNetwork(768 , 10).to(device)
wandb.watch(model)
# Define the dataloader
loomdts = LoomComparisonsDataset(pickle_file)
loomdts_loader = DataLoader(loomdts , batch_size=1 , shuffle=True)
# Run the training loop
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
for t in range(epochs):
    if t % 100 == 0:
        print(f"Epoch {t+1}" , end=', ' )
    train(loomdts_loader, model, loss_fn, optimizer)
print("Done!")
torch.save(model.state_dict(), 'model/' + str(datetime.datetime.now()).replace(' ','_') + '.pt')
