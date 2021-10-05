import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor                             # takes this image in and reshape a new tensor, in which each values
                                                                        # is normalize between 0 and 1

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = .001

class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):                                      # say to Pytorch how to manipulate the data (in which order)
        flatten_data = self.flatten(input_data)
        logits = self.dense_layers(flatten_data)
        predictions = self.softmax(logits)
        return predictions


def download_mnist_datasets():
    train_data = datasets.MNIST(                                        # MNIST is a dataset class
        root = "data",                                                  # where store the data
        download=True,                                                  # if there are no data, download them
        train=True,                                                     # We are interested in training 
        transform=ToTensor()                                            # allow to applies some sort of transformation directly to our dataset
    )
    validation_data = datasets.MNIST(                                   # MNIST is a dataset class
        root = "data",                                                  # where store the data
        download=True,                                                  # if there are no data, download them
        train=False,                                                    # We are interested in training 
        transform=ToTensor()    
    )
    return train_data, validation_data


def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for inputs, targets in data_loader:
        imputs, targets = inputs.to(device), targets.to(device)         # We need to assign the data to the device

        # calculate Loss
        predictions  = model(inputs)
        loss = loss_fn(predictions, targets)

        # backpropagate loss and update weights
        optimizer.zero_grad()                                           # at every iteration the optimizer is gonna calculate gradients to update the weights, these gradients
                                                                        # are saved. At each iteration we want to reset the gradients to zero, so we can start from scratch
        loss.backward()                                                 # apply back propagation
        optimizer.step()                                                # update the weights

    print(f'Loss: {loss.item()}')

def train(model, data_loader, loss_fn, optimizer, device, epochs):                                                            # higher level function that will use train_one_epoch at each iteration,
                                                                        # we will go throught all the epoch that we want to train the model for
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print("------------------")
    print("Train is done")

if __name__ == "__main__":
    train_data, _ = download_mnist_datasets()
    print("MNIST dataset downloaded") 

    #Now we use dataloader 
    # Dataloader --> class we can use to wrap a dataset, and it will allow us to load data in batches
    train_data_loader = DataLoader(train_data, batch_size = BATCH_SIZE)

    #build model 
    if torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cpu"

    feed_forward_net = FeedForwardNet().to(device)

    # instantiate loss function + optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(feed_forward_net.parameters(),lr=LEARNING_RATE)

    # Train our model 
    train(feed_forward_net, train_data_loader, loss_fn, optimizer, device, EPOCHS)

    torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth")                                           # state_dict is a python dictionary that contains all the state 
    print("Model trained and stored at feedforwardnet.pth")