import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Hyper Parameters
input_size = 784
num_classes = 10
#Tuned Hyper Parameters
hidden_size = 72
num_epochs = 50
batch_size = 250
learning_rate = 0.01
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.2860,], [0.3205,])])

# My Model
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//4)
        self.fc3 = nn.Linear(hidden_size//4, num_classes)
        self.logSoftMax = nn.LogSoftmax()
        self.relu = nn.ReLU()

        #Net Architecture Previous Tries
        self.LeakyRelu = nn.LeakyReLU(0.1)
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return self.logSoftMax(out)

def to_gpu(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def getLoaderErrorAndLoss(loader):
    correct,total,loss = 0,0,0
    for images, labels in loader:
        images = to_gpu(images.view(-1, 28 * 28))
        labels1 = to_gpu(labels)
        outputs = net(images)
        loss = criterion(outputs, labels1)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
    error = 1 - (float(correct) / total)
    loss = loss.item()
    return error, loss

if __name__== "__main__":
    # Fashion MNIST Dataset (Images and Labels)
    train_dataset = dsets.FashionMNIST(root = './data',train = True,download = True,transform = transform)
    test_dataset = dsets.FashionMNIST(root = './data',train=False,download = True,transform=transform)

    # Dataset Loader (Input Pipline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

    net = NeuralNet(input_size, num_classes)

    ## work on GPU
    net = to_gpu(net.cuda())
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.85)
    train_loss_list,test_loss_list,epoch_list,train_error_list,test_error_list = [],[],[],[],[]
    correct_train,total_train = 0,0
    print('Num of trainable parameters ', sum(p.numel() for p in net.parameters() if p.requires_grad))

    # Training the Model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = to_gpu(images.view(-1, 28 * 28))
            labels1 = to_gpu(labels)
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = net(images)
            train_loss = criterion(outputs, labels1)
            train_loss.backward()
            optimizer.step()
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train.cpu() == labels).sum()
            if (i + 1) % 80 == 0:
                print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
                       % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, train_loss.item()))
        train_error_list.append(1 - (float(correct_train) / total_train))
        train_loss_list.append(train_loss.item())
        epoch_list.append(epoch)
        net.eval()
        test_error, test_loss = getLoaderErrorAndLoss(test_loader)
        test_loss_list.append(test_loss)
        test_error_list.append(test_error)
        if epoch == num_epochs - 1:
            print ('Accuracy of the model on the 10000 test images: %.6f ', 1 - test_error)

    # Save the Model
    torch.save(net.state_dict(), "Trained Net.pkl")

    # figures
    # error plot
    plt.plot(epoch_list, train_error_list, epoch_list, test_error_list)
    plt.title('Errors on Train and Test')
    plt.ylabel('Error')
    plt.xlabel('Number of Epochs')
    plt.legend(['train_error', 'test error'], loc='upper right')
    plt.savefig('Errors.png')
    plt.close()

    # loss plot
    plt.plot(epoch_list, train_loss_list, epoch_list, test_loss_list)
    plt.title('Losses of Train and Test')
    plt.ylabel('Loss')
    plt.xlabel('Number of Epochs')
    plt.legend(['train_loss', 'test loss'], loc='upper right')
    plt.savefig('Losses.png')