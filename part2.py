import os
import torch
import numpy as np
import pandas as pd
from torch.functional import split
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import nn
from collections import OrderedDict
from torchvision import datasets, models, transforms
from torch.utils.data import ConcatDataset, DataLoader, Subset

translate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly",
             "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", 
             "ragno": "spider", "scoiattolo": "squirrel" }

classes = ["cane", "cavallo", "elefante", "farfalla", "gallina", "gatto", "mucca", "pecora","ragno", "scoiattolo" ]

batch_size = 60
disimage = 20
epoch = 0
learning_rate = 0.005
is_all_grad = True

def initialize_data():
    for dirname, _, filenames in os.walk('./raw-img'):
        for filename in filenames:
            path, folder = os.path.split(dirname)

    data_transform = transforms.Compose([transforms.RandomRotation(45),
                                          transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(1080),
                                          transforms.Resize(512),
                                          transforms.Resize(224),
                                          transforms.RandomRotation(45),
                                          transforms.ToTensor()])

    dataset = datasets.ImageFolder(path, transform=data_transform)

    max_length = 15
    idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] == dataset.class_to_idx[dataset.classes[0]]]
    subset = Subset(dataset, idx)
    data = Subset(subset, idx[:max_length])

    for j in range(1, 10):
        idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] == dataset.class_to_idx[dataset.classes[j]]]
        subset = Subset(dataset, idx[:max_length])
        data = ConcatDataset((data, subset))
        print(len(data))

    return data


dataset = initialize_data()

#...
def split_data(dataset_length, validation_size=0.1, test_size=0.1):
    indices = [i for i in range(dataset_length)]
    np.random.shuffle(indices)
    validation_dataset = int(np.floor((validation_size) * dataset_length))
    test_dataset = int(np.floor((validation_size+test_size) * dataset_length))
    validation_idx, test_idx, train_idx = indices[:validation_dataset], indices[validation_dataset:test_dataset], indices[test_dataset:]
    return validation_idx, test_idx, train_idx

#..
validation_idx, test_idx, train_idx = split_data(len(dataset), 0.1, 0.1)

train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size)
valid_loader = DataLoader(Subset(dataset,validation_idx), batch_size=batch_size)
test_loader = DataLoader(Subset(dataset,test_idx), batch_size=batch_size)

#...
def create_model(is_all_grad=True):
    model = models.vgg19(pretrained=True)
    model.cu
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 6000)), ('relu', nn.ReLU()), ('dropout', nn.Dropout(.5)), ('fc2', nn.Linear(6000, 10)), ('output', nn.Softmax(dim=1) )]))        
    model.classifier = classifier

    if(not is_all_grad):
        for name, param in model.named_parameters():
            print(name, param.size())
            param.requires_grad = False

        for name, param in model.named_parameters():
            if(name == "classifier.fc1.weight" or name == "classifier.fc1.bias" or name == "classifier.fc2.weight" or name == "classifier.fc2.bias"):
                print(name)
                print(param.requires_grad)
                param.requires_grad = True
                print(param.requires_grad)


    return model

model = create_model(False)




# specify optimizer (stochastic gradient descent) and learning rate = 0.001
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

loss_function = nn.NLLLoss()


def seq(model, df, name):
    train_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    for batch_i, (data, target) in enumerate(df):
        print("Batch " + str(batch_i) + ":")
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = loss_function(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        if name == 'train': 
            loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss 
        train_loss += loss.item()
        _, pred = torch.max(output, 1) 
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy())
        for i in range(len(target.data)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
        
    return class_correct, class_total, train_loss

def printdata(class_correct, class_total, train_loss, epoch, name, df, loss_values, accuracy_values):
    loss = train_loss / len(df)
    accuracy = 100.0 * np.sum(class_correct) / np.sum(class_total)
    print("Epoch ", epoch, "loss: ", loss, "\t", name, " Accuracy(Overall): ", accuracy, "(", np.sum(class_correct), "/", np.sum(class_total), ")")
    print("----")
    print(f'Epoch %d, loss: %.8f \t{name} Accuracy (Overall): %2d%% (%2d/%2d)' %(epoch,
        train_loss / len(df), 100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
    
    #delete later
    for i in range(10):
        if class_total[i] > 0:
            print(f'{name} Accuracy of %5s: %2d%% (%2d/%2d)' % (
            translate[classes[i]], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))

    loss_values.append(loss)
    accuracy_values.append(accuracy)

def trainModel(model, train_loader,valid_loader, num_epochs=1):
    train_loss_values = []
    train_accuracy_values = []
    validation_loss_values = []
    validation_accuracy_values = []

    # number of epochs to train the model
    for epoch in range(1, num_epochs+1):
        train_loss = 0.0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        ###################
        # train the model #
        ###################
       # Repeat for each batch in the training set
        model.train()
        class_correct, class_total, train_loss= seq(model,  train_loader, 'train')
        printdata(class_correct, class_total, train_loss, epoch, 'train', train_loader, train_loss_values, train_accuracy_values)
        # Repeat for each validation batch 
        model.eval()
        class_correct, class_total, train_loss= seq(model, valid_loader, 'validation')
        printdata(class_correct, class_total, train_loss, epoch, 'validation', valid_loader, validation_loss_values, validation_accuracy_values)

    plot_loss_and_accuracy(train_loss_values, train_accuracy_values, validation_loss_values, validation_accuracy_values)
    torch.save(model.state_dict(), 'model.pt')     
    

def plot_loss_and_accuracy(train_loss_values, train_accuracy_values, validation_loss_values, validation_accuracy_valeus):
    plt.subplot(1,2,1)
    plt.title("TRAIN DATA \n" + "-- Learning Rate:" + str(learning_rate) + "-- Batch:" + str(batch_size))
    plt.plot(np.arange(0, len(train_loss_values)), train_loss_values, label="train")
    plt.plot(np.arange(0, len(validation_loss_values)), validation_loss_values, label="validation")
    plt.ylabel("Cost")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig("cost.png")

    plt.subplot(1,2,2)
    plt.title("TRAIN DATA \n" + "-- Learning Rate:" + str(learning_rate) + "-- Batch:" + str(batch_size))
    plt.plot(np.arange(0, len(train_accuracy_values)), train_accuracy_values, label="train")
    plt.plot(np.arange(0, len(validation_accuracy_valeus)), validation_accuracy_valeus, label="validation")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig("accuracy.png")
    plt.show()

print(len(train_loader), len(valid_loader))
trainModel(model, train_loader,valid_loader,2)

test_loss = 0.0
class_correct = [0. for i in range(10)]
class_total = [0. for i in range(10)]
model.eval()
class_correct, class_total, train_loss= seq(model, test_loader, 'test')
printdata(class_correct, class_total, train_loss, 1, 'test', test_loader, [], [])

# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()
images.numpy()

# get sample outputs
output = model(images)
images = images.cpu()
# convert output probabilities to predicted class
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy())

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(disimage):
    ax = fig.add_subplot(2, disimage/2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
    ax.set_title("{} ({})".format(translate[classes[preds[idx]]], translate[classes[labels[idx]]]),
                 color=("green" if preds[idx]==labels[idx].item() else "red"))

plt.show()
