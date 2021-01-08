# Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

torch.manual_seed(123)

# CUDA
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('CUDA OK')
else:
    device = torch.device('cpu')
    print('CUDA DISABLED')

# Type
dtype = torch.float32

# Transforms
transforms = transforms.Compose([
    transforms.CenterCrop(120),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
 ])

# NN
class COVID_NN:
    def __init__(self, batch):
        self.criterion = nn.CrossEntropyLoss()
        self.model_gn = self.model()
        self.optimizer = optim.SGD(self.model_gn.parameters(), lr=0.001, momentum=0.9)
        self.losses = []
        self.predictions = []
        self.batch = batch
        self.dataset, self.validation = self.load_dataset(batch)

    def model(self):
        gnet = models.googlenet(pretrained=True)

        # Freeze weights
        for i, child in enumerate(gnet.children()):
            if i < 2:
                for param in child.parameters():
                    param.requires_grad = False

        gnet.fc = nn.Linear(1024, 3)

        if torch.cuda.is_available():
            gnet.cuda()

        return gnet

    def load_dataset(self, batch_size):
        path = os.getcwd() + '/COVID-19 Radiography Database'
        train_dataset = datasets.ImageFolder(path, transform=transforms)
        train, test = torch.utils.data.random_split(train_dataset, [2829, 1000])
        train = torch.utils.data.DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True
        )
        test = torch.utils.data.DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True
        )

        return train, test

    def train(self, epochs):
        for epoch in range(epochs):
            print('Epoch =' + str(epoch + 1))
            for batch_idx, (data, target) in enumerate(self.dataset):

                inputs, labels = data.to(device), target.to(device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                output = self.model_gn(inputs)
                loss = self.criterion(output, labels)
                self.losses.append(loss.item())
                loss.backward()
                self.optimizer.step()

            print(sum(self.losses) / len(self.losses))
        with torch.no_grad():
            predicted = []
            targets = []

            for batch_idx, (data, target) in enumerate(self.validation.dataset):

                inputs, labels = data.to(device), target.to(device)

                # forward + backward + optimize
                output = self.model_gn(inputs)
                output = [x.argmax() for x in output]

                predicted.append(output)
                targets.append(labels)

            predicted = sum(predicted, [])

            predicted = [i.item() for i in predicted]

            targets = torch.cat(targets).cpu().detach().numpy()

            self.predictions = predicted
            self.predictions = [predicted, targets]
            self.predictions = list(zip(self.predictions[0], self.predictions[1]))

    def predict(self, x):
        with torch.no_grad():
            output = self.model_gn(x.to(device))
            output = [x.argmax() for x in output]
            return self.model_gn(output.to(device))


nn = COVID_NN(64)
nn.train(20)

num = 0
for i in nn.predictions:
    if i[0] == i[1]:
        num += 1

num / len(nn.predictions)

FN = 0
for i in nn.predictions:
    if i[0] != 0 and i[1] == 0:
        FN += 1

FN / len(nn.predictions)


FN = 0
for i in nn.predictions:
    if i[1] != i[0]:
        print(i)
        FN += 1

FN / len(nn.predictions)
