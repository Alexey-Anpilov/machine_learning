'''
Практическая работа № 5.
Сверточные нейронные сети

Набор данных: MNIST;
Требования к архитектуре сети: число свёрточных слоёв -- 2, 
    размер окна пулинга -- 3 x 3.
'''
import torch
import torchvision
from torchvision.transforms import ToTensor
from torch.utils.data import random_split, DataLoader
from torch import nn
from torch.optim import Adam
from sklearn.metrics import classification_report
import numpy as np

TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT
BATCH_SIZE = 64
INIT_LR = 1e-3
EPOCHS = 10

# класс для реализации сверточной нейросети
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # первый сверточный слой, ReLu активация и пулинг
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=30,
                               kernel_size=(3, 3))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        # второй сверточный слой, ReLu активация и пулинг
        self.conv2 = nn.Conv2d(in_channels=30, 
                               out_channels=50,
                               kernel_size=(3, 3))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # первый полносвязный слой
        self.fc1 = nn.Linear(50*5*5, 200)
        self.relu3 = nn.ReLU()
        # второй полносвязный слой
        self.fc2 = nn.Linear(200, 10)
        # softmax для масштабирования выхода
        self.logSoftmax = nn.LogSoftmax(dim=1)

        
    def forward(self, x):
        # сверточные слои
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x, 1)

        # полносвязные слои
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        output = self.logSoftmax(x)

        return output


if __name__ == '__main__':
    # загрузка тренировочных и тестовых данных
    train_data = torchvision.datasets.MNIST(root='./data',
                           download=True,
                           train=True,
                           transform=ToTensor())

    test_data =  torchvision.datasets.MNIST(root='./data',
                           download=True,
                           train=False,
                           transform=ToTensor())
    
    # разделение тренировочных данных на обучающую и валидационную выборку
    numTrainSamples = int(len(train_data) * TRAIN_SPLIT)
    numValSamples = int(len(train_data) * VAL_SPLIT)
    train_data, val_data = random_split(train_data, 
             [numTrainSamples, numValSamples],
             generator=torch.Generator().manual_seed(42))
    
    # создаем объекты-загрузчики для группировки объектов в батчи
    train_loader = DataLoader(train_data, shuffle=True,
                             batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    trainSteps = len(train_loader.dataset) // BATCH_SIZE
    valSteps = len(val_loader.dataset) // BATCH_SIZE
    device = torch.device("cpu")

    # инициализируем модель, алгоритм оптимизации и функцию ошибок
    model = ConvNet()
    opt = Adam(model.parameters(), lr=INIT_LR)
    lossFn = nn.NLLLoss()

    # обучение модели
    for e in range(0, EPOCHS):
        model.train()
        totalTrainLoss = 0
        totalValLoss = 0
        trainCorrect = 0
        valCorrect = 0

        for (x, y) in train_loader:
            (x, y) = (x.to(device), y.to(device))
            pred = model(x)
            loss = lossFn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            totalTrainLoss += loss
            trainCorrect += (pred.argmax(1) == y).type(
                torch.float).sum().item()
        
        with torch.no_grad():
            model.eval()
            for (x, y) in val_loader:
                (x, y) = (x.to(device), y.to(device))
                pred = model(x)
                totalValLoss += lossFn(pred, y)
                valCorrect += (pred.argmax(1) == y).type(
                    torch.float).sum().item()
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        trainCorrect = trainCorrect / len(train_loader.dataset)
        valCorrect = valCorrect / len(val_loader.dataset)
        print("Epoch {}/{}".format(e + 1, EPOCHS))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
            avgTrainLoss, trainCorrect))
        print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
            avgValLoss, valCorrect))
    
    # проверяем обученную модель на тестовой выборке
    with torch.no_grad():
        model.eval()
        preds = []
        for (x, y) in test_loader:
            x = x.to(device)
            pred = model(x)
            preds.extend(pred.argmax(axis=1).cpu().numpy())
    
    # распечатать отчёт о классификации
    print(classification_report(test_data.targets.cpu().numpy(),
        np.array(preds), target_names=test_data.classes))
