'''
Практическая работа № 6.
Автокодировщики

Набор данных: CIFAR10;
Требования к архитектуре сети: число слоев кодировщика и декодеровщика -- 3,
    размер закодированных данных -- 25.
'''



import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn 
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10
from torch.optim import Adam

# класс автокодировщика
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # сеть, реализующая кодировщик
        self.encoder = nn.Sequential(
			nn.Linear(32 * 32, 250),
			nn.ReLU(),
			nn.Linear(250, 100),
			nn.ReLU(),
			nn.Linear(100, 25),
		)
		
        # сеть, реализующая декодировщик
        self.decoder = nn.Sequential(
			nn.Linear(25, 100),
			nn.ReLU(),
			nn.Linear(100, 250),
			nn.ReLU(),
			nn.Linear(250, 32 * 32),
			nn.Sigmoid()
		)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == '__main__':
    # загрузка данных
    train_data = CIFAR10(root='./data', train=True,
                         download=True,
                         transform=ToTensor())
    
    test_data = CIFAR10(root='./data', train=False,
                        download=True,
                        transform=ToTensor())
    
    # оставляем два класса по заданию
    # P.S. увидел это требование в конце, изначально делал со всем классами,
    # результаты не изменились
    idx_train = np.where((np.array(train_data.targets) == 0) | (np.array(train_data.targets) == 1))[0]
    idx_test = np.where((np.array(test_data.targets) == 0) | (np.array(test_data.targets) == 1))[0]
    train_data = Subset(train_data, idx_train)
    test_data = Subset(test_data, idx_test)

    # создаем объекты-загрузчики
    train_loader = DataLoader(train_data, batch_size=32,
                          shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)

    # инициализируем модель, алгоритм оптимизации и функцию ошибок
    model = Autoencoder()
    loss_function = nn.MSELoss()
    optimizer = Adam(model.parameters(),
                 lr=1e-4,
                 weight_decay=1e-8)
    
    losses = list()
    # обучение модели
    for epoch in range(10):
        for image, _ in train_loader:
            image = image.reshape(-1, 32*32)
            reconstructed = model(image)

            loss = loss_function(reconstructed, image)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().numpy())

        print("Epoch {}/{}".format(epoch + 1, 10))

    # выведем график из 100 последних ошибок
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(losses[-100:])
    plt.show()


    for i,(image, _) in enumerate(test_loader):
        reshaped_image = image.reshape(-1, 32*32)
        reconstructed = model(reshaped_image)
        img = reconstructed.reshape(-1, 32, 32)

        f, axarr = plt.subplots(2,1) 
        raw_image = image[0].detach().numpy()[0,:,:]
        rec_image = img[0].detach().numpy()
        axarr[0].imshow(raw_image)
        axarr[1].imshow(rec_image)
        plt.show(block=True)
        if i == 3:
            break