"""
Практическая работа №8.
Генеративно-состязательная нейросеть.
"""
import math
import random

import torch
from torch import nn

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# функция для генерации обучающего датасета
def generate_dataset(N, r=1):
    train_dataset = torch.zeros((N, 3))
    curr_len = 0
    while curr_len < N:
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if math.hypot(x,y) < 1:
            train_dataset[curr_len, 0] = x
            train_dataset[curr_len, 1] = y
            train_dataset[curr_len, 2] = (1 - x**2 - y**2)**0.5
            curr_len += 1
    return train_dataset

# класс, реализующий нейросеть дискриминатора
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid())

    def forward(self, x):
        output = self.model(x)
        return output

# класс, реализуюищй нейросеть генератора
class Generator(nn.Module):
    def __init__(self, hidden_param_num):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(hidden_param_num, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3))

    def forward(self, x):
        output = self.model(x)
        return output


def train_and_test_gan(train_loader, hidden_param_num, discr, gen):
    # инициализируем сети дискриминатора и генератора
    discriminator = discr
    generator = gen

    # параметры обучения: скорость, число эпох, функция потерь, оптимизатор
    lr = 0.001
    num_epochs = 300
    loss_function = nn.BCELoss()
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

    # для каждой эпохи:
    for epoch in range(num_epochs):
        for n, (real_samples, real_samples_1d) in enumerate(train_loader):
            # Данные для обучения дискриминатора
            real_samples_labels = real_samples_1d.view(batch_size, 1)
            latent_space_samples = torch.randn((batch_size, hidden_param_num))
            generated_samples = generator(latent_space_samples)
            generated_samples_labels = torch.zeros((batch_size,1))
            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat((real_samples_labels,
                                            generated_samples_labels))

            # Обучение дискриминатора
            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(output_discriminator,
                                               all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Данные для обучения генератора
            latent_space_samples = torch.randn((batch_size, hidden_param_num))

            # Обучение генератора
            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples)
            loss_generator = loss_function(output_discriminator_generated,
                                           real_samples_labels)
            loss_generator.backward()
            optimizer_generator.step()

        # Выводим значения функций потерь
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")

    # строим примеры точек сферы
    latent_space_samples = torch.randn((100, hidden_param_num))
    generated_samples = generator(latent_space_samples)

    # если использовалась видеокарта, выгружаем данные с неё
    generated_samples = generated_samples.detach()

    return generated_samples


def draw_train_and_generated(train_dataset, generated_samples, title=''):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title(title)
    ax.scatter(train_dataset[:, 0], train_dataset[:, 1], train_dataset[:, 2], 'b')
    ax.scatter(generated_samples[:, 0], generated_samples[:, 1], generated_samples[:, 2], 'r')
    plt.show()



if __name__ == '__main__':
    random.seed(17)
    torch.manual_seed(19)
    N = 512

    # генерирурем тренировочные данные
    train_dataset = generate_dataset(N)
    train_labels = torch.ones(N)
    train_set = [(train_dataset[i], train_labels[i]) for i in range(N)]

    # создаем загрузчик данных
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True)

    # инициализируем сети дискриминатора и генератора
    hidden_param_num = 4
    discriminator = Discriminator()
    gen = nn.Sequential(
            nn.Linear(hidden_param_num, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3))
    gen_samples = train_and_test_gan(train_loader, hidden_param_num, discriminator, gen)
    draw_train_and_generated(train_dataset, gen_samples, '')

    # проверим, что будет при исключении dropout
    discr1 = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid())
    gen_samples = train_and_test_gan(train_loader, hidden_param_num, discr1, gen)
    draw_train_and_generated(train_dataset, gen_samples, 'Without dropout')

    # проверим, что будет, если поменять число слоев в генераторе
    # добавим 1 слой
    gen1 = nn.Sequential(
            nn.Linear(hidden_param_num, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 3))
    gen_samples = train_and_test_gan(train_loader, hidden_param_num, discriminator, gen1)
    draw_train_and_generated(train_dataset, gen_samples, 'Generator with more layers')

    # уберем 1 слой
    gen2 = nn.Sequential(
            nn.Linear(hidden_param_num, 64),
            nn.ReLU(),
            nn.Linear(64, 3))
    gen_samples = train_and_test_gan(train_loader, hidden_param_num, discriminator, gen2)
    draw_train_and_generated(train_dataset, gen_samples, 'Generator with less layers')

    # В целом я пробовал разные варианты. При увеличении числа слоев и нейронов в генераторе
    # точки часто генерировались в ограниченном сегменте верхней полусферы, в некоторых случаях
    # располагались достаточно далеко от сферы(длина радиуса-вектора ~ от 1.5 до 5). Я предполагаю,
    # что сеть недообучалась, поскольку даже в случае с gen1 увеличение числа эпох в два раза позволяет
    # получить нормальный результат.

    # поменяем число скрытых параметров
    for hid_param in (2, 4, 8):
        generator = nn.Sequential(
            nn.Linear(hid_param, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3))
        gen_samples = train_and_test_gan(train_loader, hid_param, discriminator, generator)
        draw_train_and_generated(train_dataset, gen_samples, 'Number of hidden parameters: {}'.format(hid_param))


    