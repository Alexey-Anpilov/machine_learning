'''
Практическая работа №3.
Многослойный персептрон.
Вариант 1

1.  Двумерные входные данные для задания № 2:
    -   Первый класс -- точки внутри квадрата, второй -- точки вне него.

2.  Трехмерные входные данные для задания № 4:
    -   Точки внутри шара радиуса 1 с центром в (0, 0, 0);
    -   Точки выше плоскости x + y + z = 3;
    -   Точки вне параллелепипеда со сторонами (0.5, 1, 1.5) и центром (-1, 2, 1).
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from matplotlib import cm
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler

# функция для выделения на плоскости точек внутри/вне квадрата
def points_inside_square(x, y, a=0, b=0, d=2):
    return abs(x - a) + abs(y-b) <= d/2

# генерация точек внутри/вне квадрата
def generate_2d_points(n):
    X = np.random.uniform(low=-4., high=4., size=(n, 2))

    mask = points_inside_square(X[:, 0], X[:, 1])

    plt.scatter(X[mask, 0], X[mask, 1], c='r')
    plt.scatter(X[~mask, 0], X[~mask, 1], c='b')
    plt.show()

    y = np.array([int(m) for m in mask])
    return X, y


if __name__ == '__main__':
    np.random.seed(28673)
    
    # число точек в выборке
    n = 1000

    X, y = generate_2d_points(n)

    # для классификации квадрата на плоскости необходимо 
    # 4 нейрона на первом слое и 5 нейронов на втором
    activation_func = ['logistic', 'relu']
    for act in activation_func:
        mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        activation=act,
                        hidden_layer_sizes=(4, 5), random_state=42)

        mlp.fit(X, y)
        print('Precision for "{}" activation function: {}'.format(act, precision_score(y, mlp.predict(X))))

    # посмотрим на точность классификации в зависимости от числа нейронов
    precisions = list()
    for i in range(1, 9):
        row_res = list()
        for j in range(1, 9):
            mlp = MLPClassifier(solver='lbfgs',
                                activation='logistic',
                                hidden_layer_sizes=(i, j), 
                                random_state=42)
            mlp.fit(X, y)
            row_res.append(precision_score(y, mlp.predict(X), zero_division=0.0))
            if i == 3 and j == 3:
                coefs = mlp.coefs_[0]
                intercepts = mlp.intercepts_[0]
        precisions.append(row_res)
    
    precisions = np.array(precisions)

    # построим графики
    fig = plt.figure(figsize=(20, 20))
    axes = fig.subplots(nrows=4, ncols=4)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    axes = axes.flatten()
    for i in range(8):
        axes[i].plot(list(range(1,9)), precisions[i])
        axes[i].set_title('Нейронов на первом слое: {}'.format(i+1))
        axes[8 + i].plot(list(range(1,9)), precisions[:, i])
        axes[8 + i].set_title('Нейронов на втором слое: {}'.format(i+1))
    plt.show()


    '''
    # тут никогда ничего не было
    plt.figure(figsize=(12, 8))
    for i in range(coefs.shape[1]):
        w = coefs[:, i]
        b = intercepts[i]

        def logistic_function(x):
            return 1 / (1 + np.exp(-(np.dot(x, w) + b)))
        
        nx, ny = (101, 101)
        px = np.linspace(-4, 4, nx)
        py = np.linspace(-4, 4, ny)
        vx, vy = np.meshgrid(px, py)
        test_set = np.vstack([np.ravel(vx), np.ravel(vy)]).T
        y_vals = logistic_function(test_set)
        plt.scatter(test_set, y_vals)
        plt.show()
    '''
    
