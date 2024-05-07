'''
Практическая работа №4.
Кластеризация числовых и текстовых данных.
Вариант 1   

1. Алгоритм:
    -   Метод k-средних

2.  Числовые входные данные:
    -   Точки вида (a - 0.5, b, 0.5 + a + b + eps), где значения a, b, eps -- независимые
        случайные числа, равномерно распределенные на отрезке [-1, 1].
    -   Точки вида (a, b + 0.5, a * eps), где значения a, b, eps -- независимые случайные
        числа, равномерно распределенные на отрезке [-1, 1].

2.  Текстовые входные данные:
    -   Десять положительных отзывов на книги, либо фильмы и десять отрицательных отзывов
        на них.
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans, Birch

# генерация числовых данных
def generate_points(n):
    a = np.random.uniform(-1, 1, (n, 1))
    b = np.random.uniform(-1, 1, (n, 1))
    eps = np.random.uniform(-1, 1, (n, 1))

    cluster1 = np.concatenate((a - 0.5, b, 0.5 + a + b + eps), axis=1)
    cluster2 = np.concatenate((a, b + 0.5, a * eps), axis=1)
    points = np.vstack((cluster1, cluster2))
    
    return points

# отрисовка числовых данных
def draw_sample(points, labels=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if labels is None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b')
    else:
        n = len(np.unique(labels))
        for i in range(0, n):
            ax.scatter(points[labels==i, 0], points[labels==i, 1], points[labels==i, 2])
    plt.show()

# генерация списка с названиями файлов с текстовыми данными
def create_filenames_list():
    base = './texts/'
    files_list = list()
    for i in range(1, 11):
        files_list.append(base + 'pastry' + str(i) + '.txt')
    for i in range(1, 11):
        files_list.append(base + 'main_dishes' + str(i) + '.txt')

    return files_list


if __name__ == '__main__':
    # начнем с кластеризации цисловых данных

    # число точек
    n = 100

    # генерация числовой выборки
    points = generate_points(n)
    draw_sample(points)

    # выполним кластеризацию с разным числом заданных кластеров 
    # и разным значением генератора псевдослучайных чисел
    for n_cl in range(2, 6):        
        for r_st in (3, 42, 28673):
            kmeans = KMeans(n_clusters=n_cl, random_state=r_st, n_init='auto').fit(points)
            draw_sample(points, kmeans.labels_)


    # теперь кластеризация текстовых данных

    # задаём список файлов выборки
    input_files = create_filenames_list()

    # рассчитываем признаки, выводим их количество
    count_vect = CountVectorizer(input='filename')
    X_train_counts = count_vect.fit_transform(input_files)

    # кластеризуем методом k-средних
    k_means = KMeans(n_clusters=2, n_init='auto')
    result = k_means.fit(X_train_counts)
    print(result.labels_)

    # кластеризуем методом affinity propagation
    ap = Birch(n_clusters=2)
    result = ap.fit(X_train_counts)
    print(result.labels_)

