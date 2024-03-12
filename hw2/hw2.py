'''
Практическая работа №2. 
Простейшие виды классификации
Вариант 1.

1.  Генерация входных данных:
    1.  Точки вида (a - 0.5, b, 0.5 + a + b + eps), где значения a, b, eps -- независимые
        случайные числа, равномерно распределенные на отрезке [-1, 1].
    2.  Точки вида (a, b, -0.5 + b * eps), где значения a, b, eps -- независимые случайные
        числа, нормально распределенные со средним значением 0 и среднеквадратическим отклонением 0.5.
    3.  Случайные точки трехмерного пространства, лежащие внутри полушария x^2 + y^2 + z^2 = 1, z >= 0;
        способ распределения произвольный.

2.  Используемые алгоритмы:
    1. Линейная пороговая классификация.
    2. Метод опорных векторов.
    3. Наивный байесовский классификатор (любого подходящего к условиям задачи подтипа). 
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

# генерация первого набора точек
def generate_points1(n):
    a = np.random.uniform(-1, 1, (n, 1))
    b = np.random.uniform(-1, 1, (n, 1))
    eps = np.random.uniform(-1, 1, (n, 1))
    
    sample = np.concatenate((a - 0.5, b, 0.5 + a + b + eps), axis=1)
    return sample

# генерация второго набора точек
def generate_points2(n):
    a = np.random.normal(loc=0, scale=0.5, size=(n, 1))
    b = np.random.normal(loc=0, scale=0.5, size=(n, 1))
    eps = np.random.normal(loc=0, scale=0.5, size=(n, 1))
    
    sample = np.concatenate((a, b, -0.5 + b*eps), axis=1)    
    return sample

# генерация третьего набора точек
def generate_points3(n):
    theta = np.random.uniform(0, 2*np.pi, size=(n, 1))
    phi = np.arccos(np.random.uniform(0, 1, size=(n, 1)))
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    x = x[z >= 0]
    y = y[z >= 0]
    z = z[z >= 0]

    sample = np.column_stack((x, y, z))
    return sample

# генерация "ответов"(классов) для всех трех вариантов
def generate_classes(n):
    y1 = np.array([0] * n + [1] * 2 * n)
    y2 = np.array([0] * 2 * n + [1] * n)
    y3 = np.array([0] * n + [1] * n + [2] * n)
    return y1, y2, y3

# вывод показателей точности
def print_metric(metric_list, metric_name):
    print('{}: '.format(metric_name))
    models = ["SGD_classification", "SV_classification", "Naive_Bayes_classification"]
    for model, res in zip(models, metric_list):
        res_str = ' '.join(list(map(lambda x: '{:.2f}'.format(x), res)))
        print('{}: {}'.format(model, res_str))
    print()

# выводит разделяющую гиперплоскость и выборку для бинарной классификации 1 и 2 алгоритмом 
def draw_sample_and_hyperplane(X, y, models):
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')

    ax.scatter(X[y == 0, 0], X[y == 0, 1], X[y == 0, 2], c='r', marker='^', label='Class 0')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], c='b', marker='o', label='Class 1')
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 50),
                         np.linspace(X[:, 1].min(), X[:, 1].max(), 50))
    
    for model in models:
        zz = (-model.intercept_[0] - model.coef_[0][0] * xx - model.coef_[0][1] * yy) / model.coef_[0][2]
        ax.plot_surface(xx, yy, zz, alpha=0.4)
    
    ax.legend()
    plt.show()


if __name__ == "__main__":
    np.random.seed(286754)
    # размер выборки
    n = 25

    # создание выборки и соответсвующих классов-ответов для разных вариантов
    X = np.concatenate((generate_points1(n), generate_points2(n), generate_points3(n)), axis=0)
    y1, y2, y3 = generate_classes(n)

    answers = {"Sample 1": y1, 
               "Sample 2": y2, 
               "Sample 3": y3}

    # создание моделей
    sgd_clf = SGDClassifier(random_state=42)
    sv_clf = LinearSVC(dual='auto', random_state=42)
    naive_bayes = GaussianNB()

    models = [sgd_clf, sv_clf, naive_bayes]


    # обучение и вывод показателей точности -- выводятся значения precision и recall
    # для всех классов, для каждой модели, на каждой выборке
    for y_name, y in answers.items():
        precision_res = list()
        recall_res = list()
        for model in models:
            model.fit(X, y)
            y_pred = model.predict(X)
            precision_res.append(metrics.precision_score(y, y_pred, average=None, zero_division=0.0))
            recall_res.append(metrics.recall_score(y, y_pred, average=None, zero_division=0.0))
        print("{}: ".format(y_name))
        print_metric(precision_res, "Precision")
        print_metric(recall_res, "Recall")
        print('-'*10)


    # Визуализация разделяющих плоскостей для 1 и 2 алгоритмов на первой выборке
    sgd_clf.fit(X, y1)
    sv_clf.fit(X, y1)
    draw_sample_and_hyperplane(X, y1, [sv_clf, sgd_clf])
    

    # отрисовка roc-кривых для первой выборки
    fpr,tpr,t3 = metrics.roc_curve(y1, sgd_clf.decision_function(X))  
    fpr1,tpr1,t2 = metrics.roc_curve(y1, sv_clf.decision_function(X))
    fpr2, tpr2, t2 = metrics.roc_curve(y1, naive_bayes.predict_proba(X)[:, 1])
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label='SGD')
    ax.plot(fpr1, tpr1, label = 'SV')
    ax.plot(fpr2, tpr2, label='Naive Bayes')
    ax.legend()
    ax.set_title('ROC-curves')
    plt.show()    