'''
Практическая работа №1. 
Вариант 1.
Входные данные: точки двумерного пространства вида (x, sin(x)), 
где значения x -- случайные числа из отрезка [-pi/2, pi/2].

Алгоритмы:
- линейная регрессия
- сплайновая интерполяция кубическими сплайнами
'''
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin
from scipy.interpolate import CubicSpline



# размер выборки
n = 15

# генерации выборки
def generate_sample(n):
    x = np.random.uniform(-pi/2, pi/2, (n, 1))
    y = np.sin(x)
    return (x, y)

# отрисовка выборки
def draw_sample(x, y):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    plt.show()

# отрисовка выборки и предсказания
def draw_everything(x, y, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.scatter(x, y_pred, c='red')
    plt.show()

# класс для реализации линейной регрессии "руками"
class PolynomialRegression():
    def __init__(self, powers):
        self.c = None
        self.powers = sorted(powers)


    def fit(self, x, y):
        x_m = np.concatenate((np.ones((len(x), 1)), x), axis=1)
        for p in self.powers:
            x_m = np.concatenate((x_m, x**p), axis=1)
        self.c = np.linalg.inv(x_m.T @ x_m) @ x_m.T @ y
    

    def predict(self, x):
        if self.c is None:
            print("Fit before predicting")
            return 
        x_m = np.concatenate((np.ones((len(x), 1)), x), axis=1)
        for p in self.powers:
            x_m = np.concatenate((x_m, x**p), axis=1)
        return x_m @ self.c
        
    
def reshape_for_cs(x, y):
    x = x.reshape(-1)
    y = y.reshape(-1)
    sort_ind = x.argsort()
    x = x[sort_ind]
    y = y[sort_ind]   
    return x, y

# рассчет ошибки
def mse(y, y_pred):
    return np.square(np.subtract(y, y_pred)).mean()
 

def predict_and_draw(x, y, model, model_name, draw=False):
    y_pred = model(x)
    print('{} mse: {:.2}'.format(model_name, mse(y, y_pred)))
    if draw:
        draw_everything(x, y, y_pred)


if __name__ == "__main__":
    x, y = generate_sample(n)
    draw_sample(x, y)
    
    print('initial sample:')

    pr = PolynomialRegression((2,3,5))    
    pr.fit(x, y)
    predict_and_draw(x, y, pr.predict, 'polinomial regression')
    
    
    x, y = reshape_for_cs(x, y)  
    cs = CubicSpline(x, y)
    predict_and_draw(x, y, cs, 'cubic spline')
    

    print('new sample:')
    x, y = generate_sample(n)
    predict_and_draw(x, y, pr.predict, 'polinomial regression', draw=True)

    x, y = reshape_for_cs(x, y)  
    predict_and_draw(x, y, cs, 'cubic spline', draw=True)

    