import numpy as np
import matplotlib.pyplot as plt


def get_fitted_polynomial(x_values, y_values, degree):
    """
    Апроксимация полиномом.

    :param x_values: значения x
    :param y_values: значения y
    :param degree: степень полинома
    :return: функция апроксимации полиномом
    """
    coefficients = np.polyfit(x_values, y_values, degree)
    coefficients = np.array(coefficients).reshape(-1, )
    return np.poly1d(coefficients)


def plot_polynomial(x_values, y_values, degree, name, color):
    """
    Отрисовка графического представления полиномов.

    :param x_values: значения x
    :param y_values: значения y
    :param degree: степень полинома
    :param name: имя для легенды
    :param color: цвет графика
    :return:
    """
    polynom = get_fitted_polynomial(x_values, y_values, degree)

    x_fit = np.linspace(min(x_values), max(x_values), 100)
    y_fit = polynom(x_fit)

    plt.scatter(x_values, y_values, label=f'{name} points')
    plt.plot(x_fit, y_fit, color=f'{color}', label=f'Fitted {name}')
