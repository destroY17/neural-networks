import numpy as np
import train_data_config as tdc
import data_utils as du


def generate_polynomial(count):
    """
    Генерация полиномов.

    :param count: кол-во полиномов для генерации
    :return: полиномы
    """
    polynomials = []
    for _ in range(count):
        coeffs = np.random.rand(tdc.COEFFS_COUNT)
        polynomials.append(np.poly1d(coeffs))

    return polynomials


def generate_random_values(count, x_range):
    """
    Генерация рандомных значений.

    :param count: кол-во значений
    :param x_range: диапазон генерируемых значений
    :return: значения
    """
    min_x, max_x = x_range
    return np.random.uniform(min_x, max_x, count)


def generate_random_values_for_polynomials(polynomials):
    """
    Генерация рандомных значений функций полиномов.

    :param polynomials: полиномы
    :return: значения функций
    """
    y_values = []
    for _ in polynomials:
        y_values.append(generate_random_values(tdc.POINTS_COUNT, tdc.X_RANGE))
    return y_values


def generate_data_to_learn():
    polynomials = generate_polynomial(tdc.POLYNOMIALS_COUNT)

    train_data = generate_random_values_for_polynomials(polynomials)
    train_data = np.array(train_data).reshape(-1, tdc.POINTS_COUNT)
    train_data_normalized = du.get_normalized_data(train_data)

    return train_data_normalized
