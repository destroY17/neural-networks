from sklearn.preprocessing import MinMaxScaler
import numpy as np

scaler = MinMaxScaler()


def get_normalized_data(data):
    """
    Нормализация данных
    :param data: данные
    :return: нормализованные данные
    """
    return scaler.fit_transform(data)


def get_denormalized_data(data):
    return scaler.inverse_transform(data)


def calculate_error(y_values, y_values_from_autoencoder):
    """
    Подсчет и вывод ошибки в консоль
    :param y_values: значения оригинальной функции
    :param y_values_from_autoencoder: значения восстановленной автоэнкодером функции
    """
    mse = np.mean(np.square(y_values - y_values_from_autoencoder))
    print("Mean Squared Error:", mse)
