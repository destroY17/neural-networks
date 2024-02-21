import matplotlib.pyplot as plt
import numpy as np
import autoencoder as ae
import train_data_config as tdc
import autoencoder_config as ac
import polynomial_plotter as pp
import generator as gen
from sklearn.preprocessing import MinMaxScaler


def get_normalized_data(data):
    """
    Нормализация данных
    :param data: данные
    :return: нормализованные данные
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)


def calculate_error(y_values, y_values_from_autoencoder):
    """
    Подсчет и вывод ошибки в консоль
    :param y_values: значения оригинальной функции
    :param y_values_from_autoencoder: значения восстановленной автоэнкодером функции
    """
    mse = np.mean(np.square(y_values - y_values_from_autoencoder))
    print("Mean Squared Error:", mse)


def main():
    """
    Тестирование работы автоэнкодера.
    Обучение автоэнкодера, проверка его работы на тестовых данных,
    графическое представление оригинальных данных и восстановленных автоэнкодером.
    """
    # Генерация полиномов
    polynomials = gen.generate_polynomial(tdc.POLYNOMIALS_COUNT)

    # Генерация данных для обучения
    train_data = gen.generate_random_values_for_polynomials(polynomials)
    train_data = np.array(train_data).reshape(-1, tdc.POINTS_COUNT)
    train_data_normalized = get_normalized_data(train_data)

    # Создание автоэнкодера
    input_dimension = train_data_normalized.shape[1]
    autoencoder = ae.build_autoencoder(input_dimension, ac.ENCODING_DIMENSION)

    autoencoder.fit(train_data_normalized,
                    train_data_normalized,
                    epochs=50,
                    batch_size=32,
                    shuffle=True)

    # Генерация данных для теста
    x_values = gen.generate_random_values(tdc.POINTS_COUNT, tdc.X_RANGE)

    y_values = gen.generate_random_values(tdc.POINTS_COUNT, (-10, 10))
    y_values = np.array(y_values).reshape(-1, 1)
    y_values = get_normalized_data(y_values)

    # Отрисовка данных теста
    pp.plot_polynomial(x_values, y_values, tdc.DEGREE, 'Original', 'red')

    # Получение данных из автоэнкодера
    y_values = np.array(y_values).reshape(-1, tdc.POINTS_COUNT)
    y_values_from_autoencoder = autoencoder.predict(y_values)
    y_values_from_autoencoder = np.array(y_values_from_autoencoder).reshape(-1, 1)

    # Отрисовка данных из автоэнкодера
    pp.plot_polynomial(x_values, y_values_from_autoencoder, tdc.DEGREE, 'Decoded', 'blue')

    plt.legend()
    plt.show()

    # Подсчет ошибки
    calculate_error(y_values, y_values_from_autoencoder)


if __name__ == "__main__":
    main()
