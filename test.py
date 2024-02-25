import matplotlib.pyplot as plt
import numpy as np
import autoencoder as ae
import train_data_config as tdc
import autoencoder_config as ac
import polynomial_plotter as pp
import generator as gen
import data_utils as du


def generate_test_values():
    x_values = gen.generate_random_values(tdc.POINTS_COUNT, tdc.X_RANGE)

    y_values = gen.generate_random_values(tdc.POINTS_COUNT, tdc.X_RANGE)
    y_values = np.array(y_values).reshape(-1, 1)

    return x_values, y_values


def test():
    """
    Тестирование работы автоэнкодера.
    Обучение автоэнкодера, проверка его работы на тестовых данных,
    графическое представление оригинальных данных и восстановленных автоэнкодером.
    """
    # Подготовка
    (x_values, y_values) = generate_test_values()

    # Действие
    autoencoder = ae.AutoEncoder(input_dimension=tdc.POINTS_COUNT,
                                 encoding_dimension=ac.ENCODING_DIMENSION,
                                 layer_dimension=ac.LAYER_DIMENSION)
    ae.learn(autoencoder)

    y_values_for_autoencoder = du.get_normalized_data(y_values)
    y_values_for_autoencoder = np.array(y_values_for_autoencoder).reshape(-1, tdc.POINTS_COUNT)

    y_values_from_autoencoder = autoencoder.predict(y_values_for_autoencoder)
    y_values_from_autoencoder = np.array(y_values_from_autoencoder).reshape(-1, 1)
    y_values_from_autoencoder = du.get_denormalized_data(y_values_from_autoencoder)

    # Проверка
    pp.plot_polynomial(x_values, y_values, tdc.DEGREE, 'Original', 'red')
    pp.plot_polynomial(x_values, y_values_from_autoencoder, tdc.DEGREE, 'Decoded', 'blue')

    plt.legend()
    plt.show()

    du.calculate_error(y_values, y_values_from_autoencoder)


test()
