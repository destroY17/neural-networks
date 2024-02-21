import tensorflow as tf


def build_autoencoder(input_dimension, encoding_dimension):
    '''
    Создание автоэнкодера.

    :param input_dimension: размерность входящих данных
    :param encoding_dimension: размерность скрытого вектора
    :return: автоэнкодер
    '''
    input_layer = tf.keras.layers.Input(shape=(input_dimension,))

    encoder_layer = tf.keras.layers.Dense(encoding_dimension, activation='relu')(input_layer)
    decoder_layer = tf.keras.layers.Dense(input_dimension, activation='sigmoid')(encoder_layer)

    autoencoder = tf.keras.Model(input_layer, decoder_layer)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder
