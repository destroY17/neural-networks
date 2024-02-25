import tensorflow as tf
import generator as gen


class AutoEncoder(tf.keras.Model):
    def __init__(self, input_dimension, encoding_dimension, layer_dimension):
        super().__init__()

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(layer_dimension, activation='relu'),
            tf.keras.layers.Dense(layer_dimension, activation='relu'),
            tf.keras.layers.Dense(layer_dimension, activation='relu'),
            tf.keras.layers.Dense(encoding_dimension),
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(layer_dimension, activation='relu'),
            tf.keras.layers.Dense(layer_dimension, activation='relu'),
            tf.keras.layers.Dense(layer_dimension, activation='relu'),
            tf.keras.layers.Dense(input_dimension),
        ])

    def call(self, x):
        latent_x = self.encoder(x)
        decoded_x = self.decoder(latent_x)
        return decoded_x


def learn(autoencoder):
    autoencoder.compile(optimizer='adam', loss='mse')
    train_data_normalized = gen.generate_data_to_learn()

    autoencoder.fit(train_data_normalized,
                    train_data_normalized,
                    validation_split=0.2,
                    epochs=1000,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(patience=10)
                    ])
