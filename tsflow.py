import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


x = tf.constant([[2.0]], [[5.0]], [[6.0]])
y = tf.constant([[4.0]], [[10.0]], [[12.0]])


model = keras.Sequential([
    layers.Dense(5, activation="linear", input_shape=(1,))
])

model.compile(optimizer="adam", loss="mean_squared_error")

model.fit(x, y, epochs=20, batch_size=3, verbose=2, validation_data=None)


print(model.predict([[2.0]]))

