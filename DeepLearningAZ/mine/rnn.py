import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# %%

training_data = pd.read_csv(
    "./dataset/Part 3 - Recurrent Neural Networks (RNN)/Google_Stock_Price_Train.csv"
)
training_set = training_data.iloc[:, 1:2].values

# %%

sc = MinMaxScaler()
scaled_training_set = sc.fit_transform(training_set)

# %%

X_train = []
y_train = []

for i in range(60, len(training_set)):
    X_train.append(scaled_training_set[i - 60 : i, 0])
    y_train.append(scaled_training_set[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# %%

X_train = np.reshape(X_train, shape=(X_train.shape[0], X_train.shape[1], 1))

# %%

regressor = keras.Sequential(
    [
        keras.Input(shape=(X_train.shape[1], 1)),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(64),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1),
    ]
)

# %%

regressor.compile(optimizer="rmsprop", loss="mean_squared_error")
os.makedirs("models", exist_ok=True)
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="models/model.{epoch:02d}-{loss:.2f}.keras"
    ),
]

# %%

regressor.fit(X_train, y_train, epochs=100, batch_size=32, callbacks=callbacks)

# %%

test_data = pd.read_csv(
    "./dataset/Part 3 - Recurrent Neural Networks (RNN)/Google_Stock_Price_Test.csv"
)
test_set = test_data.iloc[:, 1:2].values

total_data = pd.concat((training_data["Open"], test_data["Open"]), axis=0)
inputs = total_data.loc[len(total_data) - len(test_data) - 60 :].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

# %%

X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i - 60 : i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, shape=(X_test.shape[0], X_test.shape[1], 1))

# %%

pred = regressor.predict(X_test)
pred = sc.inverse_transform(pred)

# %%

plt.plot(test_set, color="red", label="Real price")
plt.plot(pred, color="blue", label="Pred price")
plt.legend()
plt.show()
