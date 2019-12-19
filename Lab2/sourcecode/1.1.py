import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from keras.optimizers import SGD, Adam, Adamax
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
from sklearn.preprocessing import LabelEncoder
from keras import metrics
import matplotlib.pyplot as plt

dataset = pd.read_csv('./weatherHistory.csv')



# split into input (X) and output (Y) variables

X = dataset.iloc[:,5].values.reshape(-1,1)
Y = dataset.iloc[:,3].values.reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                   test_size=0.25, random_state=100)
#learning_rate = 0.3
learning_rate = 0.7
epochs = 5
b_size = 32
#b_size = 56
decay_rate = learning_rate / epochs

#adam= Adam(lr=learning_rate, decay=decay_rate)
sgd = SGD(lr=learning_rate, decay=decay_rate)

# Define the model
model = Sequential()
model.add(Dense(11, input_dim=1, activation='tanh'))
model.add(Dense(7, activation='tanh'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

model.compile(optimizer="sgd", loss='mean_squared_error', metrics=[metrics.mae])
tb = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=b_size,
                 callbacks=[tb])

# Final evaluation of the model
mae, loss = model.evaluate(X_test, Y_test, verbose=0)
print(mae, loss)