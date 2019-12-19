import numpy as np
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

df=pd.read_csv('./candy-data.csv')
df.info()
df.drop("competitorname", inplace = True, axis=1)
y = df.chocolate.values
x_data= df.drop(["chocolate"], axis = 1)
x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=45)
learning_rate=0.1
epochs=10
b_size=64
decay_rate= learning_rate / epochs
adam= Adam(lr=learning_rate, decay=decay_rate)

# Create Model
model = Sequential()
model.add(Dense(12, activation='tanh', input_dim=11))
model.add(Dense(6, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = ["accuracy"])
tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=b_size,callbacks=[tb])

# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

