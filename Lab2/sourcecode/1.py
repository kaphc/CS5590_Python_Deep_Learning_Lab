import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
tb = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
df=pd.read_csv("./weatherHistory.csv")
print(df.info)
X=df.iloc[:,5].values
Y=df.iloc[:,3].values
x=X.reshape(-1,1)
y=Y.reshape(-1,1)
def base_model():
# create model
 model = Sequential()
 model.add(Dense(11, input_dim=1, kernel_initializer='normal', activation='relu'))
 model.add(Dense(1, kernel_initializer='normal'))
# Compile model
 model.compile(loss='mean_squared_error', optimizer='adam')

 return model
numpy.random.seed(10)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=base_model, epochs=2,batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=10)
results = cross_val_score(pipeline, x, y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
