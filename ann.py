import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
'''from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def bulid_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))   
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = bulid_classifier, batch_size = 10, nb_epoch = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
   ''' 
from sklearn.metrics import confusion_matrix
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
cm = confusion_matrix(y_test, y_pred)
plt.xlabel("Credit Score")
plt.ylabel("Is Exited")
plt.plot(dataset['CreditScore'],dataset['Exited'])
my_plots = plot_partial_dependence(classifier,       
                                   features=[2], # column numbers of plots we want to show
                                   X=X_train,            # raw predictors data.
                                   feature_names=['Credit Score'], # labels on graphs
                                   grid_resolution=10) # number of values to plot on x axis
import matplotlib.pyplot as plt
import pandas
names = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
data = pandas.read_csv('Churn_Modelling.csv', names=names)
data.hist()
plt.show()
names = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
data = pd.read_csv('Churn_Modelling.csv', names=names)
data.hist()
plt.show()

