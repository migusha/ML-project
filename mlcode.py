%config Completer.use_jedi = False
# enable code auto-completion
# needed imports
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #data visualization library
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error
#import the data from the .csv
df = pd.read_csv(’StudentPerformanceFactors.csv’)
#check the shape
print(df.shape)
#select the needed columns
df = df[[’Hours_Studied’, ’Attendance’, ’Sleep_Hours’,
’Exam_Score’, ’Motivation_Level’,’Previous_Scores’]]
print(df.head(5))
#put them into different variables for plotting
hours = df[’Hours_Studied’].to_numpy()
att = df[’Attendance’].to_numpy()
sleep = df[’Sleep_Hours’].to_numpy()
exam = df[’Exam_Score’].to_numpy()
moti = df[’Motivation_Level’].to_numpy()
6
prev = df[’Previous_Scores’].to_numpy()
print(moti.shape)
#let 0 be low, 1 be medium and 2 high motivation
#make the motivation levels numbered
nmoti =[]
for m in moti:
if m == ’Low’:
nmoti.append(0)
elif m == ’Medium’:
nmoti.append(1)
else:
nmoti.append(2)
print(nmoti[0:5])
#plot them all in subplots
fig, axes = plt.subplots(2,3, figsize=(15,10))
axes[0,0].scatter(hours, exam)
axes[0,0].set_ylabel(’Exam Score’)
axes[0,0].set_xlabel(’Hours Studied’)
axes[0,1].scatter(sleep, exam, c=’r’)
axes[0,1].set_ylabel(’Exam Score’)
axes[0,1].set_xlabel(’Sleep Hours’)
axes[0,2].scatter(att, exam, c=’g’)
axes[0,2].set_ylabel(’Exam Score’)
axes[0,2].set_xlabel(’Attendace’)
axes[1,0].scatter(nmoti, exam, c=’y’)
axes[1,0].set_ylabel(’Exam Score’)
axes[1,0].set_xlabel(’Motivation Score’)
axes[1,1].scatter(prev, exam, c=’m’)
axes[1,1].set_ylabel(’Exam Score’)
axes[1,1].set_xlabel(’Previous Scores’)
fig.delaxes(axes[1,2])
#save it
fig.savefig(’effects.eps’)
7
#calculate the correlation matrix
cr_df = df.drop(columns=[’Motivation_Level’])
cr_df[’Motivation_numerated’] = nmoti
correlation_matrix = cr_df.corr()
#create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap=’coolwarm’,
vmin=-1, vmax=1, center=0)
plt.title(’Correlation Matrix of Student Performance Features’)
plt.tight_layout()
#save it
plt.savefig(’corr.eps’)
plt.show()
# If you want to focus specifically on correlations
# with Exam_Score
exam_score_correlations = correlation_matrix[
’Exam_Score’].sort_values(ascending=False)
print("Correlations with Exam Score:")
print(exam_score_correlations)
#code for fitting linear regression
#first we need to split our dataset into
#training and validation
X = cr_df.drop(columns=[’Exam_Score’, ’Sleep_Hours’,
’Motivation_numerated’]).to_numpy()
y = cr_df[’Exam_Score’].to_numpy()
regr = LinearRegression()
#X_train, X_val, y_train, y_val = ...
train_test_split(X, y, test_size = 0.3,random_state=1)
X_split, X_test, y_split, y_test = train_test_split(
X, y, test_size = 0.2,train_size = 0.8,random_state=1)
X_train, X_val, y_train, y_val = train_test_split(
X, y, test_size = 0.25,train_size=0.75,random_state=1)
regr.fit(X_train, y_train)
y_pred_train = regr.predict(X_train)
train_error = MSE(y_train, y_pred_train)
print(’Training error:’,train_error)
y_pred_val = regr.predict(X_val)
pred_error = MSE(y_val, y_pred_val)
print(’Prediction error for validation:’,pred_error)
y_pred_test = rgr.predict(X_test)
8
test_error = MSE(y_test, y_pred_test)
print(’Prediction error for test:’,pred_error)
from sklearn.neural_network import MLPRegressor
# We will use the same code as in this courses assignment 3
# but we tweak the number of layers and neurons
# code for neural network approach
## define a list of values for the number of hidden layers
num_layers = [1,2,4,6,8,10,11] # number of hidden layers
num_neurons = 5 # number of neurons in each layer
mlp_tr_errors = []
mlp_val_errors = []
for i, num in enumerate(num_layers):
hidden_layer_sizes = tuple([num_neurons]*num)
mlp_regr = MLPRegressor(
max_iter = 5000,random_state = 1,
hidden_layer_sizes = hidden_layer_sizes)
mlp_regr.fit(X_train,y_train)
y_pred_train = mlp_regr.predict(X_train)
tr_error = mean_squared_error(y_train, y_pred_train)
y_pred_val = mlp_regr.predict(X_val)
val_error = mean_squared_error(y_val, y_pred_val)
mlp_tr_errors.append(tr_error)
mlp_val_errors.append(val_error)
print(mlp_tr_errors)
print(mlp_val_errors)
# plot the training errors
plt.figure(figsize=(8, 6))
plt.plot(num_layers, mlp_tr_errors, label = ’Train’)
plt.plot(num_layers, mlp_val_errors,label = ’Valid’)
plt.xticks(num_layers)
9
plt.legend(loc = ’upper left’)
plt.xlabel(’Layers’)
plt.ylabel(’Loss’)
plt.title(’Train vs validation loss’)
plt.show()
m_errors = {"mlp_train_errors":mlp_tr_errors,
"mlp_val_errors":mlp_val_errors}
pd.DataFrame(m_errors).rename(index={0: "1 layer",
1: "2 layers", 2: "4 layers",3:"6 layers",4:"8 layers",
5:"10 layers",6:"11 layers"})
