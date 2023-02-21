import streamlit as st

st.sidebar.title('Neural Network Code Generator')
st.sidebar.header('Preprocess Data:')

# Define input parameters
data_source = st.sidebar.text_input('Data source (URL or file path)', 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv')
y_col = st.sidebar.text_input('Name of the target variable column', 'variety')
id_col = st.sidebar.text_input('Name of the ID column (optional)', '')
ml_type = st.sidebar.radio('Prediction Type', ('Binary Classification', 'Multiple Classification', 'Regression'))

code = f'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_absolute_error, ConfusionMatrixDisplay
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

df = pd.read_csv('{data_source}').dropna()

# Assign target variable
y = pd.DataFrame(df['{y_col}'])
'''
if ml_type == 'Binary Classification':
    code += f'''
y = pd.get_dummies(y, drop_first=True)  
'''
elif ml_type == 'Multiple Classification':
    code += f'''
y = pd.get_dummies(y)
'''

    
code += f'''
# Drop target variable from features
X = df.drop('{y_col}', axis=1)

# Turn all numeric object columns into numeric columns
for col in X.select_dtypes(include='object'):
    try:
        pd.to_numeric(X[col])
        X[col] = pd.to_numeric(X[col])
    except ValueError:
        pass

# Drop highly unique columns
for col in X.columns:
    if X[col].dtype == 'object':
        unique_vals = X[col].nunique()
        unique_ratio = unique_vals / len(X[col])
        if unique_ratio >= 0.8:
            X = X.drop(col, axis=1)

# Make all columns numerical
X = X.pipe(pd.get_dummies)

# Scale data
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
'''

if id_col:
    code += f'''
X = X.drop('{id_col}', axis=1)
'''

st.sidebar.header('Train Test Split Data:')
test_size = st.sidebar.slider('Test size', min_value=0.1, max_value=0.4, value=0.2)

code += f'''
# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state=0)
'''


st.sidebar.header('Neural Network Options:')

num_hidden_layers = st.sidebar.slider('Number of hidden layers', min_value=1, max_value=5, value=2)
num_neurons = st.sidebar.slider('Neurons per hidden layer', min_value=1, max_value=50, value=10)
num_epochs = st.sidebar.slider('Number of epochs', min_value=1, max_value=300, value=100)

use_dropout = st.sidebar.checkbox('Use dropout')
use_batch_norm = st.sidebar.checkbox('Use batch normalization')
use_early_stopping = st.sidebar.checkbox('Use early stopping')

if use_early_stopping:
    validation_split = st.sidebar.slider('Validation split', min_value=0.1, max_value=0.4, value=0.2)
    patience = st.sidebar.slider('Patience', min_value=1, max_value=20, value=5)

# Generate neural network code


    
code += f'''
# Build and compile the neural network
model = Sequential()

# Add input layer
model.add(Dense({num_neurons}, activation='relu', input_dim=X_train.shape[1]))
'''

code += f'''
# Add hidden layers
for i in range({num_hidden_layers}):
    model.add(Dense({num_neurons}, activation='relu'))
'''

if use_dropout:
    code += f'''
    model.add(Dropout(0.5))
    '''

if use_batch_norm:
    code += f'''
    model.add(BatchNormalization())
    '''

code += f'''
# Add output layer
model.add(Dense(y_train.shape[1], activation='{'softmax' if ml_type != 'Regression' else 'linear'}'))

# Define loss variable
'''

if ml_type == 'Regression':
    code += f'''loss_var = 'mse'
'''
if ml_type == 'Binary Classification':
    code += f'''loss_var = 'binary_crossentropy'
'''
elif ml_type == 'Multiple Classification':
    code += f'''loss_var = 'categorical_crossentropy'
'''

if use_early_stopping:
    code += f'''
# Create early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience={patience})

# Compile model
{"model.compile(loss=loss_var, optimizer='adam', metrics=['accuracy'])" if ml_type != 'Regression' else "model.compile(loss=loss_var, optimizer='adam', metrics=['mean_absolute_error'])"}

# Train model
batch_size_var = 32
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0)
history = model.fit(X_train, 
    y_train, 
    epochs=10000, 
    batch_size=batch_size_var, 
    validation_data=(X_val, y_val), 
    callbacks=[early_stopping])
'''
else:
    code += f'''
#Compile model  
model.compile(loss=loss_var, optimizer='adam', metrics=['{'accuracy' if ml_type != 'Regression' else 'mean_absolute_error'}'])

# Train model
batch_size_var = 32
history = model.fit(X, y, epochs={num_epochs}, batch_size=batch_size_var)
'''

code += f'''
# Evaluate the neural network on training data
model.evaluate(X, y)

# Make predictions on testing data
y_pred = model.predict(X_test)

'''
if ml_type == 'Regression':
    code += f'''# Find regression test metrics
r2 = r2_score(y_test, y_pred)
mean_absolute_error = mean_absolute_error(y_test, y_pred)
metric = "r2 is "+str(r2)+" and mean absolute error is "+str(mean_absolute_error)
'''
if ml_type == 'Binary Classification':
    code += f'''# Find binary classification test metrics
accuracy = accuracy_score(y_test, y_pred)
metric = "accuracy is "+str(accuracy)
'''
elif ml_type == 'Multiple Classification':
    code += f'''# Find multiple classification test metrics
y_pred = np.argmax(y_pred, axis=1) 
y_test = np.argmax(np.array(y_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
metric = "accuracy is "+str(accuracy)
'''

code += f'''
print(metric)
'''
include_cm = False

if ml_type != 'Regression':
    include_cm = st.sidebar.checkbox('Include confusion matrix')

if include_cm:
    code += f'''
# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()
'''

st.title('Neural Network Code')
st.write('The code below is generated based on the options you select in the sidebar. Note: this code is not meant to be run as is, but rather as a starting point for your own neural network. There may be some errors in the code, so be sure to check it over before running it.')
st.code(code)

