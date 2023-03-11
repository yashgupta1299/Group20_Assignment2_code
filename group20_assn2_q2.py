import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

## Reading Univariate and Bivariate data first

path_univ = '/Group20/Regression/UnivariateData/20.csv'
path_biv = '/Group20/Regression/BivariateData/20.csv'

df_univ = pd.read_csv(os.getcwd() + path_univ, header=None)
df_biv = pd.read_csv(os.getcwd() + path_biv, header=None)
print('univariate DataFrame')
print(df_univ.head())
print('bivariate DataFrame')
print(df_biv.head())

## Train-test-validation split

def train_test_valid_split(df):
    df = df.sample(frac=1).reset_index(drop=True)  
    train = df.iloc[: int(0.6*len(df)), :]
    validation = df.iloc[int(0.6*len(df)): int(0.8*len(df)), :]
    test = df.iloc[int(0.8*len(df)): , :]
    return train, validation, test

univ_train, univ_valid, univ_test = train_test_valid_split(df_univ)
print(univ_train.shape, univ_valid.shape, univ_test.shape)
# bivariate
biv_train, biv_valid, biv_test = train_test_valid_split(df_biv)
print(biv_train.shape, biv_valid.shape, biv_test.shape)

# separating ground truth and inputs for univariate case
x_train_univ, y_train_univ = univ_train.iloc[:,:1].values, univ_train.iloc[:,1].values 
x_valid_univ, y_valid_univ = univ_valid.iloc[:,:1].values, univ_valid.iloc[:,1].values
x_test_univ, y_test_univ = univ_test.iloc[:,:1].values, univ_test.iloc[:,1].values
plt.scatter(x_train_univ, y_train_univ)
plt.title('univariate training set plot')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()

# separating ground truth and inputs for bivariate case
x_train_biv, y_train_biv = biv_train.iloc[:,:2].values, biv_train.iloc[:,2].values 
x_valid_biv, y_valid_biv = biv_valid.iloc[:,:2].values, biv_valid.iloc[:,2].values
x_test_biv, y_test_biv = biv_test.iloc[:,:2].values, biv_test.iloc[:,2].values
ax = plt.axes(projection='3d')
ax.scatter3D(x_train_biv[:,0],x_train_biv[:,1],y_train_biv, alpha=0.8,  marker='*', label='actual data')
ax.set_xlabel('x-axis', fontsize=20)
ax.set_ylabel('y-axis', fontsize=20)
ax.set_zlabel('z-axis', fontsize=20)
plt.title("bivariate training set plot", fontsize=15)
plt.legend()
plt.show()

## Writing the Model for regression with user-defined hidden layers
#Activation functions and their derivatives
#logistic function
def logistic_fun(x):
    return 1/(1 + np.exp(-x))
def deriv_logistic_fun(x):
    return logistic_fun(x)*(1 - logistic_fun(x))

# Hyberbolic Tangent Function
def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
def deriv_tanh(x):
    return (1 - np.square(tanh(x)))

# Linear Activation 
def linear_fun(x):
    return x
def deriv_linear_fun(x):
    return 1

#perform forward computation for randomly generated parameters
def forward_computation(example, weight_matrix, bias_matrix):
    # The activation values in each layer of forward computation needs to be stored. 
    # As we'll need it in backpropagation step. It should be 2D numpy array.
    
    y_pred = example.reshape(-1,1)
    activations = [y_pred] 
    for i in range(len(weight_matrix)):
        y_pred = np.matmul(weight_matrix[i].T,y_pred) + bias_matrix[i]
        activations.append(y_pred)
        # activation step
        if i != len(weight_matrix) - 1:
            y_pred = logistic_fun(y_pred) # in regression output layer has linear activation function
        
#         print(y_pred, "iteration{}".format(i))
    return y_pred, activations
                           
        
        
# backpropagation would be done in two steps
# 1. calculation of delta values
# 2. updation of weight and biase using delta

#This backpropagation is customised only for regression.
def backpropagation(y_pred, y_act, activation_values, weights_matrix, bias_matrix, lr = 0.001):
    # make y_pred and y_act of same dimension
    y_act = np.array(y_act).reshape(-1,1)
    correction_values = []
    
    # as the name suggest we need to propagate the error in backward direction
    del_out = (y_act - y_pred) # dE_dw for last layer 
    w_size = len(weights_matrix)
    for i in range(w_size):
        if i != 0 :
            del_out = np.matmul(weights_matrix[w_size - i], del_out)*(deriv_logistic_fun(activation_values[w_size - i]))
        del_b_out = del_out*lr # bias correction factor
        if i != w_size - 1:
            del_w_out = np.matmul(logistic_fun(activation_values[w_size - 1 - i]), del_out.T)*lr
        else: #if i == w_size - 1:
            del_w_out = np.matmul(activation_values[w_size - 1 - i], del_out.T)*lr # weight correction factor
        # store the correction values of each layer
        correction_values.append([del_w_out, del_b_out])
#     for x in correction_values:
#         print("****************** layer last to 1")
#         print(x[0].shape)
#         print(x[1].shape)
    # Step:2 Update weights
    for i in range(len(correction_values)):
        weights_matrix[i] += correction_values[len(correction_values) - i - 1][0]
        bias_matrix[i] += correction_values[len(correction_values) - i - 1][1]
    return weights_matrix, bias_matrix
        
            
        
## Combining all the functions above, writing a regression function
### We'll make a module/(OOPs implementation of regression) later
        
def fit_regressor(inputs_: int, hidden_layer : list, output: int, x_train, y_train,  epochs: int, lr = 0.001):
    # x_valid, y_valid,
    #1. Initialization Step
    network_arch = ([inputs_] + hidden_layer + [output])
    weight_matrix = [] #list of matrices
    bias_matrix = [] # list of bias matrix or bias vector
    mu, sigma = 0, 1 # mean and standard deviation of normal distribution

    for l in range(len(network_arch)-1):
        w_i = np.random.normal(mu, sigma, (network_arch[l],network_arch[l+1]))
        b_i = np.random.normal(mu, sigma, (network_arch[l+1],1))
        weight_matrix.append(w_i)
        bias_matrix.append(b_i)
    
    #2. 
    prev_epoch_err = np.inf
    squarred_avg_err = [] # stores average MSE of each epoch
    for epoch in range(epochs):
        np.random.seed(123)
        # generate a random permutation of the indices
        idx = np.random.permutation(len(x_train))
        
        # shuffle the arrays using the same indices
        x_train = x_train[idx]
        y_train = y_train[idx]
        
        epoch_err = 0
        for i in range(len(x_train)):
            #(a) do forward computation
            y_pred, layer_outputs = forward_computation(x_train[i], weight_matrix, bias_matrix)
            
            #(b) compute instantaneous error
            ins_err = np.sum(np.square(y_pred - y_train[i].reshape(-1,1)))/2 
            epoch_err += ins_err
            
            #(c) perform backpropagation and update weights
            weight_matrix, bias_matrix = backpropagation(y_pred, y_train[i], layer_outputs, weight_matrix, bias_matrix, lr)
        
        # checking for convergence criteria
        avg_epoch_err = epoch_err/len(x_train)
        squarred_avg_err.append(avg_epoch_err)
        if abs(prev_epoch_err - avg_epoch_err) < 0.00001:
            print(f'---------------- Convergance criteria has been satisfied ------- breaking execution loop at {epoch+1}th iteration -----')
            return {'weights': weight_matrix, "bias": bias_matrix, 'avg_err': squarred_avg_err}
        prev_epoch_err = avg_epoch_err
    return {'weights': weight_matrix, "bias": bias_matrix, 'avg_err': squarred_avg_err}
        
## Let's use our Model to predict on validation and test data
    
def predict_one(example, weights, bias):
    y_pred, _ = forward_computation(example, weights, bias)
    return y_pred

def predict_many(x_inputs, weights, bias):
    y_preds = []
    for i in range(len(x_inputs)):
        y_preds.append(predict_one(x_inputs[i], weights, bias))
    y_preds = np.array(y_preds).reshape(-1,1)
    return y_preds

def MSE(y_actual, y_pred):
    y_actual = y_actual.reshape(-1,1)
    mean_squarred_error = np.sum(np.square(y_actual-y_pred))/len(y_actual)
    return mean_squarred_error

def RMSE(y_actual, y_pred):
    return np.sqrt(MSE(y_actual, y_pred))


## helper function for visualisation
def model_vs_actual(x_inputs, y_act, y_pred, title=""):
    plt.scatter(x_inputs, y_act, label='Actual Data',edgecolors='black')
    plt.scatter(x_inputs, y_pred, label='Model Output',edgecolors='black')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.legend()
    plt.title(title)
    plt.show()
    return


## univariate Case with 1 and 2 Hidden Layers
## $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ fcnn1_univ $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# checking our model for univariate case

print('\nFor univariate case: with [3]:-> hidden layer configuration--------------\n')
fcnn1_univ = fit_regressor(1,[3], 1, x_train_univ, y_train_univ, 250)
err = fcnn1_univ['avg_err']
x_ = np.arange(len(err))
plt.plot(x_, err)
plt.title('MSE vs epochs for univariate (training)')
plt.xlabel('no. of epochs')
plt.ylabel('MSE error')
plt.show()

# train
pred1_univ = predict_many(x_train_univ, fcnn1_univ['weights'], fcnn1_univ['bias'])
mse = MSE(y_train_univ, pred1_univ)
rmse = RMSE(y_train_univ, pred1_univ)
print('**For training data************************')
print('---------- Mean squarred Error for univariate ----------')
print(mse)
print('--------- Root Mean squarred Error for univariate  -----------')
print(rmse)
# print('--------------------')
model_vs_actual(x_train_univ, y_train_univ, pred1_univ, "Model vs Actual (training data)")

# validation data
pred1_univ = predict_many(x_valid_univ, fcnn1_univ['weights'], fcnn1_univ['bias'])
mse = MSE(y_valid_univ, pred1_univ)
rmse = RMSE(y_valid_univ, pred1_univ)
print('**For validation data***********************')
print('---------- Mean squarred Error for univariate ----------')
print(mse)
print('--------- Root Mean squarred Error for univariate  -----------')
print(rmse)
# print('--------------------')
model_vs_actual(x_valid_univ, y_valid_univ, pred1_univ, "Model vs Actual (validation data)")

# test
pred1_univ = predict_many(x_test_univ, fcnn1_univ['weights'], fcnn1_univ['bias'])
mse = MSE(y_test_univ, pred1_univ)
rmse = RMSE(y_test_univ, pred1_univ)
print('**For test data**********************************')
print('---------- Mean squarred Error for univariate ----------')
print(mse)
print('--------- Root Mean squarred Error for univariate  -----------')
print(rmse)
# print('--------------------')
model_vs_actual(x_test_univ, y_test_univ, pred1_univ, "Model vs Actual (test data)")

## $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ fcnn2_univ $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

print('\nFor univariate case: with [8]:-> hidden layer configuration-------------------\n')

# checking our model for univariate case
fcnn2_univ = fit_regressor(1,[8], 1, x_train_univ, y_train_univ, 250)
err = fcnn2_univ['avg_err']
x_ = np.arange(len(err))
plt.plot(x_, err)
plt.title('MSE vs epochs for univariate (training)')
plt.xlabel('no. of epochs')
plt.ylabel('MSE error')
plt.show()

# train
pred1_univ = predict_many(x_train_univ, fcnn2_univ['weights'], fcnn2_univ['bias'])
mse = MSE(y_train_univ, pred1_univ)
rmse = RMSE(y_train_univ, pred1_univ)
print('**For training data************************')
print('---------- Mean squarred Error for univariate ----------')
print(mse)
print('--------- Root Mean squarred Error for univariate  -----------')
print(rmse)
# print('--------------------')
model_vs_actual(x_train_univ, y_train_univ, pred1_univ, "Model vs Actual (training data)")

# validation data
pred1_univ = predict_many(x_valid_univ, fcnn2_univ['weights'], fcnn2_univ['bias'])
mse = MSE(y_valid_univ, pred1_univ)
rmse = RMSE(y_valid_univ, pred1_univ)
print('**For validation data***********************')
print('---------- Mean squarred Error for univariate ----------')
print(mse)
print('--------- Root Mean squarred Error for univariate  -----------')
print(rmse)
# print('--------------------')
model_vs_actual(x_valid_univ, y_valid_univ, pred1_univ, "Model vs Actual (validation data)")


# test
pred1_univ = predict_many(x_test_univ, fcnn2_univ['weights'], fcnn2_univ['bias'])
mse = MSE(y_test_univ, pred1_univ)
rmse = RMSE(y_test_univ, pred1_univ)
print('**For test data**********************************')
print('---------- Mean squarred Error for univariate ----------')
print(mse)
print('--------- Root Mean squarred Error for univariate  -----------')
print(rmse)
# print('--------------------')
model_vs_actual(x_test_univ, y_test_univ, pred1_univ, "Model vs Actual (test data)")


### ************************************************ fcnn3_univ ********************************************************

print('\nFor univariate case: with [5]:-> hidden layer configuration-------------------\n')

# checking our model for univariate case
fcnn3_univ = fit_regressor(1,[5], 1, x_train_univ, y_train_univ, 250)
err = fcnn3_univ['avg_err']
x_ = np.arange(len(err))
plt.plot(x_, err)
plt.title('MSE vs epochs for univariate (training)')
plt.xlabel('no. of epochs')
plt.ylabel('MSE error')
plt.show()

# train
pred1_univ = predict_many(x_train_univ, fcnn3_univ['weights'], fcnn3_univ['bias'])
mse = MSE(y_train_univ, pred1_univ)
rmse = RMSE(y_train_univ, pred1_univ)
print('**For training data************************')
print('---------- Mean squarred Error for univariate ----------')
print(mse)
print('--------- Root Mean squarred Error for univariate  -----------')
print(rmse)
# print('--------------------')
model_vs_actual(x_train_univ, y_train_univ, pred1_univ, "Model vs Actual (training data)")

# validation data
pred1_univ = predict_many(x_valid_univ, fcnn3_univ['weights'], fcnn3_univ['bias'])
mse = MSE(y_valid_univ, pred1_univ)
rmse = RMSE(y_valid_univ, pred1_univ)
print('**For validation data***********************')
print('---------- Mean squarred Error for univariate ----------')
print(mse)
print('--------- Root Mean squarred Error for univariate  -----------')
print(rmse)
# print('--------------------')
model_vs_actual(x_valid_univ, y_valid_univ, pred1_univ, "Model vs Actual (validation data)")


# test
pred1_univ = predict_many(x_test_univ, fcnn3_univ['weights'], fcnn3_univ['bias'])
mse = MSE(y_test_univ, pred1_univ)
rmse = RMSE(y_test_univ, pred1_univ)
print('**For test data**********************************')
print('---------- Mean squarred Error for univariate ----------')
print(mse)
print('--------- Root Mean squarred Error for univariate  -----------')
print(rmse)
# print('--------------------')
model_vs_actual(x_test_univ, y_test_univ, pred1_univ, "Model vs Actual (test data)")


### ************************************************ fcnn4_univ ********************************************************


## Univariate case with 2 Hidden layer
print('\nFor univariate case: with [14,8]:-> hidden layer configuration-------------------\n')

# checking our model for univariate case
fcnn4_univ = fit_regressor(1,[14,8], 1, x_train_univ, y_train_univ, 250)
err = fcnn4_univ['avg_err']
x_ = np.arange(len(err))
plt.plot(x_, err)
plt.title('MSE vs epochs for univariate (training)')
plt.xlabel('no. of epochs')
plt.ylabel('MSE error')
plt.show()

# train
pred1_univ = predict_many(x_train_univ, fcnn4_univ['weights'], fcnn4_univ['bias'])
mse = MSE(y_train_univ, pred1_univ)
rmse = RMSE(y_train_univ, pred1_univ)
print('**For training data************************')
print('---------- Mean squarred Error for univariate ----------')
print(mse)
print('--------- Root Mean squarred Error for univariate  -----------')
print(rmse)
# print('--------------------')
model_vs_actual(x_train_univ, y_train_univ, pred1_univ, "Model vs Actual (training data) 2 Hidden Layer")

# validation data
pred1_univ = predict_many(x_valid_univ, fcnn4_univ['weights'], fcnn4_univ['bias'])
mse = MSE(y_valid_univ, pred1_univ)
rmse = RMSE(y_valid_univ, pred1_univ)
print('**For validation data***********************')
print('---------- Mean squarred Error for univariate ----------')
print(mse)
print('--------- Root Mean squarred Error for univariate  -----------')
print(rmse)
# print('--------------------')
model_vs_actual(x_valid_univ, y_valid_univ, pred1_univ, "Model vs Actual (validation data) 2 Hidden Layer")

# test
pred1_univ = predict_many(x_test_univ, fcnn4_univ['weights'], fcnn4_univ['bias'])
mse = MSE(y_test_univ, pred1_univ)
rmse = RMSE(y_test_univ, pred1_univ)
print('**For test data**********************************')
print('---------- Mean squarred Error for univariate ----------')
print(mse)
print('--------- Root Mean squarred Error for univariate  -----------')
print(rmse)
# print('--------------------')
model_vs_actual(x_test_univ, y_test_univ, pred1_univ, "Model vs Actual (test data) 2 Hidden Layer")


## Bivariate Case with 1 and 2 Hidden Layers


# helper function for 3D visualisation of data and Model
def plot3d_model_actual(x_inputs, y_act, y_pred, title=''):
    ax = plt.axes(projection='3d')
    ax.scatter3D(x_inputs[:,0],x_inputs[:,1],y_act, alpha=0.8,  marker='*', label='Actual Data')
    ax.scatter3D(x_inputs[:,0],x_inputs[:,1],y_pred, alpha=0.8,  marker='*', label='Model Output')
    ax.set_xlabel('x-axis', fontsize=20)
    ax.set_ylabel('y-axis', fontsize=20)
    ax.set_zlabel('z-axis', fontsize=20)
    plt.title(title, fontsize=15)
    plt.legend()
    plt.show()
    return

### ************************************************ fcnn1_biv ********************************************************
print('\nFor bivariate case: with [15]:-> hidden layer configuration-------------------\n')

# checking our model for bivariate case
fcnn1_biv = fit_regressor(2,[15], 1, x_train_biv, y_train_biv, 250)
err = fcnn1_biv['avg_err']
x_ = np.arange(len(err))
plt.plot(x_, err)
plt.title('MSE vs epochs for bivariate (training) 1 Hidden Layer')
plt.xlabel('no. of epochs')
plt.ylabel('MSE error')
plt.show()

# train
pred1_biv = predict_many(x_train_biv, fcnn1_biv['weights'], fcnn1_biv['bias'])
mse = MSE(y_train_biv, pred1_biv)
rmse = RMSE(y_train_biv, pred1_biv)
print('**For training data************************')
print('---------- Mean squarred Error for bivariate ----------')
print(mse)
print('--------- Root Mean squarred Error for bivariate  -----------')
print(rmse)
# print('--------------------')
plot3d_model_actual(x_train_biv, y_train_biv, pred1_biv, "Model vs Actual (training data) 1 Hidden Layer")

# validation
pred1_biv = predict_many(x_valid_biv, fcnn1_biv['weights'], fcnn1_biv['bias'])
mse = MSE(y_valid_biv, pred1_biv)
rmse = RMSE(y_valid_biv, pred1_biv)
print('**For validation data***********************')
print('---------- Mean squarred Error for bivariate ----------')
print(mse)
print('--------- Root Mean squarred Error for bivariate  -----------')
print(rmse)
# print('--------------------')
plot3d_model_actual(x_valid_biv, y_valid_biv, pred1_biv, "Model vs Actual (validation data) 1 Hidden Layer")

# test
pred1_biv = predict_many(x_test_biv, fcnn1_biv['weights'], fcnn1_biv['bias'])
mse = MSE(y_test_biv, pred1_biv)
rmse = RMSE(y_test_biv, pred1_biv)
print('**For test data**********************************')
print('---------- Mean squarred Error for bivariate ----------')
print(mse)
print('--------- Root Mean squarred Error for bivariate  -----------')
print(rmse)
# print('--------------------')
plot3d_model_actual(x_test_biv, y_test_biv, pred1_biv, "Model vs Actual (test data) 1 Hidden Layer")


### ************************************************ fcnn2_biv ********************************************************

print('\nFor bivariate case: with [18]:-> hidden layer configuration-------------------\n')

# checking our model for bivariate case
fcnn2_biv = fit_regressor(2,[18], 1, x_train_biv, y_train_biv, 250)
err = fcnn2_biv['avg_err']
x_ = np.arange(len(err))
plt.plot(x_, err)
plt.title('MSE vs epochs for bivariate (training) 1 Hidden Layer')
plt.xlabel('no. of epochs')
plt.ylabel('MSE error')
plt.show()

# train
pred1_biv = predict_many(x_train_biv, fcnn2_biv['weights'], fcnn2_biv['bias'])
mse = MSE(y_train_biv, pred1_biv)
rmse = RMSE(y_train_biv, pred1_biv)
print('**For training data************************')
print('---------- Mean squarred Error for bivariate ----------')
print(mse)
print('--------- Root Mean squarred Error for bivariate  -----------')
print(rmse)
# print('--------------------')
plot3d_model_actual(x_train_biv, y_train_biv, pred1_biv, "Model vs Actual (training data) 1 Hidden Layer")

# validation
pred1_biv = predict_many(x_valid_biv, fcnn2_biv['weights'], fcnn2_biv['bias'])
mse = MSE(y_valid_biv, pred1_biv)
rmse = RMSE(y_valid_biv, pred1_biv)
print('**For validation data***********************')
print('---------- Mean squarred Error for bivariate ----------')
print(mse)
print('--------- Root Mean squarred Error for bivariate  -----------')
print(rmse)
# print('--------------------')
plot3d_model_actual(x_valid_biv, y_valid_biv, pred1_biv, "Model vs Actual (validation data) 1 Hidden Layer")

# test
pred1_biv = predict_many(x_test_biv, fcnn2_biv['weights'], fcnn2_biv['bias'])
mse = MSE(y_test_biv, pred1_biv)
rmse = RMSE(y_test_biv, pred1_biv)
print('**For test data**********************************')
print('---------- Mean squarred Error for bivariate ----------')
print(mse)
print('--------- Root Mean squarred Error for bivariate  -----------')
print(rmse)
# print('--------------------')
plot3d_model_actual(x_test_biv, y_test_biv, pred1_biv, "Model vs Actual (test data) 1 Hidden Layer")

### With 2 Hidden Layer

### ************************************************ fcnn3_biv ********************************************************

print('\nFor bivariate case: with [15, 18]:-> hidden layer configuration-------------------\n')

# checking our model for bivariate case
fcnn3_biv = fit_regressor(2,[15, 18], 1, x_train_biv, y_train_biv, 250)
err = fcnn3_biv['avg_err']
x_ = np.arange(len(err))
plt.plot(x_, err)
plt.title('MSE vs epochs for bivariate (training) 2 Hidden Layer')
plt.xlabel('no. of epochs')
plt.ylabel('MSE error')
plt.show()

# train
pred1_biv = predict_many(x_train_biv, fcnn3_biv['weights'], fcnn3_biv['bias'])
mse = MSE(y_train_biv, pred1_biv)
rmse = RMSE(y_train_biv, pred1_biv)
print('**For training data************************')
print('---------- Mean squarred Error for bivariate ----------')
print(mse)
print('--------- Root Mean squarred Error for bivariate  -----------')
print(rmse)
# print('--------------------')
plot3d_model_actual(x_train_biv, y_train_biv, pred1_biv, "Model vs Actual (training data) 2 Hidden Layer")

# validation
pred1_biv = predict_many(x_valid_biv, fcnn3_biv['weights'], fcnn3_biv['bias'])
mse = MSE(y_valid_biv, pred1_biv)
rmse = RMSE(y_valid_biv, pred1_biv)
print('**For validation data***********************')
print('---------- Mean squarred Error for bivariate ----------')
print(mse)
print('--------- Root Mean squarred Error for bivariate  -----------')
print(rmse)
# print('--------------------')
plot3d_model_actual(x_valid_biv, y_valid_biv, pred1_biv, "Model vs Actual (validation data) 2 Hidden Layer")

# test
pred1_biv = predict_many(x_test_biv, fcnn3_biv['weights'], fcnn3_biv['bias'])
mse = MSE(y_test_biv, pred1_biv)
rmse = RMSE(y_test_biv, pred1_biv)
print('**For test data**********************************')
print('---------- Mean squarred Error for bivariate ----------')
print(mse)
print('--------- Root Mean squarred Error for bivariate  -----------')
print(rmse)
# print('--------------------')
plot3d_model_actual(x_test_biv, y_test_biv, pred1_biv, "Model vs Actual (test data) 2 Hidden Layer")


### ************************************************ fcnn4_biv ********************************************************

print('\nFor bivariate case: with [24, 17]:-> hidden layer configuration-------------------\n')

# checking our model for bivariate case
fcnn4_biv = fit_regressor(2,[24, 17], 1, x_train_biv, y_train_biv, 250)
err = fcnn4_biv['avg_err']
x_ = np.arange(len(err))
plt.plot(x_, err)
plt.title('MSE vs epochs for bivariate (training) 2 Hidden Layer')
plt.xlabel('no. of epochs')
plt.ylabel('MSE error')
plt.show()

# train
pred1_biv = predict_many(x_train_biv, fcnn4_biv['weights'], fcnn4_biv['bias'])
mse = MSE(y_train_biv, pred1_biv)
rmse = RMSE(y_train_biv, pred1_biv)
print('**For training data************************')
print('---------- Mean squarred Error for bivariate ----------')
print(mse)
print('--------- Root Mean squarred Error for bivariate  -----------')
print(rmse)
# print('--------------------')
plot3d_model_actual(x_train_biv, y_train_biv, pred1_biv, "Model vs Actual (training data) 2 Hidden Layer")

# validation
pred1_biv = predict_many(x_valid_biv, fcnn4_biv['weights'], fcnn4_biv['bias'])
mse = MSE(y_valid_biv, pred1_biv)
rmse = RMSE(y_valid_biv, pred1_biv)
print('**For validation data***********************')
print('---------- Mean squarred Error for bivariate ----------')
print(mse)
print('--------- Root Mean squarred Error for bivariate  -----------')
print(rmse)
# print('--------------------')
plot3d_model_actual(x_valid_biv, y_valid_biv, pred1_biv, "Model vs Actual (validation data) 2 Hidden Layer")


# test
pred1_biv = predict_many(x_test_biv, fcnn4_biv['weights'], fcnn4_biv['bias'])
mse = MSE(y_test_biv, pred1_biv)
rmse = RMSE(y_test_biv, pred1_biv)
print('**For test data**********************************')
print('---------- Mean squarred Error for bivariate ----------')
print(mse)
print('--------- Root Mean squarred Error for bivariate  -----------')
print(rmse)
# print('--------------------')
plot3d_model_actual(x_test_biv, y_test_biv, pred1_biv, "Model vs Actual (test data) 2 Hidden Layer")



