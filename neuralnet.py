from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor as nn
import deltapv as dpv
from jax import numpy as jnp
from jax import grad, value_and_grad
import numpy as np
import jax

'''
This model grabs data and predicts the material properties then feeds it into 
the dpv solver to get the predicted result. Then, performs back propagation to
find the changes in the material properties that get it closer to expected
result and from that finds the changes in the model to get it closer to changed values of
material properties.

'''


#####################
#    Solving dpv    #
#####################
def create_design(params):
    '''
    Given parameters that form a solar cell, create a deltapv object that
    represents the design 
    '''
    L = 3e-4
    J = 5e-6
    Chi=params[0]
    Eg=params[1]
    eps=params[2],
    Nc=params[3],
    Nv=params[4],
    mn=params[5],
    mp=params[6],
    Et=params[7],
    tn=params[8],
    tp=params[9],
    A=params[10]


    material = dpv.create_material(Eg=Eg,
                            Chi=Chi,
                            eps=eps,
                            Nc=Nc,
                            Nv=Nv,
                            mn=mn,
                            mp=mp,
                            Et=Et,
                            tn=tn,
                            tp=tp,
                            A=A)


    des = dpv.make_design(n_points=500,
                                Ls=[J, L - J],
                                mats=[material, material],
                                Ns=[1e17, -1e15],
                                Snl=1e7,
                                Snr=0,
                                Spl=0,
                                Spr=1e7)
    return des



def f(params):
    '''
    Given a set of params that construct a solar cell, returns the efficiency 
    of that solar cell
    '''
    des = create_design(params)
    results = dpv.simulate(des, verbose=False)
    eff = results["eff"] * 100
    return eff



# df is a tuple of f and the gradient of f, two functions
df = value_and_grad(f)



def f_np(x):
    '''
    f_np(x) takes x, a set of parameters representing material properties,
    and returns the efficiency and gradient of efficiency with respect
    to each property
    '''
    y, dy = df(x)
    result = float(y), np.array(dy)
    return result


##########################
#    Data Formatting     #
##########################
mats = []




#####################
#    Neural Net     #
#####################
def InitializeWeights(layer_sizes, seed):
    weights = []

    for i, units in enumerate(layer_sizes):
        if i==0:
            w = jax.random.uniform(key=seed, shape=(units, features), minval=-1.0, maxval=1.0, dtype=jnp.float32)
        else:
            w = jax.random.uniform(key=seed, shape=(units, layer_sizes[i-1]), minval=-1.0, maxval=1.0,
                                   dtype=jnp.float32)

        b = jax.random.uniform(key=seed, minval=-1.0, maxval=1.0, shape=(units,), dtype=jnp.float32)

        weights.append([w,b])

    return weights


def Relu(X):
    '''
    Activation function that returns max(0,x) for all x in X
    '''
    return jnp.maximum(X, jnp.zeros_like(X))



def LinearLayer(layer_weights, input_data, activation=lambda x: x):
    '''
    Computes one layer of the neural network, to be used in ForwardPass
    '''
    w, b = layer_weights
    out = jnp.dot(input_data, w.T) + b
    return activation(out)



def ForwardPass(weights, input_data):
    '''
    Passes the data through the neural network and returns the output params
    '''
    layer_out = input_data

    for i in range(len(weights[:-1])):
        layer_out = LinearLayer(weights[i], layer_out, Relu)

    preds = LinearLayer(weights[-1], layer_out)

    return preds.squeeze()



def MeanSquaredErrorLoss(weights, input_data, actual):
    preds = ForwardPass(weights, input_data)
    return jnp.power(actual - preds, 2).mean()



def CalculateGradients(weights, input_data, actual):
    Grad_MSELoss = grad(MeanSquaredErrorLoss)
    gradients = Grad_MSELoss(weights, input_data, actual)
    return gradients



def TrainModel(weights, X, y, learning_rate, epochs):
    '''
    Trains on one data point per epoch, chosen randomly from the training set
    '''
    for i in range(epochs):
        rand_mat = np.random.randint(0,len(mats))

        # predicts the properties for every material
        pred_props = ForwardPass(X[rand_mat])

        # turns these predictions into predictions of efficiency
        pred_eff, grad_pred_eff = f_np(create_design(pred_props))

        # based on the predictions of efficiency, produces a list of expected material property values
        alpha = 0.05
        exp = pred_props + alpha * (y - pred_eff) * grad_pred_eff

        # Finally, update the neural net with the expected output
        loss = MeanSquaredErrorLoss(weights, X, exp)
        gradients = CalculateGradients(weights, X)

        ## Update Weights
        for j in range(len(weights)):
            weights[j][0] -= learning_rate * gradients[j][0] ## Update Weights
            weights[j][1] -= learning_rate * gradients[j][1] ## Update Biases

        if i%5 ==0: ## Print MSE every 5 epochs
            print("MSE : {:.2f}".format(loss))



if __name__ == '__main__':
    pass





# REFERENCE: 
# https://coderzcolumn.com/tutorials/artificial-intelligence/guide-to-create-simple-neural-networks-using-jax

'''
def find_params_for_nn():
    
    rf_grid_params = {'n_estimators': [250,400,550], 'learning_rate': [0.05, 0.15, 0.25], 'max_depth': [1,2,3,4,5]}
    grid = GridSearchCV(estimator = nn(), param_grid = rf_grid_params, refit = True, verbose = 2, cv = 5)
    grid.fit(X_train, y_train)

mlp = nn(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=200)
mlp.fit(X_train,y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)


'''



