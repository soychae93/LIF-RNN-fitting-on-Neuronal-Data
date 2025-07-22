# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 21:58:48 2021

@author: Soyoung Chae
"""

import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam, LBFGS
from torch.utils.data import Dataset, DataLoader

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
import scipy.io
import scipy.integrate
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_lyapunov

    
def gen_spontaneous_x(mu, sigma, N, k):
    #xsp = np.random.normal(mu, sigma, (N, 1))
    v = np.random.normal(0, 1, (N, 1))
    v = 0.15 / np.std(v) * v
    xsp = mu + v
    return xsp

def sigmoid(x):
    return 1 / (1 +np.exp(-x))

def feval(funcName, *args):
	'''
	This function is similar to "feval" in Matlab.
	Example: feval('cos', pi) = -1.
	'''
	return eval(funcName)(*args)

def transduction(X, params):
    out = np.empty_like(X)
    I = X > 0
    I2 = X < 0
    out[I] = X[I]
    out[I2] = 0
    return out

def relu(X, params):
    out = np.ones(params['NN'])*params['r0']
    I = X > params['r0']
    I2 = X < params['r0']
    out[I] = X[I]
    out[I2] = params['r0']
    out[I2] = params['r0']
    return out

def gen_input_gocue(ti):
    tau_decay = 0.5
    tau_rise = 50e-3
    tmax = np.log(tau_decay / tau_rise) * tau_decay * tau_rise / (tau_decay - tau_rise)

    t = np.exp(-ti / tau_decay) - np.exp (-ti / tau_rise)
    amax = np.exp(-tmax / tau_decay) - np.exp (-tmax / tau_rise)
    input_gocue = 5 / amax * t
    #tau_decay = 500
    #tau_rise = 50
    #input_gocue = np.exp(-ti/tau_decay) - np.exp(-ti/tau_rise)
    return input_gocue

def gen_input_samplecue(ti, s):
    input_sample_cue = s*np.ones(len(ti),)
    return input_sample_cue

def rate_dynamics_ode(X, t, W_rec, params, input_gocue, constant_input):
    r = feval('transduction', X, params)
    mfunc = interp1d(t, input_gocue, bounds_error=False, fill_value="extrapolate")
    x_dot = params['over_tau'] * (-X + np.matmul(W_rec, r) + mfunc(t) + constant_input)
    return x_dot

def integrate_dynamics(W_rec, params, ini_cond, constant_input):
    # Add noise to the initial condition if indicated
    if params['initial_cond_noise'] != 0:
        noise = np.random.normal(0,1,len(ini_cond))
        ini_cond = noise + ini_cond
        
    # Solve the ODEs governing the eneuronal dynamics
    t = np.linspace(0, params['t_final'], params['n_timepoints'])*1e-3
    y0 = ini_cond
    out = scipy.integrate.odeint(rate_dynamics_ode, y0, t, args = (W_rec, params, constant_input))
    tmp = np.empty_like(out)
    for i in range(0, params['n_timepoints']):
        tmp[i] = transduction(out[i], params)
    out = tmp
    return out


def getGramian(W):
    N = W.shape[0]
    A = W - np.eye(N)
    X = -np.eye(N)
    Q = solve_continuous_lyapunov(np.transpose(A), X)
    return Q


def ini_xstars(W_rec, xsp_, k, NE, x_stars_ini, c_ini):    
    x_stars_std = 0.2
    Q = getGramian(W_rec)
    NN = W_rec.shape[0]
    w, v = np.linalg.eig(Q) # Gramian matrix의 eigenvector (eigen value 모두 real)   
    x_stars = np.matmul(v, x_stars_ini)
    l2_norm = np.sum(x_stars**2, axis=0)
    z = NN * k * (x_stars_std)**2
    x_stars = np.sqrt(z/l2_norm) * x_stars
    c = pushNull(x_stars, xsp_[0:NE], c_ini, NE, k)
    return c, x_stars + xsp_

def pushNull(x_stars, xsp_, c_ini, NE, k):
    gamma_xstars = x_stars[0:NE,:] + xsp_
    x_stars_motor = np.concatenate((gamma_xstars, xsp_), axis=1)  
    h = np.linalg.solve(np.matmul(x_stars_motor.T, x_stars_motor), x_stars_motor.T)
    c = c_ini - np.matmul(np.matmul(c_ini, x_stars_motor), h)
    return c

def motor_cost(W_rec, params, fr_, constant_input, C_, novel_target):
    motorcost = 0
    k = params['k']
    NE = 160
    for mov in range(0, k):
        dynamics_ = integrate_dynamics(W_rec, params, fr_[:, mov], constant_input)
        err1 = np.matmul(C_[0], dynamics_[:,0:NE].T) - novel_target[:,2*mov]
        err2 = (np.matmul(C_[1], dynamics_[:,0:NE].T) - novel_target[:,2*mov+1])*3
        motorcost += np.sum(err1**2) + np.sum(err2**2)
    return motorcost

def regularize_C(C_):
    reg_C = np.linalg.norm(C_[0]) + np.linalg.norm(C_[1]) 
    return reg_C

class ExperimentData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx,:], self.y[idx]
    
def lossFun():
    NN = params['NN']
    k = params['k']
    C = [x[0:NE], x[NE:2*NE]]
    C_, fr_ = ini_xstars(W_rec, xsp, k, NE, np.reshape(x[2*NE:, ], (k, NN)).T, C)
    motorcost = motor_cost(W_rec, params, fr_, constant_input, C_, novel_target)
    reg_C = regularize_C(C_)

    err = (1/8)*(1e-3) * motorcost + reg_C/(2*NE)

    print('====================================')
    print('X :', np.sum(x[320:]))
    print('C :', np.sum(x[0:320]))
    print('error: ', err)
    print('motor cost: ', motorcost * (1/8) * (1e-3))
    print('regularization: ', reg_C)
    print('====================================')
    return err





    
class NNet(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes, loss, sigmoid=False):
        super().__init__()
        self.input_dim = input_dim
        self.layer_sizes = hidden_layer_sizes
        self.iter = 0
        # The loss function could be MSE or BCELoss depending on the problem
        self.lossFct = loss

        # We leave the optimizer empty for now to assign flexibly
        self.optim = None

        
        hidden_layer_sizes = [input_dim] + hidden_layer_sizes
        last_layer = nn.Linear(hidden_layer_sizes[-1], 1)
        self.layers =\
            [nn.Sequential(nn.Linear(input_, output_), nn.ReLU())
             for input_, output_ in 
             zip(hidden_layer_sizes, hidden_layer_sizes[1:])] +\
            [last_layer]
        
        # The output activation depends on the problem
        if sigmoid:
            self.layers = self.layers + [nn.Sigmoid()]
            
        self.layers = nn.Sequential(*self.layers)

        
    def forward(self, x):
        #x = self.layers(x)
        x = movement
        return x
    
    
    def train(self, data_loader, epochs, validation_data=None):

        for epoch in range(epochs):
            running_loss = self._train_iteration(data_loader)
            val_loss = None
            if validation_data is not None:
                y_hat = self(validation_data['X'])
                val_loss = self.lossFct(input=y_hat, target=validation_data['y']).detach().numpy()
                print('[%d] loss: %.3f | validation loss: %.3f' %
                  (epoch + 1, running_loss, val_loss))
            else:
                print('[%d] loss: %.3f' %
                  (epoch + 1, running_loss))
            
            
                
    def _train_iteration(self,data_loader):
        running_loss = 0.0
        for i, (X,y) in enumerate(data_loader):
            
            X = X.float()
            y = y.unsqueeze(1).float()
            
            X_ = Variable(X, requires_grad=True)
            y_ = Variable(y)
              
            ### Comment out the typical gradient calculation
#             pred = self(X)
#             loss = self.lossFct(pred, y)
            
#             self.optim.zero_grad()
#             loss.backward()
            
            ### Add the closure function to calculate the gradient.
            def closure():
                if torch.is_grad_enabled():
                    self.optim.zero_grad()
                output = self(X_)
                loss = self.lossFct(output, y_)
                if loss.requires_grad:
                    loss.backward()
                return loss
            
            self.optim.step(closure)
            
            # calculate the loss again for monitoring
            output = self(X_)
            loss = closure()
            running_loss += loss.item()
               
        return running_loss
    
    # I like to include a sklearn like predict method for convenience
    def predict(self, X):
        X = torch.Tensor(X)
        return self(X).detach().numpy().squeeze()
    
class ExperimentData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx,:], self.y[idx]
    
    
###

    
INPUT_SIZE = X.shape[1]
EPOCHS=5 # Few epochs to avoid overfitting
HIDDEN_LAYER_SIZE = []

data_loader = DataLoader(data, batch_size=X.shape[0])
net = NNet(INPUT_SIZE, HIDDEN_LAYER_SIZE, loss = nn.BCELoss(), sigmoid=True)
net.optim = LBFGS(net.parameters(), history_size=10, max_iter=4)
net.train(data_loader, EPOCHS, validation_data = {"X":torch.Tensor(X_val), "y":torch.Tensor(y_val).unsqueeze(1) })
