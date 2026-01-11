import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
def sigmoid(z):
    return 1/(1+np.exp(-z))

def gradient(X, y, w, b):
    m = X.shape[0] #numero de ejemplos, filas
    n = X.shape[1] #numero de columnas
    err = sigmoid(w @ X.T + b) - y 
    w_der = (X.T @ err) / m
    b_der = np.sum(err)/m
    return w_der, b_der

def gradientDescent(X, y, w, b, alpha, maxIter, epsilon):
    i = 0
    b_ant = 0
    w_Hist = np.zeros(1)
    w_Hist = w.reshape(1, -1)  
    b_Hist = np.array([b]) 
    while (i < maxIter):
        b_ant = b
        der_W, der_B = gradient(X, y, w, b)
        w = w - alpha * der_W
        b = b - alpha * der_B
        w_Hist = np.vstack([w_Hist, w])
        b_Hist = np.append(b_Hist, b)
        if ((np.linalg.norm(der_W) < epsilon) and (abs(b_ant - b) < epsilon)):
            break
        i = i + 1
    return w, b, w_Hist, b_Hist

def gradientReg(X, y, w, b, lambda_):
    m = X.shape[0] #numero de ejemplos, filas
    n = X.shape[1] #numero de columnas
    err = sigmoid(w @ X.T + b) - y 
    w_der = (X.T @ err) / m + (lambda_ * w)/m
    b_der = np.sum(err)/m
    return w_der, b_der

def gradientDescentReg(X, y, w, b, alpha, maxIter, epsilon, lambda_):
    i = 0
    b_ant = 0
    w_Hist = np.zeros(1)
    w_Hist = w.reshape(1, -1)  
    b_Hist = np.array([b]) 
    while (i < maxIter):
        b_ant = b
        der_W, der_B = gradientReg(X, y, w, b, lambda_)
        w = w - alpha * der_W
        b = b - alpha * der_B
        w_Hist = np.vstack([w_Hist, w])
        b_Hist = np.append(b_Hist, b)
        if ((np.linalg.norm(der_W) < epsilon) and (abs(b_ant - b) < epsilon)):
            break
        i = i + 1
    return w, b, w_Hist, b_Hist

def normX(x):
    m = x.shape[0]
    unos = np.ones(m)
    miu = (x.T @ unos) / m
    sigma = np.sqrt((unos @ ((x - miu)**2)) / m) #multiplicacion de matriz (solo permite n1xm1 * n2xm2 en el que m1 = n2)
    sigma[sigma == 0] = 1
    z = (x - miu) / sigma
    return z, miu, sigma

def cost(w_Hist, b_Hist, x, y):
    m = x.shape[0]
    f_wb = sigmoid(x @ w_Hist.T + b_Hist)

    eps = 1e-15
    f_wb = np.clip(f_wb, eps, 1 - eps) #deja los valores entre 0,0000...1 o 0.99999...9 para evitar log(0) = -inf 

    errors = (-y * np.log(f_wb)) - (1-y) * np.log(1-f_wb)
    cost_Hist = np.sum(errors, axis=0) / m

    return cost_Hist

def f_wbOnAll(miu, sigma, w, b, predic):
    valuesofX = predic
    valuesofX = (valuesofX - miu)/sigma
    f = sigmoid(np.dot(valuesofX, w) + b)
    return f



