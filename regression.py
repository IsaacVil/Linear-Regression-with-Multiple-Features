import numpy as np

def derivatives(x, y, w, b):
    der_B = 0
    n = x.shape[0] #numero de filas
    m = x.shape[1] #numero de columnas
    der_W = np.zeros(w.shape[0])
    for j in range(m):
        f = 0
        for i in range(n):
            f = f + (((np.dot(x[i], w) + b) - y[i]) * x[i, j])
        f = f/n
        der_W[j] = f

    f = 0
    for i in range(n):
        f = f + (((np.dot(x[i], w) + b) - y[i]))
    f = f/n
    der_B = f
    return der_W, der_B
