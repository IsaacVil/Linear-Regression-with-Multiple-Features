import numpy as np
import matplotlib.pyplot as plt

def derivativesByFor(x, y, w, b): #Metodo super ineficiente pero util para el entimiento del algoritmo de las derivadas
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

def derivativesByMatrixMult(x, y, w, b): #Al ser numpy usa C bajando lo lento de la interpretacion de python.
    der_W = np.zeros(w.shape[0])
    der_B = 0
    y_hat = x @ w + b
    err = y_hat - y
    m = x.shape[1]
    n = x.shape[0]
    der_W = (x.T @ err) / n
    """
    En si lo que hace x.T @ err / n, es esto. Solo que lo hace de una mejor manera.
    for i in range(m):
        der_W[i] = np.dot(err, x[:, i]) * (1/ n)
    """
    der_B = (np.sum(err)) / n
    return der_W, der_B


def gradientDescent(x, y, w, b, e, alpha, maxIter): #e = epsilon siendo el valor minimo que debe cambiar entre w's y b's. maxIter el limite por si acaso no acaba el programa o no queremos iterar tanto
    #alpha es el learning rate por el cual aprende nuestro modelo
    i = 0
    w_Copy = np.zeros(w.shape[0])
    b_Copy = 0
    w_Hist = np.zeros(1)
    w_Hist = w.reshape(1, -1)  
    b_Hist = np.array([b]) 
    
    while (i < maxIter):
        w_Copy = w.copy()
        b_Copy = b
        der_W, der_B = derivativesByMatrixMult(x, y, w, b)
        w = w - alpha * der_W
        b = b - alpha * der_B
        w_Hist = np.vstack([w_Hist, w]) #vamos stackeando los vectores en una matriz
        b_Hist = np.append(b_Hist, b)
        if ((np.all(np.abs(w_Copy - w) < e)) and (abs(b_Copy - b) < e)):
            break
        i = i + 1
        
    return w, b, w_Hist, b_Hist

def normX(x):
    m = x.shape[0]
    unos = np.ones(m)
    miu = (x.T @ unos) / m
    sigma = np.sqrt((unos @ ((x - miu)**2)) / m) #multiplicacion de matriz (solo permite n1xm1 * n2xm2 en el que m1 = n2)
    z = (x - miu) / sigma
    return z, miu, sigma

def dataGeneratorWithAFormula(num):
    ganancia = np.zeros(num)

    population = np.random.uniform(50000,500000, size=num) 
    gdp = np.random.uniform(10000, 150000, size=num) 
    sqrtMeters = np.random.uniform(100, 500, size=num) 
    youngPopulationIndex = np.random.uniform(0.20, 0.45, size=num) #Aqui elegimos un valor random entre 0.20 y 0.45 o sea 20% de la poblacion es joven o 40%
    ruido = np.random.normal(0, 20_000, size=num)

    ganancia = 0.001 * population + 0.8 * gdp + 100 * sqrtMeters - 2.3 * youngPopulationIndex + ruido #Esta es la formula que buscara aproximar el modelo.
    x = np.column_stack([population, gdp, sqrtMeters, youngPopulationIndex])
    y = ganancia
    return x, y

def cost(w_Hist, b_Hist, x, y):
    m = x.shape[0]
    f_wb = x @ w_Hist.T + b_Hist
    errors = (f_wb - y.reshape(-1, 1)) ** 2
    cost_Hist = np.sum(errors, axis=0) / (2 * m)
    
    return cost_Hist

def chartOfCostVsIteration(cost_Hist):
    fig, ax = plt.subplots()
    ax.plot(cost_Hist)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.set_title("Cost vs Iteration")
    plt.show()

dataGenX, dataGenY = dataGeneratorWithAFormula(1000)
dataNormX, miuX, sigmaX = normX(dataGenX)
w, b, w_Hist, b_Hist = gradientDescent(dataNormX, dataGenY, w=np.zeros(4), b=0, e=0.001, alpha=0.1, maxIter=10000000)
cost_Hist = cost(w_Hist, b_Hist, dataNormX, dataGenY)

def f_wb(miu, sigma, w, b, predic, idxfeature):
    valuesofX = miu.copy()
    valuesofX[idxfeature] = predic
    valuesofX = (valuesofX - miu)/sigma
    f = np.dot(valuesofX, w) + b
    return f

def chartOfPredictionsBasedOnGdp(miu, sigma, w, b):
    # Valores del feature 1 desde el minimo hasta al maximo espaciados en 100 (Esto es solo para graficar pues ocupa varios puntos)
    x_feat = np.linspace(
        dataGenX[:, 1].min(),
        dataGenX[:, 1].max(),
        100
    )

    #Esto es para llenar todos los valores de los puntos (w1*miu1 + w2*miu2 + w3*miu3 + w4*miu4) + b haciendo todo estatico
    X_plot = np.tile(miu, (100, 1))
    X_plot[:, 1] = x_feat #aqui se cambia para que sea w1*miu + w2*x2 siendo x2 la unica variable para la grafica

    # Normalizamos igual que el entrenamiento
    X_plot_norm = (X_plot - miu) / sigma #normalizamos los datos debido a que las ws estan normalizadas tambien

    
    y_pred = X_plot_norm @ w + b #se hace el modelo de prediccion (puntos del modelo para graficar)

    fig, ax = plt.subplots(figsize=(8,6))

    ax.scatter(dataGenX[:, 1], dataGenY, alpha=0.4, label="Real Data") #Esto es de poner los puntos de los datos de X en el feature 1 y sus ganacias
    ax.plot(x_feat, y_pred, color="red", linewidth=2, label="Model") #Esto ya es para graficar el modelo
    
    ax.set_xlabel("Gross Domestic Product (GDP)")
    ax.set_ylabel("Revenue $")
    ax.set_title("Revenue vs Gross Domestic Product (GDP)")
    ax.legend()

    plt.show()

chartOfCostVsIteration(cost_Hist)
chartOfPredictionsBasedOnGdp(miuX, sigmaX, w, b)

gdpForPrediction = 100000
PredictionBasedOnGdp = f_wb(miuX, sigmaX, w, b, gdpForPrediction, idxfeature=1) 
print(PredictionBasedOnGdp)