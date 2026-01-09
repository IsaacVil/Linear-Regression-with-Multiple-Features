import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from regression import normX, chartOfCostVsIteration, f_wbOnAll, chartOfPredictionsBasedOnFeature, gradientDescent
from sklearn.preprocessing import PolynomialFeatures

def derivativesByMatrixMult_Reg(x, y, w, b, lambda_): #Al ser numpy usa C bajando lo lento de la interpretacion de python.
    der_W = np.zeros(w.shape[0])
    der_B = 0
    y_hat = x @ w + b
    err = y_hat - y
    m = x.shape[1]
    n = x.shape[0]
    der_W = (x.T @ err) / n + (lambda_/n) * w
    """
    En si lo que hace x.T @ err / n, es esto. Solo que lo hace de una mejor manera.
    for i in range(m):
        der_W[i] = np.dot(err, x[:, i]) * (1/ n)
    """
    der_B = (np.sum(err)) / n
    return der_W, der_B

def gradientDescent_Reg(x, y, w, b, e, alpha, maxIter, lambda_): #e = epsilon siendo el valor minimo que debe cambiar entre w's y b's. maxIter el limite por si acaso no acaba el programa o no queremos iterar tanto
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
        der_W, der_B = derivativesByMatrixMult_Reg(x, y, w, b, lambda_)
        w = w - alpha * der_W
        b = b - alpha * der_B
        w_Hist = np.vstack([w_Hist, w]) #vamos stackeando los vectores en una matriz
        b_Hist = np.append(b_Hist, b)
        if ((np.linalg.norm(der_W) < e) and (abs(b_Copy - b) < e)):
            break
        i = i + 1
        
    return w, b, w_Hist, b_Hist

def data2XGeneratorWithAFormula(num):
    population = np.random.uniform(50_000, 500_000, size=num)
    gdp = np.random.uniform(10_000, 150_000, size=num)

    ruido = np.random.normal(0, 300_000, size=num)

    # RELACIÓN REAL (LINEAL)
    ganancia = 0.001 * population + 3.5 * gdp + ruido

    X = np.column_stack([population, gdp])
    y = ganancia

    return X, y


def chartDetectPolynomialBehavior(
    x_raw, y, poly, miu, sigma, w, b, feature_idx, feature_name
):
    
    x_feat = np.linspace(
        x_raw[:, feature_idx].min(),
        x_raw[:, feature_idx].max(),
        200
    )

    X_base = np.mean(x_raw, axis=0)
    X_plot = np.tile(X_base, (200, 1))
    X_plot[:, feature_idx] = x_feat

    X_poly = poly.transform(X_plot)

    X_poly_norm = (X_poly - miu) / sigma

    y_pred = X_poly_norm @ w + b

    plt.figure(figsize=(8,6))
    plt.scatter(x_raw[:, feature_idx], y, alpha=0.3, label="Real Data")
    plt.plot(x_feat, y_pred, color="red", linewidth=2, label="Model")
    plt.xlabel(feature_name)
    plt.ylabel("Revenue (100000)")
    plt.title(f"{feature_name}")
    plt.legend()
    plt.grid(True)
    plt.show()




Poly = PolynomialFeatures(degree=5, include_bias=False)
xgen, ygen = data2XGeneratorWithAFormula(num=150)
#DATOS MUY EXTRAÑOS PERO QUE DESTRUYEN LA MANERA EN LA QUE SE AJUSTA LOS POLINOMIOS QUE NO SE REGULARIZAN, SERIAN COMO CASOS EXCEPCIONALES QUE EL MODELO REGULARIZADO APRENDE A IGNORAR

X, miu, sigma = normX(Poly.fit_transform(xgen))
w = np.zeros(X.shape[1])
b = 0
lambda_ = 0.7

#not regularized
w, b, w_Hist, b_Hist = gradientDescent(X, ygen, w.copy(), b, 0.00001, 0.1, 20000)
#regularized
w_reg, b_reg, w_Hist_reg, b_Hist_reg = gradientDescent_Reg(X, ygen, w.copy(), b, 0.00001, 0.1, 20000, lambda_)

chartDetectPolynomialBehavior(xgen, ygen, Poly, miu, sigma, w, b, 1, "PoblatiGross Domestic Product (GDP), Not Regularized")
chartDetectPolynomialBehavior(xgen, ygen, Poly, miu, sigma, w_reg, b_reg, 1, "Gross Domestic Product (GDP), Regularized")

print("||w sin regularización|| =", np.linalg.norm(w))
print("||w con regularización|| =", np.linalg.norm(w_reg))