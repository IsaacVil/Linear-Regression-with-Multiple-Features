## Multiple Linear Regression Model with Several Features to Predict Restaurant Profit

This project implements **multiple linear regression** from scratch (without using machine learning libraries such as scikit-learn) to predict the **profit of a restaurant** based on several **features (input variables)**:

- Gross Domestic Product of the city (`GDP`)
- Total population of the city
- Percentage of the population that is young
- Square meters of the restaurant’s premises

Training is done using **linear algebra and derivatives**, relying only on `numpy` for matrix operations and `matplotlib` for plotting.

The main code lives in:

- [regression.py](regression.py)

---

## 1. Problem Statement

We want to build a model that **predicts the profit** (revenue) of a restaurant in a given city, using four economic and business-related features:

- $x_1$: total population of the city
- $x_2$: city `GDP`
- $x_3$: square meters of the restaurant location
- $x_4$: young population index (percentage of the population that is young)

The goal is to learn a linear function of the form:

$$
\hat{y} = f_{\mathbf{w},b}(\mathbf{x}) = w_1 x_1 + w_2 x_2 + w_3 x_3 + w_4 x_4 + b
$$

where:

- $\hat{y}$ is the **predicted profit** for a restaurant in a given city.
- $w_1, w_2, w_3, w_4$ are the **weights** (model parameters) associated with each feature.
- $b$ is the **bias** or intercept.
- $\mathbf{x} = (x_1, x_2, x_3, x_4)$ is the feature vector of one observation.

---

## 2. Synthetic Data Generation

The data does not come from an external dataset. Instead, it is **synthetically generated** in the function `dataGeneratorWithAFormula(num)` in [regression.py](regression.py).

For each example, the following variables are sampled:

- $\text{population} \sim U(50\,000, 500\,000)$
- $\text{gdp} \sim U(10\,000, 150\,000)$
- $\text{sqrtMeters} \sim U(100, 500)$
- $\text{youngPopulationIndex} \sim U(0.20, 0.45)$

Then, the **true profit** (target $y$) is computed using a linear formula plus Gaussian noise:

$$
	ext{profit} = 0.001 \cdot \text{population}
\;+\
0.8 \cdot \text{gdp}
\;+\
100 \cdot \text{sqrtMeters}
\;-\
2.3 \cdot \text{youngPopulationIndex}
\;+\
\varepsilon
$$

where $\varepsilon \sim \mathcal{N}(0, 20\,000)$ is normal (Gaussian) noise with mean 0 and standard deviation 20,000, making the problem more realistic.

The final feature vector is:

$$
\mathbf{x} = \big(\text{population}, \; \text{gdp}, \; \text{sqrtMeters}, \; \text{youngPopulationIndex}\big)
$$

And the target value is:

$$
y = \text{profit}
$$

---

## 3. Feature Normalization

To make **gradient descent** work better and converge faster, the features are normalized using standardization (z-score) in the function `normX(x)`.

Given a dataset $X \in \mathbb{R}^{m \times n}$ (with $m$ examples and $n$ features), for each column $j$ we compute:

- The mean:

$$
\mu_j = \frac{1}{m} \sum_{i=1}^{m} x^{(i)}_j
$$

- The standard deviation:

$$
\sigma_j = \sqrt{ \frac{1}{m} \sum_{i=1}^{m} \big(x^{(i)}_j - \mu_j\big)^2 }
$$

Then, each value is transformed into its normalized version:

$$
z^{(i)}_j = \frac{x^{(i)}_j - \mu_j}{\sigma_j}
$$

The code computes vectors `miu` and `sigma` that store these means and standard deviations, and returns:

- `z`: matrix of **normalized** features (used in place of `x` for training)
- `miu`: means of each feature
- `sigma`: standard deviations of each feature

These statistics are later reused to make **predictions** consistent with the trained model.

---

## 4. Multiple Linear Regression Model

Let $X \in \mathbb{R}^{m \times n}$ be the matrix of normalized data, where each row is an example and each column is a feature, and:

- $\mathbf{w} \in \mathbb{R}^n$ is the weight vector.
- $b \in \mathbb{R}$ is the bias.
- $\mathbf{y} \in \mathbb{R}^m$ is the vector of true outputs.

The **vectorized prediction** is computed as:

$$
\hat{\mathbf{y}} = X \mathbf{w} + b
$$

In [regression.py](regression.py) this is implemented as:

- `y_hat = x @ w + b`

---

## 5. Cost Function (Mean Squared Error)

The cost function used is the **standard linear regression cost function** (MSE/2):

$$
J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \big( \hat{y}^{(i)} - y^{(i)} \big)^2
$$

If we define the error vector $\mathbf{e} = \hat{\mathbf{y}} - \mathbf{y}$, the matrix form is:

$$
J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} e_i^2
$$

In the code, the function `cost(w_Hist, b_Hist, x, y)` computes the **cost across all iterations** of gradient descent, using the parameter histories `w_Hist` and `b_Hist`.

---

## 6. Partial Derivatives and Gradient Descent

To minimize $J(\mathbf{w}, b)$, the code uses **gradient descent**, which requires the partial derivatives of the cost function with respect to each parameter.

### 6.1. Analytical Derivatives

The derivative of $J$ with respect to a weight $w_j$ is:

$$
\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \big( \hat{y}^{(i)} - y^{(i)} \big) x^{(i)}_j
$$

The derivative of $J$ with respect to the bias $b$ is:

$$
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \big( \hat{y}^{(i)} - y^{(i)} \big)
$$

In vector form, if $\mathbf{e} = \hat{\mathbf{y}} - \mathbf{y}$:

$$
\nabla_\mathbf{w} J = \frac{1}{m} X^T \mathbf{e}
$$
$$
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} e_i
$$

### 6.2. Implementation in Code

In [regression.py](regression.py) there are two gradient implementations:

- `derivativesByFor(x, y, w, b)`: uses **for-loops** (educational but inefficient).
- `derivativesByMatrixMult(x, y, w, b)`: uses **vectorized numpy operations** and is the one used in training.

In the vectorized version:

- Prediction: `y_hat = x @ w + b`.
- Error: `err = y_hat - y`.
- Gradient of $\mathbf{w}$: `der_W = (x.T @ err) / n`.
- Gradient of $b$: `der_B = np.sum(err) / n`.

Here, `n` is the **number of training examples**.

### 6.3. Parameter Updates (Gradient Descent)

The **gradient descent** algorithm updates the parameters iteratively:

$$
w_j := w_j - \alpha \frac{\partial J}{\partial w_j}
$$
$$
b := b - \alpha \frac{\partial J}{\partial b}
$$

where $\alpha$ is the **learning rate**.

In the code, this is implemented in `gradientDescent(x, y, w, b, e, alpha, maxIter)`:

- `alpha`: learning rate.
- `e`: epsilon, a minimum-change threshold used as a stopping criterion.
- `maxIter`: maximum number of iterations for safety.
- All values of `w` and `b` are stored in `w_Hist` and `b_Hist` for plotting and analysis.

The stopping condition is:

- If the absolute change of **all** components of `w` and the change in `b` are smaller than `e`, the loop stops.

---

## 7. Visualizing Cost vs. Iterations

The function `chartOfCostVsIteration(cost_Hist)` plots the evolution of the **cost $J$** over the iterations of gradient descent.

The goal is to observe that the cost **decreases** gradually, showing that the model is learning.

Main steps:

- `cost_Hist` is computed using the `cost(...)` function.
- `cost_Hist` is plotted on the Y-axis against the iteration number on the X-axis.

Ideally, you will see a decreasing curve that stabilizes as training converges.

---

## 8. Visualization: Profit vs. GDP

Although the model uses **four features**, to visualize its behavior, most features are held fixed while only **GDP** is varied.

The function `chartOfPredictionsBasedOnGdp(miu, sigma, w, b)` works as follows:

1. Takes GDP values from the minimum to the maximum in the dataset (`dataGenX[:, 1].min()` to `dataGenX[:, 1].max()`).
2. Keeps the other features fixed at their mean `miu`.
3. Builds a matrix `X_plot` with these points, where only the GDP feature changes.
4. Normalizes `X_plot` using the same `miu` and `sigma` used during training.
5. Computes predictions `y_pred = X_plot_norm @ w + b`.
6. Plots:
	 - Real data points: `(GDP, true profit)`.
	 - Model line: `(GDP, predicted profit)`.

This shows **how estimated profit changes with GDP**, while population, square meters and young population percentage are kept constant at their mean values.

---

## 9. Prediction Functions

There are two helper functions to make **manual predictions** with the trained model:

- `f_wbOnOne(miu, sigma, w, b, predic, idxfeature)`: changes **only one feature**, keeping the others at their mean.
- `f_wbOnAll(miu, sigma, w, b, predic)`: allows you to provide a **full feature vector**, setting all features explicitly.

### 9.1. Changing a Single Feature (e.g., GDP)

The function `f_wbOnOne(miu, sigma, w, b, predic, idxfeature)` follows this procedure:

1. Start from a vector of mean feature values `miu`.
2. Replace the position `idxfeature` with the feature value you want to test (`predic`).
3. Normalize this vector using the same `miu` and `sigma` from training.
4. Compute:

$$
f = \mathbf{w}^T \mathbf{z} + b
$$

where $\mathbf{z}$ is the normalized feature vector.

In the script example:

- `gdpForPrediction = 100000`.
- `f_wbOnOne(miuX, sigmaX, w, b, gdpForPrediction, idxfeature=1)` is used to predict the profit varying only `GDP`.

The output is the **estimated profit** for a restaurant with:

- Population, square meters and young population index at their average value.
- City GDP = 100,000.

### 9.2. Setting All Features Explicitly

The function `f_wbOnAll(miu, sigma, w, b, predic)` lets you specify **all four features** directly. The steps are:

1. Take `predic` as the feature vector $(x_1, x_2, x_3, x_4)$.
2. Normalize it using the same `miu` and `sigma` from training.
3. Compute:

$$
f = \mathbf{w}^T \mathbf{z} + b
$$

In the script example:

- `xForPred = np.array([10000, 100000, 100, 0.30])`.
- `f_wbOnAll(miuX, sigmaX, w, b, xForPred)` is used to predict the profit based on **all features at once**:
	- Population = 10,000
	- GDP = 100,000
	- Square meters = 100
	- Young population index = 0.30

---

## 10. Code Structure

Main file: [regression.py](regression.py)

Key functions:

- `dataGeneratorWithAFormula(num)`: generates synthetic data $(X, y)$ from a linear formula plus noise.
- `normX(x)`: normalizes features and returns `z, miu, sigma`.
- `derivativesByFor(x, y, w, b)`: gradient computation with for-loops (educational).
- `derivativesByMatrixMult(x, y, w, b)`: vectorized gradient computation.
- `gradientDescent(x, y, w, b, e, alpha, maxIter)`: trains the model using gradient descent.
- `cost(w_Hist, b_Hist, x, y)`: computes cost across iterations.
- `chartOfCostVsIteration(cost_Hist)`: plots cost vs. iteration.
- `f_wbOnOne(miu, sigma, w, b, predic, idxfeature)`: prediction function where you change one feature.
- `f_wbOnAll(miu, sigma, w, b, predic)`: prediction function where you specify all features.
- `chartOfPredictionsBasedOnGdp(miu, sigma, w, b)`: plots profit vs. GDP.

At the end of the file, a typical workflow is executed:

1. Data generation: `dataGenX, dataGenY = dataGeneratorWithAFormula(1000)`.
2. Normalization: `dataNormX, miuX, sigmaX = normX(dataGenX)`.
3. Model training with gradient descent to obtain `w, b, w_Hist, b_Hist`.
4. Computation of the cost history `cost_Hist`.
5. Plotting **cost vs. iteration** and **profit vs. GDP**.
6. Making two predictions:
	- A **single-feature prediction** varying only GDP (e.g. 100,000).
	- A **full-feature prediction** using a specific vector of all four features.

---

## 11. How to Run the Project

### Requirements

- Python 3.x
- Python libraries:
	- `numpy`
	- `matplotlib`

You can install them with:

- `pip install numpy matplotlib`

### Execution

1. Clone or download this project.
2. Open a terminal in the project folder.
3. Run:

- `python regression.py`

This will:

- Generate synthetic data for 1000 examples.
- Train the multiple linear regression model.
- Show the **cost vs. iteration** plot.
- Show the **profit vs. GDP** plot.
- Print to the console the **predicted profit** for a specific GDP (by default 100,000).

---

## 12. Conceptual Summary

This project is a complete and educational example of:

- **Multiple linear regression** with several features.
- Manual implementation of:
	- Data normalization.
	- MSE cost function.
	- Analytical gradient computation.
	- Gradient descent.
	- Visualization of the training process.
- Use of realistic, business-relevant features:
	- City GDP.
	- Total population.
	- Percentage of young population.
	- Square meters of the restaurant.

The main focus is to understand **step by step** how to train a multiple linear regression model to predict **restaurant profit** from these economic and business factors.
