# Machine Learning Algorithms from Scratch

A comprehensive collection of fundamental machine learning and artificial intelligence algorithms implemented completely from first principles ("hard-coded") in Python and NumPy, without relying on black-box machine learning frameworks (such as scikit-learn, PyTorch, or TensorFlow) for the core learning mechanisms.

---

## Table of Contents

- [About This Repository](#about-this-repository)
- [Repository Structure](#repository-structure)
- [Algorithm Deep Dives](#algorithm-deep-dives)
  - [1. Neural Networks and Backpropagation from Scratch](#1-neural-networks-and-backpropagation-from-scratch)
  - [2. The Perceptron: Foundations of Connectionism](#2-the-perceptron-foundations-of-connectionism)
  - [3. Decision Trees and Information Theory](#3-decision-trees-and-information-theory)
  - [4. Unsupervised Clustering Algorithms](#4-unsupervised-clustering-algorithms)
  - [5. Regression and Statistical Diagnostics](#5-regression-and-statistical-diagnostics)
- [Commit Log and Project Evolution](#commit-log-and-project-evolution)
- [Getting Started](#getting-started)
- [Tech Stack and Dependencies](#tech-stack-and-dependencies)
- [Author](#author)

---

## About This Repository

> "What I cannot create, I do not understand." - Richard Feynman

When studying Machine Learning and Artificial Intelligence, high-level libraries make it straightforward to call `.fit()` and `.predict()`. However, relying solely on high-level abstractions can obscure the underlying mechanics: matrix calculus, backpropagation, decision boundaries, information entropy, and distance metric optimization.

This repository was created to document a hands-on learning journey through the foundational, base-level concepts of AI. Every core algorithm here is built manually from scratch:
- Matrix operations and linear algebra instead of automatic differentiation engines.
- Explicit forward and backward passes with analytical gradient derivations.
- Recursive entropy-based partitioning instead of black-box tree solvers.
- Vectorized distance calculations and iterative centroid/medoid updates for clustering.
- Statistical diagnostics (VIF and correlation analysis) to understand feature interactions.

---

## Repository Structure

```
Machine-Learning-Algorithims/
|-- HardCodedNeuralNetwork(LeakyReLu).ipynb  # 2-layer Neural Network with LeakyReLU and Softmax (MNIST)
|-- HardCodedNeuralNetwork(ReLu).ipynb       # 2-layer Neural Network with standard ReLU (Dying ReLU study)
|-- Perceptron.py                            # Forward Perceptron with step activation function (Logic Gates)
|-- SelfLearningPerceptron.py                # Self-learning Perceptron with Rosenblatt learning rule
|-- DecisionTree.py                          # Step-by-step Shannon Entropy and Information Gain calculations
|-- DecisionTreeAlgo.py                      # Automated recursive ID3 Decision Tree builder and ASCII visualizer
|-- KMeans.py                                # K-Means clustering with Euclidean distance and centroid updates
|-- KMedoids.py                              # K-Medoids (PAM) clustering minimizing intra-cluster distances
|-- RegressionModle1.ipynb                   # California Housing regression with feature correlation analysis
|-- RegressionModle2.ipynb                   # Multicollinearity analysis and Variance Inflation Factor (VIF)
|-- requirements.txt                         # Python environment dependencies
`-- README.md                                # Project documentation
```

---

## Algorithm Deep Dives

### 1. Neural Networks and Backpropagation from Scratch

- Files: `HardCodedNeuralNetwork(LeakyReLu).ipynb`, `HardCodedNeuralNetwork(ReLu).ipynb`
- Dataset: MNIST Handwritten Digits (70,000 images, 28x28 pixels = 784 input features, 10 digit classes)
- Architecture: 784 -> 10 -> 10 (Input -> Hidden Layer -> Softmax Output)

#### Mathematical Formulation

```
Forward Propagation:
  Linear step 1:     Z1 = W1 * A0 + b1       [W1 shape: 10 x 784, b1 shape: 10 x 1]
  Activation step 1: A1 = LeakyReLU(Z1)      [alpha = 0.1 for negative values]

  Linear step 2:     Z2 = W2 * A1 + b2       [W2 shape: 10 x 10,  b2 shape: 10 x 1]
  Activation step 2: A2 = Softmax(Z2)        [Normalized probabilities across 10 classes]

Softmax Normalization:
  Softmax(z_i) = exp(z_i - max(z)) / sum( exp(z_j - max(z)) )

Backward Propagation (Analytical Gradients):
  Layer 2 error:     dZ2 = A2 - Y_onehot
  Layer 2 weights:   dW2 = (1 / m) * dZ2 * (A1^T)
  Layer 2 biases:    db2 = (1 / m) * sum(dZ2, axis=1)

  Layer 1 error:     dZ1 = (W2^T * dZ2) * d_LeakyReLU(Z1)
  Layer 1 weights:   dW1 = (1 / m) * dZ1 * (A0^T)
  Layer 1 biases:    db1 = (1 / m) * sum(dZ1, axis=1)

Parameter Updates (Gradient Descent):
  W1 = W1 - (learning_rate * dW1)
  b1 = b1 - (learning_rate * db1)
  W2 = W2 - (learning_rate * dW2)
  b2 = b2 - (learning_rate * db2)
```

#### The Dying ReLU Problem and Breakthrough

In `HardCodedNeuralNetwork(ReLu).ipynb`, using standard ReLU (`max(0, z)`) resulted in neurons that output zero whenever activations were non-positive. Because the derivative of ReLU for non-positive inputs is zero, the gradient cannot flow backward through those units, causing neurons to permanently "die." As a result, the network could not learn effectively and remained stuck at approximately 10% accuracy (equivalent to random guessing among 10 classes).

In `HardCodedNeuralNetwork(LeakyReLu).ipynb`, switching to LeakyReLU (`max(alpha * z, z)` with `alpha = 0.1`) ensured that a non-zero gradient was always maintained even for negative values. This single architectural adjustment allowed gradient flow across all units, boosting test classification accuracy from ~10% to ~80% without using external deep learning libraries.

---

### 2. The Perceptron: Foundations of Connectionism

- Files: `Perceptron.py`, `SelfLearningPerceptron.py`

#### Forward Step Evaluation (`Perceptron.py`)

A single artificial neuron computing the linear dot product of inputs and weights plus a bias weight, passed through a bipolar threshold step function:

```
Net Input:
  S = base_weight + sum( w_i * x_i )    for i = 1 to n

Threshold Function:
  Tau(S) = +1   if S >= 0
  Tau(S) = -1   if S < 0
```

Tested on boolean logic tables (such as the OR function) using fixed weights.

#### Self-Learning Perceptron (`SelfLearningPerceptron.py`)

Implements Frank Rosenblatt's perceptron learning algorithm from scratch. Starting with randomly initialized weights between -5 and 5, the model repeatedly passes through the training examples and updates its parameters whenever a misclassification occurs:

```
Error Calculation:
  error = y - y_hat

Weight and Bias Updates:
  base_weight = base_weight + (learning_rate * error)
  w_j         = w_j         + (learning_rate * error * x_j)
```

The training loop continues across epochs until the cumulative error over the entire truth table reaches zero, proving convergence on linearly separable functions.

---

### 3. Decision Trees and Information Theory

- Files: `DecisionTree.py`, `DecisionTreeAlgo.py`
- Dataset: Classic "Play Tennis" Weather Dataset (Features: Outlook, Temperature, Humidity, Wind; Target: Play Tennis)

#### Mathematical Formulation

```
Shannon Entropy of a set S:
  H(S) = - sum( p_i * log2(p_i) )
  where p_i is the proportion of samples belonging to class i.

Weighted Entropy after splitting on feature A:
  H(S, A) = sum( (|S_v| / |S|) * H(S_v) )
  for each subset S_v corresponding to unique value v of feature A.

Information Gain:
  IG(S, A) = H(S) - H(S, A)
```

#### Manual Calculations vs. Automated Tree Building

- `DecisionTree.py`: Traces each calculation manually at the root and subsequent branches (Sunny, Rain, Overcast) to verify the exact Information Gain for every candidate feature.
- `DecisionTreeAlgo.py`: Automates the complete ID3 decision tree algorithm. It recursively selects the feature with maximum information gain (minimum weighted entropy), partitions the dataset, identifies pure leaf nodes (`H(S) == 0`), and renders the learned tree in clear ASCII format:

```
Outlook
|-- Sunny
|   H
|   |-- High -> N
|   `-- Normal -> Y
|-- Overcast -> Y
`-- Rain
    W
    |-- Weak -> Y
    `-- Strong -> N
```

---

### 4. Unsupervised Clustering Algorithms

- Files: `KMeans.py`, `KMedoids.py`

#### Mathematical Formulation

```
Euclidean Distance:
  distance(p, q) = sqrt( sum( (p_i - q_i)^2 ) )

K-Means Centroid Update:
  mu_k = (1 / |C_k|) * sum( x )   for all points x assigned to cluster C_k

K-Medoids PAM Update:
  m_k = argmin_{y in C_k} ( sum_{x in C_k} distance(x, y) )
```

#### K-Means vs. K-Medoids Comparison

- **K-Means (`KMeans.py`)**: Partitions points into `k` clusters by assigning each point to the closest centroid, then recalculates the centroid as the geometric mean of that cluster. Tested on both an 8-point dataset with known initializations and a 40-point synthetic random dataset.
- **K-Medoids (`KMedoids.py`)**: Rather than computing an artificial center that may not exist in the data, K-Medoids requires that the cluster center (the medoid) be an actual data point from the dataset. It iterates through candidate points in the cluster to find the one that minimizes the sum of distances to all other cluster members. This makes K-Medoids significantly more robust to noise and outliers.

---

### 5. Regression and Statistical Diagnostics

- Files: `RegressionModle1.ipynb`, `RegressionModle2.ipynb`
- Dataset: California Housing (`fetch_california_housing`)
- Target: Median House Value (`MedHouseVal`)

#### Exploration and Diagnostics

- Exploratory Data Analysis: Feature correlation matrices, distributions, and heatmap visualizations using Seaborn and Matplotlib.
- Comparative Modeling: Evaluated Ordinary Least Squares (`LinearRegression`), Stochastic Gradient Descent (`SGDRegressor`), L1 Regularization (`Lasso`), and L2 Regularization (`Ridge`).
- Multicollinearity and VIF (`RegressionModle2.ipynb`):
  Identified high collinearity between `AveRooms` and `AveBedrms` (correlation approximately 0.85). Calculated the Variance Inflation Factor (VIF):

```
Variance Inflation Factor:
  VIF_i = 1 / (1 - R_i^2)
```

Where `R_i^2` is the coefficient of determination when regressing feature `x_i` against all other independent variables. Diagnosed variance inflation and demonstrated that dropping redundant collinear predictors stabilizes regression coefficients and improves model interpretability.

---

## Commit Log and Project Evolution

This project was developed incrementally to build an intuitive, mathematically grounded understanding of AI algorithms:

1. **Regression Models and Multicollinearity Analysis**
   - Built initial baseline regression models on the California Housing dataset.
   - Diagnosed multicollinearity using correlation matrices and Variance Inflation Factor (VIF).
   - Explored the impact of regularized regression (Ridge and Lasso).

2. **Decision Tree from First Principles**
   - `DecisionTree.py`: Hard-coded manual entropy and information gain calculations for root and child branches on the Play Tennis dataset.
   - `DecisionTreeAlgo.py`: Implemented recursive ID3 algorithm to automate tree construction and print clean ASCII decision trees.

3. **Perceptron and Gradient Learning**
   - `Perceptron.py`: Implemented single-neuron forward evaluation with step activation on logic gates.
   - `SelfLearningPerceptron.py`: Built an autonomous learning loop using Rosenblatt's perceptron rule and gradient descent to learn truth table weights from random initialization.

4. **Multi-Layer Neural Networks and Backpropagation**
   - Implemented a 2-layer neural network from scratch using raw NumPy matrix operations on MNIST.
   - Explored the Dying ReLU phenomenon in `HardCodedNeuralNetwork(ReLu).ipynb`.
   - Solved vanishing gradients with LeakyReLU in `HardCodedNeuralNetwork(LeakyReLu).ipynb`, raising accuracy from ~10% to ~80%.

5. **Unsupervised Clustering Algorithms**
   - `KMeans.py`: Implemented K-Means from first principles with Euclidean distance and centroid updates.
   - `KMedoids.py`: Built K-Medoids (Partitioning Around Medoids) to explore exemplar-based clustering and outlier robustness.

6. **Documentation and Standardization**
   - Added clean `README.md` without emojis or fragile symbols.
   - Added `requirements.txt` to enable straightforward reproducibility across environments.

---

## Getting Started

### Prerequisites

Ensure you have Python 3.8 or higher installed.

```bash
git clone https://github.com/mohamedthabet7/Machine-Learning-Algorithims.git
cd Machine-Learning-Algorithims
pip install -r requirements.txt
```

### Running the Python Scripts

- Run Automated Decision Tree:
  ```bash
  python DecisionTreeAlgo.py
  ```

- Run Self-Learning Perceptron:
  ```bash
  python SelfLearningPerceptron.py
  ```

- Run K-Means Clustering:
  ```bash
  python KMeans.py
  ```

- Run K-Medoids Clustering:
  ```bash
  python KMedoids.py
  ```

### Exploring Jupyter Notebooks

Launch Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

- Open `HardCodedNeuralNetwork(LeakyReLu).ipynb` to view the full MNIST neural network trained from scratch.
- Open `RegressionModle2.ipynb` to explore the multicollinearity and VIF analysis.

---

## Tech Stack and Dependencies

- Language: Python 3
- Numerical Computing: NumPy (Vectorized matrix operations, linear algebra)
- Data Structures: Pandas (DataFrame manipulation, feature slicing)
- Visualization: Matplotlib, Seaborn
- Statistical Tools: Statsmodels (VIF calculation)
- Datasets: Scikit-learn (Used strictly for loading OpenML MNIST and California Housing data)

---

## Author

- Mohamed Thabet
- GitHub: [@mohamedthabet7](https://github.com/mohamedthabet7)
- Developed as a hands-on exploration into the fundamental mathematical foundations of Artificial Intelligence and Machine Learning.
