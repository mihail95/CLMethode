# Logistic Regression

## Generative and Discriminative Classifiers
* naive Bayes - **generative**
* logistic regression - **discriminative**

In $\hat{c} = argmax\ P(d|c)\cdot P(c)$, we have a **likelihood** P(d|c) and a **prior** P(c).\
Whereas Bayes makes use of the **likelihood** to extrapolate P(c|d), discriminative models directly compute that probability.

### Components of a probabalistic ML classifier:
1. Feature representation of the input - vector of features $[x_1, x_2, ..., x_n]$
2. A classification function that computes $\hat{y}$, via $p(y|x)$
3. An objective function for learning
4. An algorithm for optimising the objective function

Logistic Regression has 2 Main Phases:
1. **training** using stochastic gradient descent and the cross-entropy loss
2. **test** Given a test example x we compute p(y|x) and return y=1 or y=0

## The sigmoid function
$$z = (\sum{w_ix_i})+b = \textbf{w}\cdot\textbf{x} + b$$
$$\sigma(z) = \frac{1}{1+e^{-z}}$$
