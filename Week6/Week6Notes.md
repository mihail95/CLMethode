# Naive Bayes and Sentiment Classification

## Classification
* Assigning a label or category to an input

Uses include:
* Sentiment analysis
* Spam detection
* Language identification
* Authorship attribution
<br><br><br>

## Bayes' theorem
$$P(A)\cdot P(B|A) = P(A\cap B) = P(B)\cdot P(A|B)$$
$$P(A|B) = P_B(A) = \frac{P(A)\cdot P(B|A)}{P(B)} = \frac{P(A\cap B)}{P(B)}$$


## Supervised Machine Learning

Input (**document**): $x\ (or\ d)$\
Output Classes: $Y = \{y_1, y_2, ..., y_m\}$\
Returns: Predicted **class** $y\ (or\ c)\in Y$

Training set: N documents labeled with class\
$\{(d_1, c_1), ..., (d_N, c_N)\}$

Goal: Learn a classifier capable to map form a new document to its class\
A **probabalistic classifier** will additionaly tell us the probability\
$d\rightarrow c \in C$, where $C$ is a set of useful document classes

Algorithms to build classifiers:\
**Generative** (model of how a class could generate input data) vs **Discriminative** (learn what features differentiate different classes)<br><br><br>

## Naive Bayes Classifiers
### Multinomial Naive Bayes Classifier (Bag of Words)

* Text document is represented as an unordered set of words (keeping track of frequencies)

****
For $d$ the classifier returns the class $\hat{c}$, which has the maximum posterior probability given the document

$$\hat{c} = \underset{c\in C}{argmaxP}(c|d)$$

Applying Bayes' rule:
$$\hat{c} = \underset{c\in C}{argmaxP}(c|d) = \underset{c\in C}{argmax}\frac{P(d|c)P(c)}{P(d)}$$

We can simplify, bacause $P(d)$ is constant:
$$\hat{c} = \underset{c\in C}{argmaxP}(c|d) = \underset{c\in C}{argmax}P(d|c)P(c)$$

****
Naive Bayes is a **Generative Model**\
$P(c)$  is the prior probability of the class\
$P(d|c)$ is the likelihood of the document

We can express the likelihood as a set of features:\
$P(d|c) = P(f_1, f_2, ..., f_n|c)$<br><br>

*Simplifying assumptions:*
* Position doesn't matter
* naive Bayes assumption: $P(f_1, f_2, ..., f_n|c) = \underset{f\in F}\prod{P(f_i|c)}$
***
***Application to a document***:
$$c_{NB} = \underset{c\in C}{argmax}\ logP(c) + \underset{i\in positions}\sum{logP(w_i|c)}$$
***
Features in log space => prediction is a linear function of input features => **lienar classifier**

### Training the Naive Bayes Classifier
1. Learn the probability $P(c)$:\
$N_c$ is the number of documents with the class $c$\
$N_{doc}$ is the total number of documents
$$\hat{P}(c) = \frac{N_c}{N_{doc}}$$

2. Assume a feature is just the existence of a word and learn $P(w_i|c)$:\
$c$ is a category like 'positive' or 'negative' (aggregated from all documents)\
$w_i$ is our target word\
denominator is the sum of all positive words\
$V$ is an union of all the word types in all classes
$$\hat{P}(w_i|c) = \frac{count(w_i, c)}{\sum_{w\in V}{count(w,c)}}$$

3. Laplace Smoothing ($\alpha = 1$ per default) to avoid Zero-Probabilities (causing a class probability of 0)
$$\hat{P}(w_i|c) = \frac{count(w_i, c) + \alpha}{(\sum_{w\in V}{count(w,c)}) + \alpha\cdot|V|}$$

4. Ignore **unknown words**; Remove them from the test document

5. (Maybe) Ignore **stop words**

<br><br>
### Optimizing for Sentiment Analysis
* Set the maximum frequency to 1 **per document** (binary naive Bayes)
* Deal with negation - prepend 'NOT_' to every word after a negation token (n’t, not, no, never)  until next punctuatuion mark
* Use sentiment lexicons
<br><br>
### Naive Bayes for other text classification tasks
#### Spam Detection
* Predefine sets of phrases/features (also not purely linguistic)
#### Language ID
Features: character n-grams /  byte n-grams

<br>

### Naive Bayes as a Language Model
$$P(sentence|c)=\underset{i\in positions}\prod{P(w_i|c)}$$
<br>

### Evaluation: Precision, Recall, F-measure
**gold labels** = human-defined labels for each document that we are trying to match

Steps:
1. Build **confusion matrix**
2. Precision = TP / TP + FP
3. Recall = TP / TP + FN
4. F-Measure (β > 1 favors recall, β < 1 favors precision)
$$F_{\beta} = \frac{(\beta^2+1)PR}{\beta^2P+R}$$
$$F_{\beta = 1} = F_1 = \frac{2PR}{P +R}$$

