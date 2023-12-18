"""
Aufgabenbeschreibung: 

Implementieren Sie eine logistische Regression, die Wörter als stopwords vs. nonstopwords klassifiziert.
Benutzen Sie dazu KEINE externen Libraries außer math.
Sie benötigen dazu den Stochastic Gradient Descent, für den Sie sich am Pseudocode in Figure 5.6 aus Jurafsky & Martin
orientieren sollten.

Der Datensatz besitzt folgende Spalten, getrennt durch Kommata:
[0] = word, [1] = starts_with_vowel, [2] = charlen (Anzahl Buchstaben), [3] = logfreq (logarithmierte Frequenz), [4] = category
Die Spalten [2] und [3] sollen als Inputs (X) für den Klassifizierer fungieren, Spalte [4] soll klassifiziert werden (y).
Dabei soll die Kategorie 'stopword' auf y = 1 und 'nonstopword' auf y = 0 gemappt werden.

Der Datensatz besteht aus 211 Zeilen.
Davon sollen die ersten 160 Zeilen für das Training des Klassifizierers verwendet werden
und die restlichen 51 Zeilen sollen in einem Testdurchlauf klassifiziert werden.


Hinweis zum Datensatz (in dieser Form erstellt von Tom Juzek und zusätzlich geshuffled):
Die Stopwords sind aus NLTK genommen (https://gist.github.com/sebleier/554280),
die Frequenzen aus Googles Unigramkorpus (https://www.kaggle.com/rtatman/english-word-frequency).
Die Daten wurden zu Übungszwecken leicht angepasst.

Folgendes ist gegeben:
Die Learning Rate (eta) ist 0.1.
Die initialen Gewichte w und der initiale Bias b sind jeweils 0.
Die Decision Boundary liegt bei 0.5, d.h. wenn y_estimated > 0.5 wird die Klasse 1 (stopword) vorhergesagt, ansonsten Klasse 0 (nonstopword)

Das Training soll so ablaufen, dass alle 160 Trainingsdatenpunkte einmal der Reihe nach betrachtet werden
(keine Zufallsreihenfolge), also 160 Durchläufe insgesamt. Dabei baut ein Durchlauf auf dem Theta des vorherigen Durchlaufs auf.

Teilen Sie Ihren Code sinnvoll in Funktionen auf und beachten Sie die weiteren Best Practices zu Kommentierung etc.! Der Code soll zudem so
allgemein gehalten sein, dass nicht hartcodiert ist, dass es genau zwei Input Features gibt, sondern der Code soll auch noch funktionieren,
falls mehr Input-Features hinzukommen sollten.
Sollten Sie externe Quellen zurate ziehen, müssen die entsprechenden Stellen als solche eindeutig gekennzeichnet sein. Die Aufgabe gilt allerdings
nur dann als bestanden, wenn die Eigenleistung groß genug ist.


OUTPUT:

Geben Sie einmal ganz zu Beginn und einmal ganz am Ende des Trainings folgendes auf der Konsole aus:

-----------------------------------------------
Initiales Ergebnis (bzw. Finales Ergebnis)             
TP:    ...
TN:    ...
FP:    ...
FN:    ...
ACCURACY: ... (gerundet auf 3 Nachkommastellen)
THETA: ...
-----------------------------------------------

Dabei stehen TP, TN, FP und FN für True Positives, True Negatives, False Positives und False Negatives


#############################################################################

Hilfe zur Selbstüberprüfung

Nach Durchlauf des 1. Trainingsdatenpunkts sollten Sie folgende Werte erhalten:

y_estimated 0.5
Gradient [4.5, 7.864, 0.5]
Updated Theta: [-0.45, -0.7864, -0.05]

Evaluation auf den Testdaten: 
TP: 0
TN: 22
FP: 0
FN: 29
ACCURACY: 0.431


Nach Durchlauf des 2. Trainingsdatenpunkts sollten Sie folgende Werte erhalten:

y_estimated 4.653776764769605e-08
Gradient [-4.999999767311162, -18.543999137003638, -0.9999999534622324]
Updated Theta: [0.049999976731116225, 1.067999913700364, 0.049999995346223236]


Evaluation auf den Testdaten:
TP: 29
TN: 0
FP: 22
FN: 0
ACCURACY: 0.569

##############################################################################

"""

import math

###############################
# Classes
###############################

class LRModel():
    # Construct the Linear Regression Model
    def __init__(self, docsTrain, docsTest, yTrain, yTest):
        # Training and Test Documants and Categories
        self.docsTrain = docsTrain
        self.docsTest = docsTest
        self.yTrain = yTrain
        self.yTest = yTest
        # Weight vector (1xf)
        self.theta = [0 for _ in range(len(self.docsTrain[0]))]
        # Bias 
        self.bias = 0
        
    
    # Fit the model
    def fit(self):
        """Trains the model, given two parallel lists of training documents and training categories"""
        print(f"Initial theta: {self.theta}\nInitial bias: {self.bias}\n")

        for idx, doc in enumerate(self.docsTrain):
            lossGradient = self.compute_gradient(doc, self.yTrain[idx])
            self.compute_new_theta(lossGradient)
            if idx in [0,(len(self.docsTrain)-1)]:
                self.evaluate_model(idx)


    def compute_gradient(self, doc, cat):
        # Compute estimated y value
        yHat = self.compute_y_hat(doc)
        gradient = [(yHat-cat)*x for x in doc]
        gradient.append(yHat-cat)

        return gradient

    def compute_y_hat(self, doc):
        # Multiply each weight with the corresponding feature and sum the results
        # Then add the bias at the end
        zValue = sum(map(lambda w, x: w*x, self.theta, doc)) + self.bias
        yHat = 1/(1+math.exp(-zValue))

        return yHat
        
    def compute_new_theta(self, gradient):
        eta = 0.1
        self.theta = list(map(lambda w, g: w - (eta*g), self.theta, gradient[:-1]))
        self.bias = self.bias - (eta*gradient[-1])

    def evaluate_model(self, idx):
        testCats = self.predict_test_categories()
        trueP, trueN, falseP, falseN = self.compute_confusion_matrix(testCats)
        accuracy, precision, recall, fMeasure = self.compute_evaluations(trueP, trueN, falseP, falseN)
        print("-----------------------------------------------------------------------------")
        if (idx == 0):
            print("Initiales Ergebnis")
        elif (idx == len(self.docsTrain)-1):
            print("Finales Ergebnis")
        print(f"TP: {trueP}")
        print(f"TN: {trueN}")
        print(f"FP: {falseP}")
        print(f"FN: {falseN}")
        print(f"ACCURACY: {round(accuracy,3)}")
        print(f"THETA: {self.theta, self.bias}")
        print("-----------------------------------------------------------------------------\n\n")

    def predict_test_categories(self):
        yHats = [self.compute_y_hat(doc) for doc in self.docsTest]
        return [1 if yHat >= 0.5 else 0 for yHat in yHats]
    
    def compute_confusion_matrix(self, predicted):
        TP, FP, TN, FN = 0,0,0,0
        for (idx, item) in enumerate(predicted):
            if item == 1:
                if item == self.yTest[idx]: TP += 1  
                else: FP += 1
            elif item == 0:
                if item == self.yTest[idx]: TN += 1  
                else: FN += 1
        return (TP, TN, FP, FN)
    
    def compute_evaluations(self, TP, TN, FP, FN):
        all = TP + TN + FP + FN
        #Accuracy = correct / all
        accuracy = (TP + TN)/all
        #Precision = true positive / all system positives
        try:
            precision = TP / (TP + FP)
        except ZeroDivisionError:
            precision = 0
        #Recall = true positive / all gold positives
        recall = TP / (TP + FN)
        #F1-Score
        beta = 1
        try:
            f1 = (((beta*beta)+1)*precision*recall)/(((beta*beta)*precision) + recall)
        except ZeroDivisionError:
            f1 = 0
        
        return accuracy, precision, recall, f1


################################
# Funktionen
################################

def read_data(file):
    """Read out data from a text file with the following columns\\
    [0] = word, [1] = starts_with_vowel, [2] = charlen (Anzahl Buchstaben),
    [3] = logfreq (logarithmierte Frequenz), [4] = category\\
    Returns a dictionary of words as keys and the features as values (in a dictionary)"""

    # Create an empty dictionary to fill
    stopwordDict = {} 

    # Open the input file and read out the contents - the dictionary keys are specific for the data set provided
    # This must be modified if the data set changes
    with open(file, encoding='utf-8', mode='r') as inputFile:
        for line in inputFile:
            lineList = line.strip().split(',')
            stopwordDict[lineList[0]] = {
                'vowelStart' : float(lineList[1]),
                'charlen' : float(lineList[2]),
                'logfreq': float(lineList[3]),
                'category': 1 if lineList[4] == 'stopword' else 0
            }
    return stopwordDict

def train_test_split(featureCols, data, trainSize):
    """Splits the given data into a training and a test set\\
    The set size (of the training set) is decided by the trainSet parameter\\
    Returns 4 lists - documents and classifications for both sets, only containing the predefined feature columns"""

    # Create our empty return lists
    docsTrain = []
    docsTest = []
    yTrain = []
    yTest = []

    # Itterate over the raw data set and split it into training and test data
    for idx, entry in enumerate(data.items()):
        if (idx < trainSize):
            # Append an empty feature array in the training documents
            docsTrain.append([])
            for feature in featureCols:
                # Append each feature into the empty feature array
                docsTrain[idx].append(entry[1][feature])
            # Append the category into the parallel list with the training categories
            yTrain.append(entry[1]['category'])
        else:
            # Do the same as above, but for the test documents and categories
            docsTest.append([])
            for feature in featureCols:
                docsTest[idx-trainSize].append(entry[1][feature])
            yTest.append(entry[1]['category'])

    return (docsTrain, docsTest, yTrain, yTest)


# Funktion, die alle weiteren Funktionen aufruft
def run_script(data_file):
    """Funktion, die alle weiteren Funktionen aufruft"""
    
    # Read out the raw data
    dataSet = read_data(data_file)
    # Define the needed features
    featureCols = ['charlen', 'logfreq']
    # Split our training and test data (without randomization)
    docsTrain, docsTest, yTrain, yTest = train_test_split(featureCols, dataSet, trainSize = 160)

    # Create the LR Model and train it
    RegressionModel = LRModel(docsTrain, docsTest, yTrain, yTest)
    RegressionModel.fit()

    

################################
# Hauptprogramm
################################

if __name__ == "__main__":
    
    data_file = "stopword_corpus_shuffled.txt"
       
    # rufe Funktion auf, die alle weiteren Funktionen aufruft
    run_script(data_file)