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



################################
# Funktionen
################################




# Funktion, die alle weiteren Funktionen aufruft
def run_script(data_file):
    """Funktion, die alle weiteren Funktionen aufruft"""
    
    ...
    

################################
# Hauptprogramm
################################

if __name__ == "__main__":
    
    data_file = "stopword_corpus_shuffled.txt"
       
    # rufe Funktion auf, die alle weiteren Funktionen aufruft
    run_script(data_file)