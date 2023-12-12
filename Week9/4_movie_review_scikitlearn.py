"""
Aufgabenbeschreibung: 
Gegeben ist der Datensatz movie_reviews_10k.csv, der 10.000 Filmkritiken enthält, die jeweils als "positive" oder
"negative" klassifiziert wurden. Aufgabe: Erstellen Sie mithilfe der Library scikit-learn zwei Klassifikatoren,
die jeweils die ersten 80% des Datensatzes als Trainingsdaten und die restlichen Daten als Testdaten
(jeweils ohne Shuffle!) nutzen:

a)  einen Naive Bayes Klassifikator mit Bag-of-Word Features

b)  einen Logistic Regression Klassifikator mit den Features, die auf S. 83 in Jurafsky & Martin verwendet werden.
    Dies sind pro Dokument:
    - Anzahl positiver Lexikonwörter
    - Anzahl negativer Lexikonwörter
    - ob "no" enthalten ist
    - Anzahl Pronomen der 1. und 2. Person
    - ob "!" enthalten ist
    - Anzahl Tokens (logarithmiert mit der Funktion math.log)

Als Sentiment Lexikon steht die Datei WKWSCISentimentLexicon_v1.1.xlsx zur Verfügung. Darin stehen negative Werte für
ein negatives Sentiment und positive Werte für ein positives Sentiment.

Für jeden der beiden Klassifikatoren soll am Ende die Accuracy auf dem Testset berechnet und ausgegeben werden.

Neben scikit-learn dürfen beliebige weitere Libraries verwendet werden, z.B. für die Tokenisierung.

Zusatzaufgabe (Challenge): Optimieren Sie das Klassifikationsergebnis, indem Sie z.B. die Features verändern.
Achtung: Sie dürfen dazu mehrfach auf dem gleichen Testset testen, was in der Praxis eigentlich nicht erlaubt wäre.
Verwenden Sie für die Zusatzaufgabe ein neues Python-Skript, sodass das Skript mit den "Original-Features" als HA
gewertet werden kann.

Datenquellen:
movie_reviews_10k.csv: die ersten 10.000 Einträge aus diesem Datensatz: https://github.com/SK7here/Movie-Review-Sentiment-Analysis/blob/master/IMDB-Dataset.csv
WKWSCISentimentLexicon_v1.1.xlsx: https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/DWWEBV
"""
from pathlib import Path
import nltk
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

################################
# Funktionen
################################

def evaluate_NBModel(data):
    """
    Evaluates a Multinomial NBModel using the given dataframe\\
    Returns the accuracy of the trained NBModel
    """
    # Split our training and test data (without randomization)
    docs_train, docs_test, y_train, y_test = train_test_split(data['review'], data['sentiment'], shuffle=False,
                                                          train_size = 0.80)

    # Initialize a CountVectorizer (with the parameters from the Notebook)
    movieVzer= CountVectorizer(min_df=2, max_features=3000) # use top 3000 words only.

    # Fit and transform the training data
    docs_train_counts = movieVzer.fit_transform(docs_train)

    NBModel = MultinomialNB()
    
    # Train the model on our train features and train classes 
    NBModel.fit(docs_train_counts, y_train)

    # Only transform (without fitting) our test data - this provides the counts for features that already exist in the model
    docs_test_counts = movieVzer.transform(docs_test)

    # Predict the classes for those counts
    y_predict = NBModel.predict(docs_test_counts)

    # Evaluate the accuracy of our model and return the result
    NBAccuracy = accuracy_score(y_test, y_predict)

    return NBAccuracy


def prepare_data(data, lexicon):
    """Calculate the needed features and write them to the provided dataframe"""

    # Create an empty data frame with our needed column names (not sure if needed)
    preparedData = pd.DataFrame(columns = ['sentiment', 'pos_count', 'neg_count', 'contains_no', 'pron_count', 'contains_exclam', 'token_log'])

    # Calculate our features (hopefully works without having to itterate over the DF rows) 
    preparedData['sentiment'] = data['sentiment']
    preparedData['pos_count'] = count_word_sentiments(data['review'], lexicon, sentimentType = 'positive')
    preparedData['neg_count'] = count_word_sentiments(data['review'], lexicon, sentimentType = 'negative')
    preparedData['contains_no'] = determine_contains_no(data['review'])
    preparedData['pron_count'] = count_pronouns(data['review'])
    preparedData['contains_exclam'] = determine_contains_exclam(data['review'])
    preparedData['token_log'] = count_tokens(data['review'])

    return preparedData

def count_word_sentiments(series, lexicon, sentimentType):
    ...

def determine_contains_no(series):
    ...

def count_pronouns(series):
    ...

def determine_contains_exclam(series):
    ...

def count_tokens(series):
    ...

def evaluate_LRModel(data, lexicon):
    """
    Evaluates a Linear Regression Model using the given dataframe\\
    Returns the accuracy of the trained LRModel
    """
    
    # Prepare our data before we train the model
    preparedData = prepare_data(data, lexicon)

    # Define our feature columns
    feature_cols = ['pos_count', 'neg_count', 'contains_no', 'pron_count', 'contains_exclam', 'token_log']

    # Split our features and target
    docs = preparedData[feature_cols]
    y = preparedData['sentiment']
    docs_train, docs_test, y_train, y_test = train_test_split(docs, y, train_size=0.80, shuffle=False)

    # Train the model
    LRModel = LogisticRegression(random_state=16)
    LRModel.fit(docs_train, y_train)

    # Predict the test classifications
    y_pred = LRModel.predict(docs_test)

    # Evaluate the model accuracy
    LRAccuracy = accuracy_score(y_test, y_pred)

    return LRAccuracy

# Funktion, die alle weiteren Funktionen aufruft
def run_script(movie_reviews, sentiment_lexicon):
    """Funktion, die alle weiteren Funktionen aufruft
    :param movie_reviews: Dateiname der Reviews
    :param sentiment_lexicon: Dateiname des Sentiment Lexikons
    """
    # Read the movie reviews as a dataFrame
    movieReviewsDF = pd.read_csv(movie_reviews)

    # # Train our NB Model and get its accuracy
    # NBModelAccuracy = evaluate_NBModel(movieReviewsDF)
    # print(f"NBModel Accuracy is: {NBModelAccuracy*100}%")

    # Read the sentiment lexicon (Sheet 'WKWSCI sentiment lexicon no POS') as a dataframe
    sentimentLexiconDF = pd.read_excel(sentiment_lexicon, sheet_name='WKWSCI sentiment lexicon no POS')

    # Train our Regression Model and get its accuracy
    LRmodelAccuracy = evaluate_LRModel(movieReviewsDF, sentimentLexiconDF)

################################
# Hauptprogramm
################################

if __name__ == "__main__":

    movie_reviews = Path("movie_reviews_10k.csv")
    sentiment_lexicon = Path("WKWSCISentimentLexicon_v1.1.xlsx")
    
    # rufe Funktion auf, die alle weiteren Funktionen aufruft
    run_script(movie_reviews, sentiment_lexicon)



