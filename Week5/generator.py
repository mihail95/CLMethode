'''Aufgabe:

Ein Programm 'generator.py' schreiben, das, gegeben ein Inputwort, Text generiert.
Dabei sollen Bigramm-Übergangswahrscheinlichkeiten von Wörtern Anwendung finden, die 
aus einem Korpus gelernt wurden und für die Aufgabe bereits vorgegeben sind. Die Übergangs-
wahrscheinlichkeiten beinhalten kein Smoothing.

Folgendes soll beachtet werden:
- Ein:e Nutzer:in soll das Startwort eingeben können. Sollte dies ein Wort sein, das nicht
im Korpus vorhanden ist, soll das Programm ein zufälliges Beispielwort ausgeben und den:die
Nutzer:in erneut zur Eingabe eines Wortes auffordern
- Das folgende Wort soll zufällig, aber gemäß den gegebenen Übergangswahrscheinlichkeiten
ausgewählt werden, und zwar nach dem Verfahren, wie in Abbildung 3.3 in Jurafsky & Martin gezeigt.
- Der generierte Satz soll enden, wenn ein Satzgrenzen-Symbol (<S>) erreicht wurde oder der
Satz 20 Wörter lang ist.

'''

import ast
import random

###############
# Funktionen
###############


def create_transitions_dict(transitions) -> dict:
    """
    Creates a dictionary out ouf our transitions\\
    Key is the word itself, value is a dictionary of probabilities
    """
    transitionsDict = {}
    for transition in transitions.items():
        transitionsDict[transition[0]] = transition[1]
    
    return transitionsDict

def choose_start_word(transitionsDict:dict) -> tuple:
    """
    Prompts the user to choose a starting word\\
    If the word is not in our probability dictionary, we ask again and give a suggestion
    Returns a tuple with the current word as key and its possible bigram-pairs as values
    """
    userInput = input("Please enter a word: ")
    while userInput not in transitionsDict.keys():
        suggestion = random.choice(list(transitionsDict.keys()))
        userInput = input(f"Your word is unavailable! Please choose another (Suggestion: '{suggestion}'): ")
    
    return (userInput, transitionsDict[userInput])

def choose_next_word(transitionsDict:dict, possibleFollowers:dict) -> tuple:
    """
    Chooses the next word for our generated sentence\\
    Returns a tuple with the word as key and its possible bigram-pairs as values
    """
    sequentialFollowers = sort_followers(possibleFollowers)
    randValue = random.uniform(0, 1)

    for freq in sequentialFollowers.keys():
        if randValue <= freq:
            return (sequentialFollowers[freq], transitionsDict[sequentialFollowers[freq]])
    

def sort_followers(possibleFollowers:dict) -> dict:
    """
    Sorts the follower dictionary by value\\
    Returns a sorted dictionary with the accumulated probabilities as keys and the words as values
    """
    # Sorts our possibleFollowers according to their frequency in reverse order
    # The lambda function is used as a shorthand instead of declaring a separate function to return the value of the current item
    sortedFollowers = dict(sorted(possibleFollowers.items(), key=lambda follower: follower[1], reverse=True))
    sequentialFrequencies = {}
    freqSum = 0
    for (follower, frequency) in sortedFollowers.items():
        freqSum += frequency
        sequentialFrequencies[freqSum] = follower
    
    return sequentialFrequencies

def generate_sentence(dictionary:dict, tokenLimit:int) -> str:
    """
    Generates a sentence on the basis of a dictionary of bigram probabilities\\
    Returns a string of randomly (based on probabilities) generated words/tokens
    """
    sentence = ""
    currentWord = choose_start_word(dictionary)
    sentence += currentWord[0] + " "
    
    # Maximal Tokens = tokenLimit - 1, because we already chose a starting word
    for ctr in range(tokenLimit-1):
        currentWord = choose_next_word(dictionary, currentWord[1])
        if currentWord[0] == "<S>":
            break
        sentence += currentWord[0] + " "

    return sentence

#######################################
# Funktion, die alle weiteren Funktionen aufruft

def run_script() -> None:

    # 1. Einlesen der Übergangswahrscheinlichkeiten
    # dict transitions:
    # key: erstes Wort
    # val: dict with key: nächstes Wort, val: Wahrscheinlichkeit
    with open("nzz_transition_probs.dic", encoding="utf-8", mode="r") as f:
        file = f.read()

    transitions = ast.literal_eval(file)

    # Build a dictionary out of our transitions
    transitionsDict = create_transitions_dict(transitions)
    # Beispiel: print(transitionsDict["beleuchtet"])

    # Generate a sentence with a given token limit
    print(generate_sentence(dictionary = transitionsDict, tokenLimit = 20))



    
###########################################################
# Hauptprogramm
###########################################################

if __name__ == "__main__":
    
    
    # Funktion, die alle weiteren Funktionen aufruft
    run_script()



