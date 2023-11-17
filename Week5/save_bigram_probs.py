'''
Aufgabe: Bigramm-Übergangswahrscheinlichkeiten aus einem Korpus
als dictionary in einer separaten Datei analog zu nzz_transition_probs.dic speichern.

Hinweise:
- Als Korpus soll die Datei nzz-10000.wpl verwendet werden. Jede Zeile enthält ein Token,
Satzgrenzen sind durch Leerzeilen gekennzeichnet.
- Als Satzgrenzenmarkierer (Padding) für die Bigramm-Übergangswahrscheinlichkeiten soll das Symbol <S>
verwendet werden.
- Es soll kein Smoothing angewandt werden.
'''

from more_itertools import locate

###############
# Funktionen
###############

def extract_tokens(file) -> list:
    tokens = ["<S>"]
    with open(file, encoding="utf-8", mode="r") as f:
        for idx, line in enumerate(f):
            if len(line) != 1:
                tokens.append(line.strip())
            else:
                tokens.append("<S>")
    return tokens

def count_bigrams(tokenList, tokenIndeces) -> dict:

    countDict = {}
    for idx in tokenIndeces:
        if idx < len(tokenList)-1 and tokenList[idx+1] not in countDict:
            countDict[tokenList[idx+1]] = 1
        elif idx == len(tokenList)-1:
            break
        else:
            countDict[tokenList[idx+1]] += 1

    return countDict

def extract_token_frequencies(tokenList) -> dict:
    tokenSet = set(tokenList)
    freqDict = {}
    for token in tokenSet:
        tokenIndeces = list(locate(tokenList, lambda x: x == token))
        freqDict[token] = {}
        tokenCount = len(tokenIndeces)
        bigramCounts = count_bigrams(tokenList, tokenIndeces)

        for bigramTuple in bigramCounts.items():
            freqDict[token][bigramTuple[0]] = bigramTuple[1]/tokenCount
    
    return freqDict

#######################################
# Funktion, die alle weiteren Funktionen aufruft

def run_script(infile):

    tokenList = 4544(infile)
    tokenFrequencies = extract_token_frequencies(tokenList)

    with open('myfreqs.txt', 'w') as f:
        print(tokenFrequencies, file=f)

    



    
###########################################################
# Hauptprogramm
###########################################################

if __name__ == "__main__":
    
    infile = "nzz-10000.wpl"
    
    # Funktion, die alle weiteren Funktionen aufruft
    run_script(infile)


