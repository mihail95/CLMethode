"""Aufgabe: 
Ein Programm schreiben, das einen Satzsegmentierer fuer deutsche Texte implementiert.

Das Programm soll 2 Argumente haben: Inputdatei und Datei mit einem Abkürzungslexikon.
Input: "roher" Text
Output: eine segmentierte Fassung des Textes im Format 'one sentence per line'

Bsp:
- Input: Das ist ein Satz. Und noch ein zweiter.
- Output: 
Das ist ein Satz.
Und noch ein zweiter.

Als Hilfsmittel kann das Abkuerzungslexikon 'abbrev.lex' eingesetzt werden.

Nutzen Sie das vorgegebene Programmgerüst und teilen Sie Ihr Programm sinnvoll in Funktionen auf.

"""
import re

###############################################
# FUNKTIONEN
###############################################

###############################################
# Builds an abbrevialtion lexicon
###############################################
def build_abbrev_lex(input):
    lexicon = []
    for line in input:
        lexicon.append(line.replace("\n",""))
    return lexicon

###############################################
# Nimmt den Text als inuput und ersetzt alle abbr. durch einen index aus 1_abbrev
###############################################
def remove_abrreviations(input, lexicon):
    textWithoutAbbr = []
    for line in input:
        for (idx, token) in enumerate(line.split()):
            if ("." in token and not line.split()[idx+1][0].isupper()):
                lexicon.append(token)
            elif("." in token and len(token)<=5):
                lexicon.append(token)
                
            if token in lexicon:
                token = '#AbbrevN' + str(lexicon.index(token)) + '#'
            textWithoutAbbr.append(token)
    
    return (' '.join(textWithoutAbbr), lexicon)

###############################################
# Nimmt den bearbeiteten und splittet ihm in Sätze.
###############################################
def split_sentences(input):
    output = []
    currentSentence = ''
    for word in input.split():
        currentSentence += word + ' '
        if (("." in word or "?" in word or "!" in word) and '"' not in word):
            output.append(currentSentence)
            currentSentence = ''
    
    return output

###############################################
# Nimmt den bearbeiteten Text als inuput und ersetzt alle abbr. Indexes durch den Abbrev.
###############################################
def add_abbreviations(input, lexicon):
    newTextArray = []
    for sentence in input:
        newSentence = ''
        for token in sentence.split():
            res = re.search(r"#AbbrevN(\d+)#", token)
            if (res != None):
                token = lexicon[int(res.group(1))]
            newSentence += token + ' '
        newTextArray.append(newSentence)

    return newTextArray


##############################################
# Funktion, die alle weiteren Funktionen aufruft

def run_script(input_text, input_lex):
    with open(input_lex, encoding="utf-8") as input:
        abbreviationLexicon = build_abbrev_lex(input)

    with open(input_text, encoding="utf-8") as input:
        processedText, abbreviationLexicon = remove_abrreviations(input, abbreviationLexicon)

    splitSentencesArray = split_sentences(processedText)
    finalText = add_abbreviations(splitSentencesArray, abbreviationLexicon)

    print(finalText)
    
        

###########################################################
# Hauptprogramm
###########################################################

if __name__ == "__main__":
    
    input_text = "data/1_bsp.txt"
    input_lex = "data/1_abbrev.lex"
    
    # Funktion, die alle weiteren Funktionen aufruft
    run_script(input_text, input_lex)
