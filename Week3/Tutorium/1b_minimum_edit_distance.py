'''Aufgabe:

Ein Programm  schreiben, das die Levenshtein-Distanz von einem Source-String zu einem Target-string berechnet.

Das Programm soll 2 Argumente haben: Source und Target String
Input:  2 Strings
Output: Distanz der beiden Strings

Orientieren sollte man sich an dem Pseudocode aus dem Lehrbuch (Abb. 2.17)


Nutzen Sie das vorgegebene Programmgerüst und teilen Sie Ihr Programm sinnvoll in Funktionen auf.
'''

###############################################
# FUNKTIONEN
###############################################

###############################################
# Prints a 2D array in a readable fashion
###############################################
def show_matrix(matrix):
    for row in matrix:
        print(row)

###############################################
# Initializes the cost-matrix
# Input: tuple dimensions - (rows, columns)
# Output: 2D Array - costMatrix
###############################################
def initialize_matrix(dimensions):
    costMatrix = [[0]*(dimensions[1]+1)]*(dimensions[0]+1)


    return costMatrix


#######################################
# Funktion, die alle weiteren Funktionen aufruft

def run_script(source:str, target:str) -> None:
    dimensions = (len(source), len(target))
    costMatrix = initialize_matrix(dimensions)
    show_matrix(costMatrix)
    ...

###########################################################
# Hauptprogramm
###########################################################

if __name__ == "__main__":
    
    source = "ROT"
    target = "TORE"
    
    # Funktion, die alle weiteren Funktionen aufruft
    run_script(source, target)
