'''Aufgabe:

Ein Programm  schreiben, das die Levenshtein-Distanz von einem Source-String zu einem Target-string berechnet.

Das Programm soll 2 Argumente haben: Source und Target String
Input:  2 Strings
Output: Distanz der beiden Strings

Orientieren sollte man sich an dem Pseudocode aus dem Lehrbuch (Abb. 2.17)


Nutzen Sie das vorgegebene Programmgerüst und teilen Sie Ihr Programm sinnvoll in Funktionen auf.
'''

import numpy as np
###############################################
# FUNKTIONEN
###############################################

###############################################
# Prints a 2D array in a readable format
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
    print(f"Dimensions: n:{dimensions[0]}; m:{dimensions[1]}\n")
    costMatrix = np.empty(shape=dimensions)
    costMatrix.fill(0)
    for rowNum in range(1,dimensions[0]):
        costMatrix[rowNum, 0] = costMatrix[rowNum-1, 0] + 1

    for colNum in range(1,dimensions[1]):
        costMatrix[0, colNum] = costMatrix[0, colNum-1] + 1
    return costMatrix

###############################################
# Goes through the initialized Cost Matrix and calculates Levenstein Distances
# Input: string source; string target; 2D Array costMatrix - in initialized format
# Output: costMatrix filled with the correct costs
###############################################
def calculate_costs(source, target, dimensions, costMatrix):
    for rowIndex in range(1, dimensions[0]):
        for colIndex in range(1, dimensions[1]):
            delCost = costMatrix[rowIndex-1, colIndex]+1 # Item Above + 1 (delCost)
            subCost = costMatrix[rowIndex-1, colIndex-1]+(0 if source[rowIndex-1] == target[colIndex-1] else 2)
            insCost = costMatrix[rowIndex, colIndex-1]+1 # Item to the Left + 1 (insCost)
            costMatrix[rowIndex, colIndex] = min(delCost,subCost,insCost)
    return costMatrix


#######################################
# Funktion, die alle weiteren Funktionen aufruft

def run_script(source:str, target:str) -> None:
    dimensions = (len(source)+1, len(target)+1)
    costMatrix = initialize_matrix(dimensions)
    costMatrix = calculate_costs(source, target, dimensions, costMatrix)
    show_matrix(costMatrix)
    print(f"\nLevenstein Distance: {costMatrix[dimensions[0]-1, dimensions[1]-1]}")

###########################################################
# Hauptprogramm
###########################################################

if __name__ == "__main__":
    
    source = "ROT"
    target = "TORE"
    
    # Funktion, die alle weiteren Funktionen aufruft
    run_script(source, target)
