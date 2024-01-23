"""
Aufgabenstellung: Implementieren Sie die XOR-Lösung aus Abbildung 7.6 aus Jurafsky & Martin mit den dort angegebenen Gewichten und ReLu
als Aktivierungsfunktion. x1 und x2 sollen dabei vom User eingegeben werden und der Output y auf der Konsole ausgegeben werden. Orientieren Sie sich an der
Berechnung aus Figure 7.10. Der Bias wird repräsentiert als Input x0 mit Wert 1.
Benutzen Sie  x0, x1, x2, x, W, U, h, y so wie in der Figure angegeben für Ihre Variablennamen.
Es sollen keine externen Libraries benutzt werden!
"""

# Funktionen
def relU(z):
    """Apply the relU activation function to a z-Value"""
    return max(0, z)

def z_transform(bias, weights, inputs):
    """Return the dot product of the given weights and bias vectors plus the given bias"""
    return sum(lambda w, x: x*w, weights, inputs) + bias

def validate_bool(input):
    if input in (0,1):
        return input
    else: raise ValueError()

def get_bool_inputs():
    """Return 2 bools - x1 and x2. Only valid inputs are 1 or 0"""
    x1 = None
    while True:
        try:
            x1 = validate_bool(int(input("Please enter a bool value for x1: ")))
            break
        except ValueError:
            print("The excepted values are only 1 or 0! Please try again!")

    x2 = None
    while True:
        try:
            x2 = validate_bool(int(input("Please enter a bool value for x2: ")))
            break
        except ValueError:
            print("The excepted values are only 1 or 0! Please try again!")

    return x1, x2
    


# Funktion, die alle weiteren Funktionen aufruft
def run_script():
    """Funktion, die alle weiteren Funktionen aufruft"""
    # Define the bias as x0 (1 per default)
    x0 = 1

    # Ask the user for input and validate
    x1, x2 = get_bool_inputs()


    # Concatinate the inputs in an input 'vector'
    x = [x1, x2]

    print(x)


################################
# Hauptprogramm
################################

if __name__ == "__main__":

    run_script()