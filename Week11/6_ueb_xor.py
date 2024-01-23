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

def z_transform(biasWeight, weights, inputs):
    """Return the dot product of the given weights and bias vectors plus the weighted given bias"""
    # Bias is always one in this example, but I included it nevertheless
    x0 = 1
    return sum(map(lambda w, x: x*w, weights, inputs)) + (x0 * biasWeight)

def validate_bool(input):
    """Checks if the given input is a valid boolean value, otherwise throws a ValueError()"""
    if input in (0,1):
        return input
    else: raise ValueError()

def get_bool_inputs():
    """Return 2 bools - x1 and x2. Only valid inputs are 1 or 0"""
    # This can be written in a more generalized way, but I'll leave it as it is for now
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
    # Ask the user for input and validate
    x1, x2 = get_bool_inputs()

    # Concatinate the inputs in an input 'vector'
    x = [x1, x2]
    print("Inputs: ", x)

    # Set the initial input (W) and bias(b) weights
    # Dimensions of W are dependent on the ammout of neurons in the h-layer and the lenght of the input vector (2x2 in this case)
    W = [[1,1],[1,1]]
    # The bias vector has as many members as there are neurons in the h-layer (2 in this case)
    b_hidden = [0, -1]

    # How many hidden layer neurons are there?
    h_size = 2
    # Compute the hidden layer neurons
    h = [ relU(z_transform(b_hidden[idx], W[idx], x)) for idx in range(h_size) ]
    print("Hidden layer: ", h)

    # Set the input and bias weights for the output layer
    U = [[1, -2]] # Dimensions 1x2, since only 1 output neuron
    b_output = [0]

    # How many output neurons are there?
    y_size = 1
    # Compute the output layer
    y = [ relU(z_transform(b_output[idx], U[idx], h)) for idx in range(y_size) ]
    print("Output layer: ", y)

################################
# Hauptprogramm
################################

if __name__ == "__main__":

    run_script()