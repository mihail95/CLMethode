# To predict: predictable with no fun = negative

import math
class NBModel:
    # Construct our Naive Bayes Model Class - alpha is set to 1 by default, log is set to off
    def __init__(self, alpha = 1, log = False):
        self.alpha = alpha
        self.log = log
        self.data = {}
        self.priorProbs = {}
        self.vocabulary = set()
        self.likelihoods = {}
    
    def set_data(self, className:str, input:str) -> None:
        "Set the training data - expects a classname and one or multiple lines of text"

        for line in input.split("\n"):
            self.data.setdefault(className, []).append(line)

    def get_data(self) -> dict: 
        "Get the model data from outside the model as a dictionary"

        return self.data
    
    def train_model(self) -> None:
        "Use the training data to train our model"

        # Initialize some needed variables first
        classNum = {}
        docNum = 0
        wordCounts = {}

        # Count number of documents for each class and gather our vocabulary
        for (className, instances) in self.data.items():
            classNum[className] = len(instances)
            wordCounts[className] = 0
            for instance in instances:
                self.vocabulary.update(instance.split())
                wordCounts[className] += len(instance.split())

        # Set the total number of documents
        for value in classNum.values():
            docNum += int(value)

        # Set our a priori probabilities, use log depending on class parameter self.log (set in the constructor)
        for className in classNum.keys():
            if not self.log:
                self.priorProbs[className] = classNum[className] / docNum
            else: 
                self.priorProbs[className] = math.log10(classNum[className] / docNum)

        # Set our word counts given a class
        for word in self.vocabulary:
            self.likelihoods[word] = {}
            for (className,instances) in self.data.items():
                self.likelihoods[word][className] = 0
                for instance in instances:
                    self.likelihoods[word][className] += instance.count(word)

                # Use the counts to calculate the word probability given each class, use log depending on class parameter self.log (set in the constructor)
                if not self.log: 
                    self.likelihoods[word][className] = (self.likelihoods[word][className] + self.alpha) / (wordCounts[className] + (self.alpha * len(self.vocabulary)))
                else:
                    self.likelihoods[word][className] = math.log10((self.likelihoods[word][className] + self.alpha) / (wordCounts[className] + (self.alpha * len(self.vocabulary))))


    def get_max(self, sums:dict) -> str:
        "Receives a dictionary of class names with counts and returns the class name with the highest count"
        classMax = ""
        currentMax = None
        print(sums)
        for (className, value) in sums.items():
            if currentMax == None or value > currentMax :
                classMax = className
                currentMax = value

        return classMax
    
    def predict_class(self, target) -> str:
        "Make a prediction for a given target, based on the training data"
        classSum = {}
        for className in self.data.keys():
            # Set the initial sum to the a priori probability
            classSum[className] = self.priorProbs[className]
            for word in target.split():
                if word in self.vocabulary: # Ignore words, which are missing in the training data
                    # Calculate the sum probabilities (sum or product depending on log = True / False)
                    if not self.log: 
                        classSum[className] *= self.likelihoods[word][className]
                    else: 
                        classSum[className] += self.likelihoods[word][className]
        
        return self.get_max(classSum)


#######################################
# Funktion, die alle weiteren Funktionen aufruft

def run_script() -> None:
    # This weeks task below: 
    movie_reviews = {
        "just plain boring" : "-",
        "entirely predictable and lacks energy" : "-",
        "no surprises and very few laughs" : "-",
        "very powerful" : "+",
        "the most fun film of the summer" : "+"
        }
    
    # Für dieses Test-Dokument möchten wir die wahrscheinlichste Klasse bestimmen
    test_document = "predictable with no fun"

    # Initialize our class (alpha = 1, log = True)
    naiveBayesModel = NBModel(log = True) 

    for (item, className) in movie_reviews.items():
        naiveBayesModel.set_data(className, item)

    naiveBayesModel.train_model()
    print(f"\"{test_document}\" is (likely) from class: {naiveBayesModel.predict_class(test_document)}")

    # # The commented out section is another way of using the same class, but the data is read from separate files (saved under trainingData/#className#.txt)
    # classes = ["positive", "negative"]
    # # classes = ["action", "comedy"]
    # naiveBayesModel = NBModel() # Initialize our class (alpha = 1, log = False)
    # # This code reads data from text files, named after the classes defined in the comment above
    # for c in classes:
    #     with open(f"trainingData/{c}.txt", encoding="utf-8", mode="r") as f:
    #         file = f.read()
    #         naiveBayesModel.set_data(c, file)
    # naiveBayesModel.train_model()
    # sentenceToPredict = "predictable with no fun"
    # print(f"\"{sentenceToPredict}\" is (likely) from class: {naiveBayesModel.predict_class(sentenceToPredict)}")


###########################################################
# Hauptprogramm
###########################################################

if __name__ == "__main__":

    # Funktion, die alle weiteren Funktionen aufruft
    run_script()