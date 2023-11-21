# To predict: predictable with no fun = negative

class NBModel:
    def __init__(self, alpha = 1):
        self.alpha = alpha
        self.data = {}
        self.priorProbs = {}
        self.vocabulary = set()
        self.likelihoods = {}
    
    def set_data(self, className:str, file:str) -> None:
        for line in file.split("\n"):
            self.data.setdefault(className, []).append(line)

    def get_data(self) -> dict: 
        return self.data
    
    def train_model(self) -> None:
        classNum = {}
        docNum = 0
        wordCounts = {}
        # Number of documents for each class and vocabulary
        for (c, instances) in self.data.items():
            classNum[c] = len(instances)
            wordCounts[c] = 0
            for instance in instances:
                self.vocabulary.update(instance.split())
                wordCounts[c] += len(instance.split())

        # Total number of documents
        for value in classNum.values():
            docNum += int(value)

        # A priori probabilities
        for c in classNum.keys():
            self.priorProbs[c] = classNum[c] / docNum

        # Word probabilities given class
        for word in self.vocabulary:
            self.likelihoods[word] = {}
            for (c,instances) in self.data.items():
                self.likelihoods[word][c] = 0
                for instance in instances:
                    self.likelihoods[word][c] += instance.count(word)
                self.likelihoods[word][c] = (self.likelihoods[word][c] + self.alpha) / (wordCounts[c] + (self.alpha* len(self.vocabulary)))

    def get_max(self, sums:dict) -> str:
        classMax = ""
        currentMax = 0
        for (c, value) in sums.items():
            if value > currentMax:
                classMax = c

        return classMax
    
    def predict_class(self, target) -> str:
        classSum = {}
        for c in self.data.keys():
            classSum[c] = self.priorProbs[c]
            for word in target.split():
                if word in self.vocabulary:
                    classSum[c] *= self.likelihoods[word][c]
        
        return self.get_max(classSum)


#######################################
# Funktion, die alle weiteren Funktionen aufruft

def run_script() -> None:
    # classes = ["positive", "negative"]
    classes = ["action", "comedy"]
    naiveBayesModel = NBModel()

    for c in classes:
        with open(f"trainingData/{c}.txt", encoding="utf-8", mode="r") as f:
            file = f.read()
            naiveBayesModel.set_data(c, file)
    
    naiveBayesModel.train_model()
    sentenceToPredict = "fast couple shoot fly"
    print(f"\"{sentenceToPredict}\" is from class: {naiveBayesModel.predict_class(sentenceToPredict)}")



###########################################################
# Hauptprogramm
###########################################################

if __name__ == "__main__":
    
    
    # Funktion, die alle weiteren Funktionen aufruft
    run_script()