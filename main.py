# imports
import Output
import reader
from ApplyClassification import ApplyClassification
from ClassificationNN import ClassificationNN
from TestDataPreparation import TestDataPreparation
import os

CACHING_FILES_ENABLED = True

SCIENTIFICLABELS = {0: "Biology",
                    1: "Computer Science",
                    2: "Electrical Engineering",
                    3: "Economics",
                    4: "Physics"}


def main(loadPdfs, trainNN, prepareInput, inputPath="", modelPath="model.pt"):
    preparation = TestDataPreparation()
    training = ClassificationNN()
    application = ApplyClassification(SCIENTIFICLABELS)
    if loadPdfs:
        preparation.executeLoading(CACHING_FILES_ENABLED, SCIENTIFICLABELS)
    if trainNN or (not (os.path.exists("lookupdict.txt") and os.path.exists("model.pt") and os.path.exists("tfIdf.txt") and os.path.exists("idf.txt"))):
        trainData, testData, lookupDict = preparation.executePreparation(SCIENTIFICLABELS)
        training.startTraining(trainData, testData, len(lookupDict), SCIENTIFICLABELS, modelPath)
    data, size = reader.readInput(prepareInput, inputPath)
    application.classifyData(data, size, modelPath)
    Output.subplotCorrectPredAndTfIdf()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(False, False, False, "./input/")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
