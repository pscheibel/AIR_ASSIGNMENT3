# imports
import Output
import reader
from ApplyClassification import ApplyClassification
from ClassificationNN import ClassificationNN
from TestDataPreparation import TestDataPreparation

CACHING_FILES_ENABLED = True

SCIENTIFICLABELS = {0: "Biology",
                    1: "Computer Science",
                    2: "Electrical Engineering",
                    3: "Mathematics",
                    4: "Phyiscs"}


def main(loadPdfs, trainNN, prepareInput, inputPath="", modelPath="model.pt"):
    preparation = TestDataPreparation()
    training = ClassificationNN()
    application = ApplyClassification(SCIENTIFICLABELS)
    if loadPdfs:
        preparation.executeLoading(CACHING_FILES_ENABLED, SCIENTIFICLABELS)
    if trainNN:
        trainData, testData, lookupDict = preparation.executePreparation(SCIENTIFICLABELS)
        training.startTraining(trainData, testData, len(lookupDict), SCIENTIFICLABELS, modelPath)
    data, size = reader.readInput(prepareInput, inputPath)
    application.classifyData(data, size, modelPath)
    #Output.plotRanking()
    Output.subplotCorrectPredAndTfIdf()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(False, False, False, "./input/")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
