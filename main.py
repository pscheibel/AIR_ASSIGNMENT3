# imports
import Output
import reader
from ApplyClassification import ApplyClassification
from ClassificationNN import ClassificationNN
from TestDataPreparation import TestDataPreparation
import os

CACHING_FILES_ENABLED = True
DATASET_CREATION_ENABLED = True
TRAINING_ENABLED = True
INPUT_PREPARATION_ENABLED = True
DATA_PER_CATEGORY = 535

SCIENTIFICLABELS = {0: "Biology",
                    1: "Computer Science",
                    2: "Electrical Engineering",
                    3: "Economics",
                    4: "Physics"}


def main(loadPdfs, trainNN, prepareInput, inputPath="", modelPath="model.pt", catchingFiles=True, dataPerCategory=1):
    loadPdfs, trainNN, prepareInput, DATA_PER_CATEGORY, CACHING_FILES_ENABLED = readConfig("./input/config/config.txt",
                                                                                           loadPdfs, trainNN,
                                                                                           prepareInput,
                                                                                           dataPerCategory,
                                                                                           catchingFiles)

    preparation = TestDataPreparation()
    training = ClassificationNN()
    application = ApplyClassification(SCIENTIFICLABELS)
    if loadPdfs:
        preparation.executeLoading(CACHING_FILES_ENABLED, SCIENTIFICLABELS, DATA_PER_CATEGORY)
    if trainNN or (not (os.path.exists("lookupdict.txt") and os.path.exists("model.pt") and os.path.exists(
            "tfIdf.txt") and os.path.exists("idf.txt"))):
        trainData, testData, lookupDict = preparation.executePreparation(SCIENTIFICLABELS)
        training.startTraining(trainData, testData, len(lookupDict), SCIENTIFICLABELS, modelPath)
    data, size = reader.readInput(prepareInput, inputPath)
    application.classifyData(data, size, modelPath, inputPath)
    # Output.plotRanking()
    Output.subplotCorrectPredAndTfIdf()


def readConfig(confPath, loadPdfs, trainNN, prepareInput, dataPerCategory, catchingEnabled):
    try:
        queriesFile = open(confPath, 'r').readlines()
        CACHING_FILES_ENABLED = (True if (str(queriesFile[0].split(";")[1].strip().lstrip()) == str("True")) else False)
        DATASET_CREATION_ENABLED = (
            True if (str(queriesFile[1].split(";")[1].strip().lstrip()) == str("True")) else False)
        TRAINING_ENABLED = (True if (str(queriesFile[2].split(";")[1].strip().lstrip()) == str("True")) else False)
        INPUT_PREPARATION_ENABLED = (
            True if (str(queriesFile[3].split(";")[1].strip().lstrip()) == str("True")) else False)
        DATA_PER_CATEGORY = int(queriesFile[4].split(";")[1].strip().lstrip())
        return DATASET_CREATION_ENABLED, TRAINING_ENABLED, INPUT_PREPARATION_ENABLED, DATA_PER_CATEGORY, CACHING_FILES_ENABLED
    except:
        print("Could not load config, use default values.")
        return loadPdfs, trainNN, prepareInput, dataPerCategory, catchingEnabled


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(DATASET_CREATION_ENABLED, TRAINING_ENABLED, INPUT_PREPARATION_ENABLED, "./input/", CACHING_FILES_ENABLED,
         DATA_PER_CATEGORY)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
