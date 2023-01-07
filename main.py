# imports
from ClassificationNN import ClassificationNN
from TestDataPreparation import TestDataPreparation

CACHING_FILES_ENABLED = True

SCIENTIFICLABELS = {0: "Biology",
                    1: "Computer Science",
                    2: "Electrical Engineering",
                    3: "Mathematics",
                    4: "Phyiscs"}


def main(loadPdfs, trainNN):
    preparation = TestDataPreparation()
    training = ClassificationNN()
    if loadPdfs:
        preparation.executeLoading(CACHING_FILES_ENABLED, SCIENTIFICLABELS)
    if trainNN:
        trainData, testData, lookupDict = preparation.executePreparation(SCIENTIFICLABELS)
        training.startTraining(trainData, testData, len(lookupDict), SCIENTIFICLABELS)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(False, True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
