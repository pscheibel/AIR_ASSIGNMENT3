from torch.utils.data import DataLoader

from TextClassificationModel import TextClassificationModel
import torch
import json
import numpy as np
import os

# classification training based on
# https://medium.com/analytics-vidhya/a-simple-neural-network-classifier-using-pytorch-from-scratch-7ebb477422d2(03.01.2023)
# and https://torchtutorialstaging.z5.web.core.windows.net/beginner/text_sentiment_ngrams_tutorial.html (03.01.2023)
from reader import inputDestination


class ApplyClassification:

    def __init__(self, scientificLabels):
        self.scientificLabels = scientificLabels

    def classifyData(self, data, vocabSize, modelPath, inputPath):
        num_class = len(self.scientificLabels)
        # print(num_class)
        emsize = 64
        model = TextClassificationModel(vocabSize + 1, emsize, num_class)
        model.load_state_dict(torch.load(modelPath))
        model.eval()
        rankingDictionaries = self.tfIdfClassification(data)
        correctlyPredicted = {}
        correctTfIdf = {}
        predictedLabelResults = {}
        for category in self.scientificLabels.values():
            correctlyPredicted[category] = 0
            correctTfIdf[category] = 0
        for fileName, fileContent in data:
            # print("tensor: ", torch.tensor(fileContent, dtype=torch.int64))
            predicted_label = model(torch.tensor(fileContent, dtype=torch.int64), torch.tensor([0]))
            predicted_label = self.scientificLabels[predicted_label.argmax(1).item()]
            predictedLabelResults[fileName] = predicted_label
            for category in self.scientificLabels.values():
                if category in fileName:
                    if category == predicted_label:
                        correctlyPredicted[category] = correctlyPredicted[category] + 1
                    if category == list(rankingDictionaries[fileName].keys())[0]:
                        correctTfIdf[category] = correctTfIdf[category] + 1
            print("file ", fileName, " is ", predicted_label)
            print("cosine similarity for file ", fileName, ": ")
            print(rankingDictionaries[fileName])
            print("\n")
        print("correctly predicted: ")
        print(correctlyPredicted)
        print("correct with tf-idf: ")
        print(correctTfIdf)
        self.queryDocs(inputPath, predictedLabelResults)
        with open("correctPredictions.txt", 'w') as wordfile:
            wordfile.write(json.dumps(correctlyPredicted))
        with open("correctTF-IDFs.txt", 'w') as wordfile:
            wordfile.write(json.dumps(correctTfIdf))
        return

    def tfIdfClassification(self, data):
        with open('tfIdf.txt') as tfIdf_file:
            tfIdf = json.load(tfIdf_file)
        with open('idf.txt') as idf_file:
            idf = json.load(idf_file)
        categoryVectors = {}
        for category in self.scientificLabels.values():
            categoryVectors[category] = np.array(list(tfIdf[category].values()))
        rankingDictionaries = {}
        perCentDictionaries = {}
        for fileName, fileContent in data:
            rankingDict = {}
            perCentDict = {}
            with open(inputDestination + "/" + fileName) as json_file:
                fileDict = json.load(json_file)
            for item in fileDict.items():
                fileDict[item[0]] = fileDict[item[0]] / max(fileDict.values())
            for category in self.scientificLabels.values():
                comparisonFile = {}
                for word in tfIdf[category]:
                    if word in fileDict:
                        comparisonFile[word] = fileDict[word] * idf[word]
                    else:
                        comparisonFile[word] = 0
                fileVector = np.array(list(comparisonFile.values()))
                rankingDict[category] = np.dot(fileVector, categoryVectors[category]) / (
                        np.linalg.norm(fileVector) * np.linalg.norm(categoryVectors[category]))
            for category in self.scientificLabels.values():
                perCentDict[category] = round((rankingDict[category] / max(rankingDict.values())) * 100)
            rankingDict = dict(sorted(rankingDict.items(), key=lambda item: item[1], reverse=True))
            perCentDict = dict(sorted(perCentDict.items(), key=lambda item: item[1], reverse=True))
            rankingDictionaries[fileName] = rankingDict
            perCentDictionaries[fileName] = perCentDict
        with open("tfIDFRankingDictionaries.txt", 'w') as wordfile:
            wordfile.write(json.dumps(rankingDictionaries))
        with open("perCentRankingDictionaries.txt", 'w') as wordfile:
            wordfile.write(json.dumps(perCentDictionaries))
        return perCentDictionaries

    def queryDocs(self, inputPath, predictedLabels):
        fullpath = inputPath + "queries/queries.txt"
        resultFile = open(str("queriesResult.txt"), "w")
        nounTxtPath = "./nounInput"
        inputFiles = {}
        if os.path.exists(fullpath):
            fileNames = os.listdir(nounTxtPath)
            for fileName in fileNames:
                inputFiles
                with open(nounTxtPath + "/" + fileName) as nounFile:
                    inputFiles[fileName] = json.load(nounFile)
            queriesFile = open(fullpath, 'r').readlines()
            for query in queriesFile:
                foundResult = False
                query = query.strip()
                resultFile.write(query + ":\n")
                parts = query.split(";")
                for key, value in predictedLabels.items():
                    if value == parts[0]:
                        for noun in inputFiles[key]:
                            if parts[1] in noun:
                                foundResult = True
                                resultFile.write(key+"\n")
                                break
                if not foundResult:
                    resultFile.write("No result found for this query...\n")
        else:
            resultFile.write("No queries found!")
        resultFile.close()
