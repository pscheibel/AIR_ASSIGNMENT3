from torch.utils.data import DataLoader

from TextClassificationModel import TextClassificationModel
import torch
import json
import numpy as np

# classification training based on
# https://medium.com/analytics-vidhya/a-simple-neural-network-classifier-using-pytorch-from-scratch-7ebb477422d2(03.01.2023)
# and https://torchtutorialstaging.z5.web.core.windows.net/beginner/text_sentiment_ngrams_tutorial.html (03.01.2023)
from reader import inputDestination


class ApplyClassification:

    def __init__(self, scientificLabels):
        self.scientificLabels = scientificLabels

    def classifyData(self, data, vocabSize, modelPath):
        num_class = len(self.scientificLabels)
        # print(num_class)
        emsize = 64
        model = TextClassificationModel(vocabSize + 1, emsize, num_class)
        model.load_state_dict(torch.load(modelPath))
        model.eval()
        rankingDictionaries = self.tfIdfClassification(data)
        for fileName, fileContent in data:
            # print("tensor: ", torch.tensor(fileContent, dtype=torch.int64))
            predicted_label = model(torch.tensor(fileContent, dtype=torch.int64), torch.tensor([0]))
            print("file ", fileName, " is ", self.scientificLabels[predicted_label.argmax(1).item()])
            print("cosine similarity for file ", fileName, ": ")
            print(rankingDictionaries[fileName])
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
