from torch.utils.data import DataLoader

from TextClassificationModel import TextClassificationModel
import torch
import time

# classification training based on
# https://medium.com/analytics-vidhya/a-simple-neural-network-classifier-using-pytorch-from-scratch-7ebb477422d2(03.01.2023)
# and https://torchtutorialstaging.z5.web.core.windows.net/beginner/text_sentiment_ngrams_tutorial.html (03.01.2023)

class ApplyClassification:

    def __init__(self, scientificLabels):
        self.scientificLabels = scientificLabels

    def classifyData(self, data, vocabSize, modelPath):
        num_class = len(self.scientificLabels)
        # print(num_class)
        emsize = 64
        model = TextClassificationModel(vocabSize+1, emsize, num_class)
        model.load_state_dict(torch.load(modelPath))
        model.eval()
        for fileName, fileContent in data:
            #print("tensor: ", torch.tensor(fileContent, dtype=torch.int64))
            predicted_label = model(torch.tensor(fileContent, dtype=torch.int64), torch.tensor([0]))
            print("file ", fileName, " is ", self.scientificLabels[predicted_label.argmax(1).item()])
        return
