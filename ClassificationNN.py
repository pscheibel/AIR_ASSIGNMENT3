from torch.utils.data import DataLoader

from Output import plotAccuracyPerEpoch, plotDocumentsPerTopic, subplots
from TextClassificationModel import TextClassificationModel
import torch
import time

# classification training based on
# https://medium.com/analytics-vidhya/a-simple-neural-network-classifier-using-pytorch-from-scratch-7ebb477422d2(03.01.2023)
# and https://torchtutorialstaging.z5.web.core.windows.net/beginner/text_sentiment_ngrams_tutorial.html (03.01.2023)

device = "cpu"


class ClassificationNN:

    def startTraining(self, trainData, testData, vocabSize, scientificLabels, modelPath):
        #print(trainData)
        trainDataloader = DataLoader(trainData, batch_size=5, shuffle=True, collate_fn=self.collate_batch)
        testDataloader = DataLoader(testData, batch_size=5, shuffle=True, collate_fn=self.collate_batch)
        num_class = len(scientificLabels)
        # print(num_class)
        emsize = 64
        model = TextClassificationModel(vocabSize + 1, emsize, num_class).to(device)
        # Binary Cross Entropy loss
        # standard loss for binary classification
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.99)

        epochs = 45
        rememberEpoch = 0
        rememberAcc = 0
        testAccAll = {}
        training_plot_data = {}
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            training_plot_data[t] = self.train(trainDataloader, model, loss_fn, optimizer)
            print("Test accuracy: ")
            acc = self.test(testDataloader, model, loss_fn)
            print(str(acc))
            if acc > rememberAcc:
                rememberEpoch = t + 1
                rememberAcc = acc
            testAccAll[t] = acc
        #plotAccuracyPerBatch(training_plot_data)
        print("Done!")
        print("--------Recap-----------")
        print("Final Test Accuracy after " + str(epochs) + " epochs" + " is " + str(acc))
        print("Best Total Test Accuracy: " + str(rememberAcc) + " In epoch: " + str(rememberEpoch))

        #plots
        plotAccuracyPerEpoch(testAccAll)
        plotDocumentsPerTopic()
        subplots(testAccAll)

        torch.save(model.state_dict(), modelPath)

    def collate_batch(self, batch):
        labels, docs, offsets = [], [], [0]
        # print(batch)
        for (label, doc) in batch:
            # print("label: ", label)
            # print("doc: ", doc)
            labels.append(label)
            tensor = torch.tensor(doc, dtype=torch.int64)
            offsets.append(tensor.size(0))
            docs.append(tensor)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        # print("labelsize: ", len(labels), "docsize: ", len(docs), "offsetsize: ", len(offsets))
        labels = torch.tensor(labels, dtype=torch.int64)
        docs = torch.cat(docs)
        return labels.to(device), docs.to(device), offsets.to(device)

    def train(self, dataloader, model, loss_fn, optimizer):
        model.train()
        total_acc, total_count = 0, 0
        log_interval = 10
        start_time = time.time()
        acc_per_batches = {}
        for idx, (label, text, offsets) in enumerate(dataloader):
            #print("label: ", label)
            #print("text: ", text)
            #print("offsets: ", offsets)
            optimizer.zero_grad()
            predicted_label = model(text, offsets)
            #print("predicted: ", predicted_label)
            #print("actual: ", label)
            loss = loss_fn(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print('| {:5d}/{:5d} batches '
                      '| accuracy {:8.3f}'.format(idx, len(dataloader),
                                                  total_acc / total_count))
                acc_per_batches[idx] = (total_acc/total_count)
                total_acc, total_count = 0, 0
                start_time = time.time()

        return acc_per_batches

    def test(self, dataloader, model, loss_fn):
        model.eval()
        total_acc, total_count = 0, 0

        with torch.no_grad():
            for idx, (label, text, offsets) in enumerate(dataloader):
                predicted_label = model(text, offsets)
                loss = loss_fn(predicted_label, label)
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
        return total_acc / total_count
