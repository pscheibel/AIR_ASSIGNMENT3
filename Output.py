import json
import matplotlib.pyplot as plt
import os


# saves plot as png and pdf in output
def savePlot(plt, name):
    path = "./output"
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + '/' + name + '.png')
    plt.savefig(path + '/' + name + '.pdf')


def newlineForTopics(topics):
    x = 0
    for topic in topics:
        if len(topic) > 7:
            topics[x] = topic.replace(" ", "\n")
        x += 1

    return topics


# makes subplot with accuracy per epoch and trainigndata
def subplots(allAcc):
    # data first plot
    epochs = list(allAcc.keys())
    acc = list(allAcc.values())
    max_acc = max(acc)

    # data second plot
    fileCounts = getFileCountsPerTopic()
    topics_unedited = list(fileCounts.keys())
    counts = list(fileCounts.values())
    max_count = max(counts)
    topics = newlineForTopics(topics_unedited)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 12))
    plt.xticks(rotation=60)
    fig.tight_layout(pad=8.0)

    colors = ["green" if i != max_acc else "maroon" for i in acc]
    ax1.bar(epochs, acc, color=colors, width=0.4)
    ax1.set_xlabel('Epoch', fontsize=16)
    ax1.set_ylabel('Accuracy', fontsize=16)
    ax1.set_title("Test Accuracy per Epoch", fontsize=22)

    colors = ["green" if i != max_count else "maroon" for i in counts]
    ax2.bar(topics, counts, color=colors, width=0.4)
    ax2.set_xlabel('Topic', fontsize=16)
    ax2.set_ylabel('Documents', fontsize=16)
    ax2.set_title("Trainingdata", fontsize=22)

    savePlot(plt, 'subplots')


def subplotCorrectPredAndTfIdf():
    if not os.path.exists('correctPredictions.txt'):
        return
    with open('correctPredictions.txt') as data:
        correctPredictions = json.load(data)
    if not os.path.exists('correctTF-IDFs.txt'):
        return
    with open('correctTF-IDFs.txt') as data:
        correctTfIdf = json.load(data)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 12))
    fig.tight_layout(pad=8.0)

    topics = newlineForTopics(list(correctPredictions.keys()))
    ax1.plot(topics, correctPredictions.values(), color='green')
    ax1.set_xlabel('Topic', fontsize=16)
    ax1.set_ylabel('Prediction', fontsize=16)
    ax1.set_title("Prediction per Topic", fontsize=22)
    ax1.fill_between(topics, correctPredictions.values(), 0, color='green', alpha=.1)

    topics = newlineForTopics(list(correctTfIdf.keys()))
    ax2.plot(topics, correctTfIdf.values(), color='green')
    ax2.set_xlabel('Topic', fontsize=16)
    ax2.set_ylabel('TF-IDF', fontsize=16)
    ax2.set_title("Correct TF-IDF per Topic", fontsize=22)
    ax2.fill_between(topics, correctTfIdf.values(), 0, color='green', alpha=.1)

    savePlot(plt, 'subplot_corrPred_tfIdf')


# accuracy for each epoch
def plotAccuracyPerEpoch(allAcc):
    epochs = list(allAcc.keys())
    acc = list(allAcc.values())

    max_acc = max(acc)
    plt.figure(figsize=(10, 5))
    colors = ["green" if i != max_acc else "maroon" for i in acc]
    plt.bar(epochs, acc, color=colors, width=0.4)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy per Epoch")

    savePlot(plt, 'accuracy_per_epoch_barchart')

    return plt.figure


# count files per topic
def getFileCountsPerTopic():
    fileNames = os.listdir("labelledData")
    # TODO change if other topics get added
    fileCounts = {}
    fileCounts["Computer Science"] = 0
    fileCounts["Economics"] = 0
    fileCounts["Biology"] = 0
    fileCounts["Electrical Engineering"] = 0
    fileCounts["Physics"] = 0
    for file in fileNames:
        for topic in fileCounts:
            if topic in file:
                fileCounts[topic] += 1

    return fileCounts


def plotDocumentsPerTopic():
    fileCounts = getFileCountsPerTopic()
    topics = list(fileCounts.keys())
    counts = list(fileCounts.values())
    max_count = max(counts)

    plt.figure(figsize=(10, 5))
    colors = ["green" if i != max_count else "maroon" for i in counts]
    plt.bar(topics, counts, color=colors, width=0.4)
    plt.xlabel("Topic")
    plt.ylabel("Documents")
    plt.title("Trainingdata")

    savePlot(plt, 'docs_per_topic_barchart')

    return plt.figure


# not needed
def plotAccuracyPerBatch(allAcc):
    for epoch in allAcc:
        batches = list(allAcc.get(epoch).keys())
        acc = list(allAcc.get(epoch).values())
        plt.figure(figsize=(10, 5))
        plt.plot(batches, acc)
        plt.xlabel("Batches")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy per Batch in Epoch " + str(epoch))
        path = "./output"
        if not os.path.exists(path):
            os.makedirs(path)
        path = "output/training_accuracy_plots"
        if not os.path.exists(path):
            os.makedirs(path)

        plt.savefig(path + "/accuracy_per_epoch_barchart_" + str(epoch) + ".png")
        plt.savefig(path + "/accuracy_per_epoch_barchart_" + str(epoch) + ".pdf")
        plt.show()


# not needed
def plotRanking():
    if not os.path.exists('perCentRankingDictionaries.txt'):
        return
    if not os.path.exists('/output/ranking'):
        os.makedirs('/output/ranking')
    with open('perCentRankingDictionaries.txt') as data:
        rankingDict = json.load(data)

        for text_file in rankingDict:
            # x = newlineForTopics(rankingDict.get(text_file).keys())
            x = rankingDict.get(text_file).keys()
            print(x)
            # x = newlineForTopics(x_unedited)
            y = rankingDict.get(text_file).values()

            plt.figure(figsize=(10, 5))
            plt.plot(x, y, label=text_file)
            print(text_file)
            plt.title(text_file)
            # plt.show()
            # savePlot(plt, 'ranking/rankingOf' + str(text_file[5:10]))
