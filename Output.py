import matplotlib.pyplot as plt
import os


# makes subplot with accuracy per epoch and trainigndata
def subplots(allAcc):
    # data first plot
    epochs = list(allAcc.keys())
    acc = list(allAcc.values())
    max_acc = max(acc)

    # data second plot
    fileCounts = getFileCountsPerTopic()
    # TODO change abbrevations if different topics are used
    shorts = ["CS", "Math", "Bio", "Elec. Eng.", "Phys."]
    # topics = list(fileCounts.keys())
    counts = list(fileCounts.values())
    max_count = max(counts)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 12))
    fig.tight_layout(pad=5.0)
    # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

    colors = ["green" if i != max_acc else "maroon" for i in acc]
    ax1.bar(epochs, acc, color=colors, width=0.4)
    ax1.set(xlabel='Epoch', ylabel='Accuracy')
    ax1.set_title("Test Accuracy per Epoch")

    colors = ["green" if i != max_count else "maroon" for i in counts]
    ax2.bar(shorts, counts, color=colors, width=0.4)
    ax2.set(xlabel='Topic', ylabel='Documents')
    ax2.set_title("Trainingdata")

    savePlot(plt, 'subplots')


# saves plot as png and pdf in output
def savePlot(plt, name):
    path = "./output"
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + '/' + name + '.png')
    plt.savefig(path + '/' + name + '.pdf')


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
    fileCounts["Mathematics"] = 0
    fileCounts["Biology"] = 0
    fileCounts["Electrical Engineering"] = 0
    fileCounts["Phyiscs"] = 0
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
