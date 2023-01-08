import matplotlib.pyplot as plt
import os

def plotAccuracyPerEpoch(allAcc):
    #TODO Chrissi, make to 1 plot out of 2 subplots
    #First Subplot is this
    #Second subplot is a count of files in folder labbeled data (Physic 300,Biology 308, ....) with titel "training data"
    epochs = list(allAcc.keys())
    acc = list(allAcc.values())

    max_acc = max(acc)
    plt.figure(figsize=(10, 5))
    colors = ["green" if i != max_acc else "maroon" for i in acc]
    plt.bar(epochs, acc, color=colors, width = 0.4)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy per Epoch")
    path = "./output"
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path+'/accuracy_per_epoch_barchart.png')
    plt.savefig(path+'/accuracy_per_epoch_barchart.pdf')

def plotAccuracyPerBatch(allAcc):
    #TODO not necessary!!!
    for epoch in allAcc:
        batches = list(allAcc.get(epoch).keys())
        acc = list(allAcc.get(epoch).values())
        plt.figure(figsize=(10, 5))
        plt.plot(batches, acc)
        plt.xlabel("Batches")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy per Batch in Epoch " + str(epoch))
        path= "./output"
        if not os.path.exists(path):
            os.makedirs(path)
        path="output/training_accuracy_plots"
        if not os.path.exists(path):
            os.makedirs(path)

        plt.savefig(path+"/accuracy_per_epoch_barchart_" + str(epoch) + ".png")
        plt.savefig(path+"/accuracy_per_epoch_barchart_" + str(epoch) + ".pdf")
        plt.show()