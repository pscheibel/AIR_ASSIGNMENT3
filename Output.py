import matplotlib.pyplot as plt

def plotAccuracyPerEpoch(allAcc):
    epochs = list(allAcc.keys())
    acc = list(allAcc.values())
    max_acc = max(acc)
    plt.figure(figsize=(10, 5))
    colors = ["green" if i != max_acc else "maroon" for i in acc]
    plt.bar(epochs, acc, color=colors, width = 0.4)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy of each Epoch")
    plt.savefig('output/accuracy_per_epoch_barchart.png')
    plt.savefig('output/accuracy_per_epoch_barchart.pdf')

def plotAccuracyPerBatch(allAcc):
    for epoch in allAcc:
        batches = list(allAcc.get(epoch).keys())
        acc = list(allAcc.get(epoch).values())
        plt.figure(figsize=(10, 5))
        plt.plot(batches, acc)
        plt.xlabel("Batches")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy per Batch in Epoch " + str(epoch))
        plt.savefig("output/training_accuracy_plots/accuracy_per_epoch_barchart_" + str(epoch) + ".png")
        plt.savefig("output/training_accuracy_plots/accuracy_per_epoch_barchart_" + str(epoch) + ".pdf")
        plt.show()