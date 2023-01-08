import os

import PyPDF2
import numpy as np
import json
from textblob import TextBlob

inputDestination = "./nounInput"


def readInput(prepareInput, filePath):
    if prepareInput:
        createNounFiles(filePath)
    return convertToIndexFiles(readNounFiles())


def createNounFiles(filePath):
    if os.path.exists(filePath):
        fileNames = os.listdir(filePath)
        for fileName in fileNames:
            reader = PyPDF2.PdfReader(filePath + fileName)
            text = ""
            for pageNum in range(0, len(reader.pages)):
                pageObj = reader.pages[pageNum]
                text += pageObj.extract_text()

            text = text.encode('ascii', errors='ignore').decode()

            blob = TextBlob(text)

            nouns = []
            # extract nouns and count occurrence
            for noun in blob.noun_phrases:
                if len([tup for tup in nouns if tup[0] == noun]) > 0:
                    idx = [idx for idx, tup in enumerate(nouns) if tup[0] == noun]
                    nouns[idx[0]] = (noun, 1 + nouns[idx[0]][1])
                else:
                    nouns.append((noun, 1))
            nouns.sort(key=lambda x: x[1], reverse=True)

            isExist = os.path.exists(inputDestination)
            if not isExist:
                os.mkdir(inputDestination)
            f = open(str(inputDestination + "/" + fileName[:-4] + ".txt"), "w")
            mystr = ""
            for tup in nouns:
                mystr += str(tup) + ";"
            f.write(mystr)
            f.close()
            print("File " + fileName + " created")


def readNounFiles():
    dataset = {}
    fileNames = os.listdir(inputDestination)
    for fileName in fileNames:
        fullPath = str(inputDestination + "/" + fileName)
        datafromfile = np.loadtxt(fullPath, dtype=str, delimiter=';')
        datafromfile = datafromfile[0:len(datafromfile) - 2]
        associatedWords = []
        for ent in list(datafromfile):
            myEnt = str(ent).replace('(', '')
            myEnt = str(myEnt).replace(')', '')
            myEnt = str(myEnt).replace('\'', '')

            word = myEnt.split(',')[0]
            associatedWords.append(word)
        dataset[fileName] = associatedWords

    return dataset


def convertToIndexFiles(dataset):
    with open('lookupDict.txt') as json_file:
        lookupDict = json.load(json_file)

    valueDocs = np.empty(len(dataset), dtype=object)

    valIdx = 0
    for item in dataset.items():
        valueDocs[valIdx] = (item[0], np.empty(len(item[1]), dtype=float))
        for idx in range(len(item[1])):
            if item[1][idx] not in lookupDict:
                valueDocs[valIdx][1][idx] = len(lookupDict)
            else:
                valueDocs[valIdx][1][idx] = lookupDict[item[1][idx]]
        valIdx += 1

    # for idx, (k, v) in enumerate(self.lookupDict.items()):
    #   if idx == 10: break
    #   print(k, v)
    # print("length lookup: ", len(self.lookupDict))
    return valueDocs, len(lookupDict)
