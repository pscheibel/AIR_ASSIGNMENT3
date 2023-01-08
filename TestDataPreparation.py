import io
import os
import string
import urllib.request

import PyPDF2
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup  # pip install beautifulsoup4
from nltk.corpus import stopwords
from textblob import TextBlob
import json


class TestDataPreparation:
    scientificPapersPerCategory = {}
    dataset = []
    testData = []
    trainData = []
    lookupDict = {}

    def executeLoading(self, cachingFiles, scientificLabels):
        self.loadPdfs()
        self.createNounFiles(cachingFiles, scientificLabels)

    def executePreparation(self, scientificLabels):
        self.setClassificationsOfTrainingData(scientificLabels)
        valueDocs = self.createLookUpDict()
        self.initializeTrainDataStructures(valueDocs)
        return self.trainData, self.testData, self.lookupDict

    def get_pdfs(self, my_url, maxLoopDepth, desiredDataPerCategory, links, currentItem):
        if currentItem >= maxLoopDepth or len(links) >= desiredDataPerCategory:
            return links

        more_url = my_url + "#item" + str(currentItem)
        print("process: " + str(more_url))
        html = urllib.request.urlopen(more_url).read()
        soup = BeautifulSoup(html, features="html.parser")
        for link in soup.findAll('a'):
            if "/pdf/" in str(link.get('href')):
                links.append("https://arxiv.org" + str(link.get('href')) + ".pdf")
            if len(links) >= desiredDataPerCategory:
                return links
        return self.get_pdfs(my_url, maxLoopDepth, desiredDataPerCategory, links, currentItem + 1)

    def loadPdfs(self):
        numOfDocs = 300
        MaxUsedPages = 10
        inputComputerScience = self.get_pdfs("https://arxiv.org/list/cs/pastweek?show=" + str(numOfDocs),
                                             MaxUsedPages, numOfDocs, [], 1)
        inputBio = self.get_pdfs("https://arxiv.org/list/q-bio/pastweek?show=" + str(numOfDocs),
                                 MaxUsedPages, numOfDocs, [], 1)
        inputPhysics = self.get_pdfs("https://arxiv.org/list/physics/pastweek?show=" + str(numOfDocs),
                                     MaxUsedPages, numOfDocs, [], 1)
        inputElectricalEngineering = self.get_pdfs("https://arxiv.org/list/eess/pastweek?show=" + str(numOfDocs),
                                                   MaxUsedPages, numOfDocs, [], 1)
        inputMath = self.get_pdfs("https://arxiv.org/list/math/pastweek?show=" + str(numOfDocs), MaxUsedPages,
                                  numOfDocs, [], 1)

        self.scientificPapersPerCategory = {"Biology": inputBio,
                                            "Computer Science": inputComputerScience,
                                            "Electrical Engineering": inputElectricalEngineering,
                                            "Mathematics": inputMath,
                                            "Physics": inputPhysics}
        #print(self.scientificPapersPerCategory)

    def createNounFiles(self, cachingFiles, scientificLabels):
        skippedCount = 0

        print("Start")
        for category in scientificLabels.values():
            catCnt = 0
            for url in self.scientificPapersPerCategory[category]:
                if cachingFiles and os.path.exists(("./labelledData/" + category + str(catCnt) + ".txt")):
                    catCnt = catCnt + 1
                    skippedCount = skippedCount + 1
                    continue
                try:
                    r = requests.get(url)
                    cont = r.content

                    EOF_MARKER = b"%%EOF"  # https://github.com/py-pdf/pypdf/issues/480
                    if not EOF_MARKER in cont:
                        cont = cont + EOF_MARKER
                    else:
                        cont = cont
                        # cont = cont.replace(EOF_MARKER, b'')
                        # cont = cont + EOF_MARKER
                    f = io.BytesIO(cont)

                    text = ""
                    reader = PyPDF2.PdfReader(f)
                    for pageNum in range(0, len(reader.pages)):
                        pageObj = reader.pages[pageNum]
                        text += pageObj.extract_text()

                    text = text.encode('ascii', errors='ignore').decode()
                except:
                    print("Problem while loading and reading file: " + url)
                    continue

                # en_stemmer = SnowballStemmer(language='english')
                en_stopwords = stopwords.words('english')

                # print(en_stopwords)

                new_row_words = []
                text = text.replace('\n', '')
                for mark in string.punctuation:
                    text = text.replace(mark, '')
                for word in str(text).split(' '):
                    if word.lower() not in en_stopwords:
                        new_row_words.append(word)
                text = " ".join(new_row_words)
                # for word in text.split(' '):
                #    text = text.replace(word, en_stemmer.stem(word))
                # print(text[0:100])

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

                # print(nouns[0:100])
                fileName = category + str(catCnt)
                catCnt = catCnt + 1
                isExist = os.path.exists("./labelledData")
                if not isExist:
                    os.mkdir("./labelledData")
                f = open(str("./labelledData/" + fileName + ".txt"), "w")
                mystr = ""
                for tup in nouns:
                    mystr += str(tup) + ";"
                f.write(mystr)
                f.close()
                print("File " + fileName + " created")
            print("The number of already preloaded category files (skip count) is " + str(skippedCount))
            skippedCount = 0
        print("finished!")

    def setClassificationsOfTrainingData(self, scientificLabels):
        pt = "./labelledData/"
        filePathNames = os.listdir(pt)

        tfIdf = {}
        idf = {}
        for category in scientificLabels.values():
            tfIdf[category] = {}

        # print(filePathNames)
        categoryCnt = len(scientificLabels)
        unique_words = set()
        isExist = os.path.exists("./relevance")
        if not isExist:
            os.mkdir("./relevance")
        f = open(str("./relevance/relevance1.csv"), "w")
        f.write("DocumentNumber,CategoryId\n")
        mystr = ""
        fileIdx = 0
        for fileName in filePathNames:
            try:
                fullPath = str(pt + fileName)
                datafromfile = np.loadtxt(fullPath, dtype=str, delimiter=';')
                datafromfile = datafromfile[0:len(datafromfile) - 2]
                associatedWords = []
                for ent in list(datafromfile):
                    myEnt = str(ent).replace('(', '')
                    myEnt = str(myEnt).replace(')', '')
                    myEnt = str(myEnt).replace('\'', '')

                    word = myEnt.split(',')[0]
                    associatedWords.append(word)
                self.dataset.append(associatedWords)
            except:
                fileIdx += 1
                continue

            fileIdx += 1
            categoryId = 0
            myText = ""
            for category in scientificLabels.values():
                if category in fileName:
                    myText += str(fileIdx) + ","
                    myText += str(categoryId)
                    # print(myText)
                    f.write(myText + "\n")
                    for word in associatedWords:
                        if word in tfIdf[category]:
                            tfIdf[category][word] = tfIdf[category][word] + 1
                        else:
                            tfIdf[category][word] = 1
                    break
                categoryId = categoryId + 1
        f.close()
        for category in scientificLabels.values():
            maxVal = max(tfIdf[category].values())
            for word in tfIdf[category]:
                # term frequency per category normalized by max frequency
                tfIdf[category][word] = tfIdf[category][word] / maxVal
                if word not in idf:
                    documentFrequency = 0
                    for otherCategory in scientificLabels.values():
                        if word in tfIdf[otherCategory]:
                            documentFrequency += 1
                    inverseDocFreq = np.log10(categoryCnt / documentFrequency)
                    idf[word] = inverseDocFreq
                tfIdf[category][word] = tfIdf[category][word] * inverseDocFreq
            tfIdf[category] = dict(sorted(tfIdf[category].items(), key=lambda item: item[1], reverse=True))
        # print("finished")
        with open('tfIdf.txt', 'w') as wordfile:
            wordfile.write(json.dumps(tfIdf))
        with open('idf.txt', 'w') as idfFile:
            idfFile.write(json.dumps(idf))

    def createLookUpDict(self):
        dictIdx = 0
        for doc in self.dataset:
            for idx in range(len(doc)):
                if doc[idx] not in self.lookupDict:
                    self.lookupDict[doc[idx]] = dictIdx
                    dictIdx += 1

        valueDocs = np.empty(len(self.dataset), dtype=object)

        valIdx = 0
        for doc in self.dataset:
            valueDocs[valIdx] = np.empty(len(doc), dtype=float)
            for idx in range(len(doc)):
                valueDocs[valIdx][idx] = self.lookupDict[doc[idx]]
            valIdx += 1

        # for idx, (k, v) in enumerate(self.lookupDict.items()):
        #   if idx == 10: break
        #   print(k, v)
        # print("length lookup: ", len(self.lookupDict))
        with open('lookupdict.txt', 'w') as dict_file:
            dict_file.write(json.dumps(self.lookupDict))
        return valueDocs

    def initializeTrainDataStructures(self, valueDocs):
        relevance_data = pd.read_csv("./relevance/relevance1.csv")

        relevanceLength = len(relevance_data.index)

        for idx in range(relevanceLength):
            if idx % 10 == 0:
                self.testData.append((relevance_data.iloc[idx][1], valueDocs[idx]))
            else:
                self.trainData.append((relevance_data.iloc[idx][1], valueDocs[idx]))
