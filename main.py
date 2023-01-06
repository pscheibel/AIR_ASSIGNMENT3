#imports


def main(prepareData, trainNN):
    all_documents = documentUtil.readInAllDocuments(paper_directory)
    # all_documents = []
    query_parts = documentUtil.extractContentOfSearchQuery(search_query)
    # print(query_parts)
    indexed_data = documentUtil.applySearchIndex(query_parts, all_documents, usedAlgorithm, usedQueryComparison)
    # visualizeIndexedDocumentList(indexed_data)
    if len(indexed_data[0]) != 0:
        radar_chart(indexed_data)
    output = prepare_output(indexed_data)
    return output


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(False, True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
