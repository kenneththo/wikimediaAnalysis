from WikiMediaParser import WikiMediaParser
from NegativeSamplingModel import NegativeSamplingModel
from eigenVectorCentralityTest import eigenVectorCentralityTest




def main(result):

    if result.command == 'download':
        parser = WikiMediaParser(None)
        filename = parser.downloadSourceFile(result.date)
        print("Output filename:", filename)
        useMongo = True
        if useMongo:
            parser.loadWikiMediaXML(file=filename)

    if result.command == 'analysis':
        '''
        if result.file is not None:
            parser = WikiMediaParser(result.file)
            dict_page = parser.loadWikiMediaXML()
            df, title_refs= parser.PrepareData(dict_page)
        else:
            parser = WikiMediaParser(None)
            dict_page, title_refs, title_urls = parser.getDataFromMongoDB()
        '''
        parser = WikiMediaParser(None)
        dict_page, title_refs, title_urls = parser.getDataFromMongoDB()
        parser.RunStats(dict_page, title_refs, title_urls, plot=True)

        sample_title_refs_dict = parser.getSamples('California', 2, title_refs)

        centralityTest = eigenVectorCentralityTest()
        print("generating vertice & edges...")
        edgeList = centralityTest.createEdges(sample_title_refs_dict)
        print("creating graph...")
        centralityTest.createGraph(edgeList)
        #centralityTest.plot()
        print("computing centrality...")
        res = centralityTest.ComputeTopKEigenVectorCentrality(centralityTest.graph, 10)
        print("")
        subgraph = centralityTest.getSubGraph('Los Angeles', 2)
        centralityTest.ComputeTopKEigenVectorCentrality(subgraph, 10)
        centralityTest.plot(subgraph)


    if result.command == 'build':
        model = NegativeSamplingModel(result.model, embeddedVectorDimSize=100, num_epochs=1, window_size=5)
        word_target, word_context, labels, vocab_size, dictionary, reversed_dictionary, refList = \
            model.generateInputSet(result.file, sampling='California')
        model.createModel(word_target, word_context, labels, vocab_size)

    elif result.command == 'predict':
        word1 = result.x1.replace(' ','')
        word2 = result.x2.replace(' ', '')
        model = NegativeSamplingModel(result.model, embeddedVectorDimSize=100, num_epochs=3, window_size=5)
        word_target, word_context, labels, vocab_size, dictionary, reversed_dictionary, refList = \
            model.generateInputSet(result.file, sampling='California')
        model.Load()
        topk_df = model.PredictSimilarity(word1, word2, word_context, dictionary, reversed_dictionary)
        model.plotTopK(topk_df, word1, word2)

if __name__ == "__main__":

    import sys
    import argparse

    parser = argparse.ArgumentParser(description="wikimedia analysis")
    parser.add_argument('-c', action='store', dest='command', required=True,
                        help='-download date | -analysis [-f file] | -c build -n modelName [-f file] | -c predict -x1 word1 -x2 word2 [-f file]')
    parser.add_argument('-n', action='store', dest='model',
                        help='model name')
    parser.add_argument('-x1', action='store', dest='x1',
                        help='word1')
    parser.add_argument('-x2', action='store', dest='x2',
                        help='word2')
    parser.add_argument('-date', action='store', dest='date',
                        help='wikimedia dump date')
    parser.add_argument('-f', action='store', dest='file',
                        help='wikimedia filename')
    result = parser.parse_args()

    main(result)
