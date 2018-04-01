import collections
import numpy as np
from keras.preprocessing.text import text_to_word_sequence
from WikiMediaParser import WikiMediaParser
from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Dense, Reshape, merge
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class NegativeSamplingModel(object):
    def __init__(self, modelName, embeddedVectorDimSize = 300, num_epochs = 10, window_size = 10):
        self.modelName = modelName
        self.model = None
        self.embeddedVectorDimSize = embeddedVectorDimSize
        self.num_epochs = num_epochs
        self.window_size = window_size

    def build_dataset(self, words, n_words):
        """Process raw inputs into a dataset."""
        count = [['UNK', -1]]
        # use the most common words
        count.extend(collections.Counter(words).most_common(n_words - 1))
        # use the least common words
        #count.extend(reversed(collections.Counter(words).most_common()[-n_words+1:]))
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            data.append(index)
        #count[0][1] = unk_count
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, dictionary, reversed_dictionary


    def createModel(self, word_target, word_context, labels, vocab_size):
        # create some input variables
        input_target = Input((1,))
        input_context = Input((1,))

        embedding = Embedding(vocab_size, self.embeddedVectorDimSize, input_length=1, name='embedding')

        target = embedding(input_target)
        target = Reshape((self.embeddedVectorDimSize, 1))(target)
        context = embedding(input_context)
        context = Reshape((self.embeddedVectorDimSize, 1))(context)
        similarity = merge([target, context], mode='cos', dot_axes=0)
        # now perform the dot product operation to get a similarity measure
        dot_product = merge([target, context], mode='dot', dot_axes=1)
        dot_product = Reshape((1,))(dot_product)
        # add the sigmoid output layer
        output = Dense(1, activation='sigmoid')(dot_product)
        # create the primary training model
        self.model = Model(input=[input_target, input_context], output=output)
        self.model.compile(loss='binary_crossentropy', optimizer='adam')

        self.model.fit([word_target, word_context], labels, epochs=self.num_epochs)

        self.model.save(self.modelName)

    def generateInputSet(self, file, sampling=None):
        '''
        wParser = WikiMediaParser(file)
        dict_page = wParser.loadWikiMediaXML()
        df, title_refs_dict = wParser.PrepareData(dict_page)
        '''
        wParser = WikiMediaParser(None)
        dict_page, title_refs_dict, title_urls = wParser.getDataFromMongoDB()

        if sampling is not None:
            title_refs_dict = wParser.getSamples(sampling, 1, title_refs_dict)


        # generate sequence of words for all the "selected" pages
        unique_words_set = set()
        for ref in title_refs_dict:
            if ref in dict_page:
                unique_words_set.update(text_to_word_sequence(dict_page[ref], lower=True))

        # add the titles to the word list as well
        titleList = [element.lower() for element in title_refs_dict.keys()]
        unique_words_set.update(titleList)
        print("number of unique words:", len(unique_words_set))
        # filter out non alphabet
        finalWords = [x for x in list(unique_words_set) if x.isalpha()]
        print("number of unique alphabet-only words", len(finalWords))
        #print("and here are some samples:", finalWords[:10])

        num_most_common_words = len(finalWords)
        #if sampling is not None:
        #    num_most_common_words = int(num_most_common_words/3)
        data, dictionary, reversed_dictionary = self.build_dataset(list(finalWords), num_most_common_words)
        #print("data:", data[:7])

        vocab_size = len(finalWords)
        sampling_table = sequence.make_sampling_table(vocab_size)
        couples, labels = skipgrams(data, vocab_size, window_size=self.window_size, sampling_table=sampling_table)


        word_target, word_context = zip(*couples)
        word_target = np.array(word_target, dtype="int32")
        word_context = np.array(word_context, dtype="int32")

        #print(couples[:10], labels[:10])

        return word_target, word_context, labels, vocab_size, dictionary, reversed_dictionary, titleList


    def Load(self):
        self.model = load_model(self.modelName)

    def PredictSimilarity(self, word1, word2, word_context, dictionary, reversed_dictionary, topK=10):

        word_context_set = set()
        word_context_set.update(word_context)
        #print("word_context_set=", len(word_context_set), len(word_context))

        word_context_set_arr = np.array(list(word_context_set), dtype="int32")

        word1_context = np.zeros(shape=(word_context_set_arr.shape[0]))
        word1_context.fill(dictionary[word1])
        word2_context = np.zeros(shape=(word_context_set_arr.shape[0]))
        word2_context.fill(dictionary[word2])

        #print("Size of word1:",len(word1_context), word1_context[:10] )
        #print("Size of word2:",len(word2_context), word2_context[:10] )
        #print("word_context_set_arr:", word_context_set_arr[:10])

        indexes = [reversed_dictionary[x] for x in word_context_set_arr]

        predict = self.model.predict([word1_context,word_context_set_arr])
        #print(len(word1_context), len(word_context_set_arr), len(predict))

        df1 = pd.DataFrame(predict, index=indexes, columns=[word1])

        predict = self.model.predict([word2_context,word_context_set_arr])
        #print(len(word2_context), len(word_context_set_arr), len(predict))

        df2 = pd.DataFrame(predict, index=indexes, columns=[word2])

        df_all = pd.concat([df1, df2], axis=1)

        df_all[word1+'rank']=df_all[word1].rank(ascending=0,method='max')
        df_all[word2+'rank']=df_all[word2].rank(ascending=0,method='max')
        #print(df_all.sort_values(by=[word1+'rank'], ascending=[1]).head(10))
        #print(df_all.sort_values(by=[word2+'rank'], ascending=[1]).head(10))
        df_all['rank'] = df_all[word1+'rank'] + df_all[word2+'rank']
        topk_df = df_all.sort_values(by=['rank'], ascending=[1])[:topK]
        topk_df.index.name = 'word'
        print(topk_df)
        #topk_df.to_csv(self.modelName + "_topk.csv")
        #top_match = df_all.sort_values(by=['rank'], ascending=[0]).index.values[:10]
        return topk_df

    def plotTopK(self, topk_df, x, y):
        topk_df[x] = topk_df[x] * 100
        topk_df[y] = topk_df[y] * 100

        ax = sns.lmplot(x,  # Horizontal axis
                        y,  # Vertical axis
                        data=topk_df,  # Data source
                        fit_reg=False,  # Don't fix a regression line
                        size=10,
                        aspect=2)  # size and dimension

        plt.title('Similarity Plot')
        # Set x-axis label
        plt.xlabel(x)
        # Set y-axis label
        plt.ylabel(y)
        topk_df['label'] = topk_df.index.values
        #print(topk_df.head(10))

        def label_point(x, y, val, ax):
            a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
            print(a)
            for i, point in a.iterrows():
                #print(str(point['val']))
                ax.text(point['x'] + .02, point['y'], str(point['val']))

        label_point(topk_df[x], topk_df[y], topk_df['label'], plt.gca())
        plt.show()


