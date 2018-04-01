import re
import urllib.request  # python3.5+
#import urllib
import bz2
from xml.etree import ElementTree
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from MongoDBClient import MongoDBClient
import numpy as np



class WikiMediaParser(object):
    def __init__(self, file):
        self.file = file
        self.client = MongoDBClient()

    def filterText(self, text):
       print("filtering ", text)
       if text is None:
         return False
       bad_chars = '(){}<>/\\'
       for c in bad_chars: text = text.replace(c, "")
       text = text.replace(" ", "")
       print("new text=", text)
       check = text.isalpha()
       print("is alpha=", check)
       if check == False:
          return False
       else:
          return True

    def extractLink(self, article):
        # pages has reference link and url
        #regex = re.compile(".*http=(.*) .*")
        #urlList = regex.findall(article)
        urlList = self.find_between(article, "https://", " ")
        urlList += self.find_between(article, "http://", " ")
        refList = self.find_between(article, '[[', ']]')
        return urlList, refList


    def isRedirect(self, article):
        if article.startswith("#REDIRECT"):
            return True
        else:
            return False

    def find_between(self, s, first, last):
        res = []
        try:
            stop = False
            i = 0
            while not stop:
                s = s[i:]
                start = s.index( first ) + len( first )
                end = s.index( last, start )
                res.append(s[start:end])
                i = end
            return res
        except ValueError:
            return res

    def processBlock(self, text):
        titles= self.find_between(text, "<title>", "</title>")
        articles= self.find_between(text, "<text", "</text>")
        if len(titles) > 0 and len(articles) > 0:
            return titles[0], articles[0]
        return None, None

    def loadWikiMediaXML(self, file=None, mongo=True):
        if file is not None:
            self.file = file
        print("Loading ", self.file)
        dict_page = {}
        pageBlock = ""
        i=0
        for line in open(self.file):
            pagebegin = False
            if "<page>" in line:
                pageBlock = ""
            if "</page>" in line:
                title, article= self.processBlock(pageBlock)
                if title is None or article is None or self.isRedirect(article): continue
                dict_page[title] = article
                urlList, refList = self.extractLink(article)
                i += 1
                #if i >= max:
                #    break
                if i%500 == 0: print("processed ", i, "pages")
                if mongo:
                    self.client.insert(title, article, urlList, refList)
            pageBlock = pageBlock + line

        return dict_page

    # https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search
    def DepthLimitedDFS(self, node, depth, adjList, nodeSet):
        if depth == 0:
            return nodeSet
        if (depth > 0) and (node in adjList):
            for child in adjList[node]:
                if child is None: continue
                nodeSet.update([child])
                nodeSet = self.DepthLimitedDFS(child, depth - 1, adjList, nodeSet)

        return nodeSet

    def getSamples(self, root_title, depth, title_refs_dict):
        if root_title not in title_refs_dict:
            print("hmmm, thats very wierd")
            return None

        sample_titles = set()
        sample_titles = self.DepthLimitedDFS(root_title, depth, title_refs_dict, sample_titles)

        print("after sample :", list(sample_titles)[:10],len(sample_titles))

        sample_title_refs_dict = dict((k, title_refs_dict[k]) for k in sample_titles if k in title_refs_dict)

        return sample_title_refs_dict


    def downloadSourceFile(self, create_date="20180320"):
        url = "https://dumps.wikimedia.org/enwikivoyage/"+ create_date + "/"
        filename = "enwikivoyage-"+ create_date +"-pages-articles-multistream.xml.bz2"
        print("Downloading ", url + filename + " .....")
        filename, _ = urllib.request.urlretrieve(url + filename, filename)  # python 3.5+
        #filename, _ = urllib.urlretrieve(url + filename, filename)
        print("Download completed, extracting ", filename + " .....")

        zipfile = bz2.BZ2File(filename)  # open the file
        data = zipfile.read()  # get the decompressed data
        newfilepath = "./" + filename[:-4]  # assuming the filepath ends with .bz2
        open(newfilepath, 'wb').write(data)  # write a uncompressed file
        print(newfilepath, " created")
        return newfilepath

    '''
    def PrepareData(self, dict_page, plot=True, mongo=True):
        title_refs_dict = {}
        df = pd.DataFrame(columns=['title', 'size_article', 'num_link'])
        i = 0
        #max = 20000
        for title in dict_page.keys():
            if title is not None:
                article = dict_page[title]
                urlList, refList = self.extractLink(article)
                df.loc[i] = [title, len(article), len(urlList) + len(refList)]
                title_refs_dict[title] = refList
                i += 1
                #if i >= max:
                #    break
                if i%500 == 0: print("processed ", i, "articles")
                if mongo:
                    self.client.insert(title, article, urlList, refList)
        return df, title_refs_dict
    '''
    def getDataFromMongoDB(self):
        dict_page, title_refs, title_urls = self.client.getAll()
        return dict_page, title_refs, title_urls



    def RunStats(self, dict_page, title_refs, title_urls, plot=True):
        print("Building dataframe....")
        df = pd.DataFrame(columns=['title', 'size_article', 'num_link', 'num_ref'])
        i=0
        arr_article = np.empty(len(title_refs.keys()))
        arr_UrlLink = np.empty(len(title_refs.keys()))
        arr_ReferenceLink = np.empty(len(title_refs.keys()))
        titlelist = []
        for title in dict_page.keys():
            refs = title_refs[title]
            urlList = title_urls[title]
            article = dict_page[title]
            refs = title_refs[title]
            arr_article[i] = len(article)
            arr_UrlLink[i] = len(urlList)
            arr_ReferenceLink[i] = len(refs)
            titlelist.append(title)
            i += 1

        df['title'] = titlelist
        df['size_article'] = arr_article
        df['num_link'] = arr_UrlLink
        df['num_ref'] = arr_ReferenceLink
        df['total_link'] = df['num_link'] + df['num_ref']

        # run a stats for total links (url + ref)
        print("Maximum:", df.iloc[df['total_link'].argmax()])
        print("Minimum:", df.iloc[df['total_link'].argmin()])
        print("Average # of Links:", df['total_link'].mean())
        print("Median # of Links:", df['total_link'].median())

        from statsmodels.formula.api import ols
        model = ols("size_article ~ total_link", df).fit()
        print(model.summary())

        if plot:
            sns.regplot(x='total_link', y='size_article', data=df)
            plt.show()

        # run a stats for url only
        print("Maximum:", df.iloc[df['num_link'].argmax()])
        print("Minimum:", df.iloc[df['num_link'].argmin()])
        print("Average # of Links:", df['num_link'].mean())
        print("Median # of Links:", df['num_link'].median())

        from statsmodels.formula.api import ols
        model = ols("size_article ~ num_link", df).fit()
        print(model.summary())

        if plot:
            sns.regplot(x='num_link', y='size_article', data=df)
            plt.show()


