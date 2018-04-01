from pymongo import MongoClient
import json

class MongoDBClient(object):
    def __init__(self, port=27017):
        self.client = MongoClient(port=27017)
        self.db = self.client.wikimedia

    def insert(self, title, article, urlList, refList):
        page = {
            'title': title,
            'article': article,
            'urls': urlList,
            'refs': refList
        }
        result = self.db.pages.insert_one(page)


    def read(self, title):
        json_text = self.db.pages.find_one({'title': title})
        return json_text['title'], json_text['article'], json_text['urls'], json_text['refs']


    def getAll(self):
        cursor = self.db.pages.find()
        dict_page = dict()
        title_refs = dict()
        title_urls = dict()
        for json_text in cursor:
            if 'article' in json_text:
                dict_page[json_text['title']] = json_text['article']
            if 'refs' in json_text:
                title_refs[json_text['title']] = json_text['refs']
            if 'urls' in json_text:
                title_urls[json_text['title']] = json_text['urls']
        print(len(dict_page))
        print(len(title_refs))
        print(len(title_urls))
        return dict_page, title_refs, title_urls





'''
client = MongoDBClient()
#lists = ['http://1111', 'wwww.www..wwww']
#client.insert("test", "article11111", lists)
title, article, urls, refs = client.read("San Francisco")
print(title)
print(article)
print(urls)
print(refs)

client.getAll()

'''
