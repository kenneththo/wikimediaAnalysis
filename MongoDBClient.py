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





client = MongoDBClient()
lists = ['http://1111', 'wwww.www..wwww']
#client.insert("test", "article11111", lists)
client.read("test")