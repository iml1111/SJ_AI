import LDA
import FastText
import sys
from pymongo import MongoClient
from bson.objectid import ObjectId

def refac(id, pw, collection,  host = 'localhost:27017'):
	client = MongoClient('mongodb://%s:%s@%s' %(id, pw, host))
	col = client["soojle"][collection]
	posts = list(col.find())
	for idx,post in enumerate(posts):
		if idx % 100 == 0:
			sys.stdout.write("\033[F") 
			print("##",idx,"docs")

		topic_str = post["tag"] + post["token"]
		col.update_one(
			{"_id":post["_id"]},
			{"$set": {
					"topic":LDA.get_topics(topic_str).tolist(),
					"ft_vector":FastText.get_doc_vector(topic_str).tolist()
				}
			}
		)
