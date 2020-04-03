import LDA
import FastText
import sys
from pymongo import MongoClient
from bson.objectid import ObjectId

def refac(id, pw, collection,  host = 'localhost:27017'):
	client = MongoClient('mongodb://%s:%s@%s' %(id, pw, host))
	col = client["soojle"][collection]
	posts = list(col.find())

	print("# 전체 포스트 토픽 리팩토링 개시\n")
	for idx,post in enumerate(posts):
		if idx % 100 == 0:
			sys.stdout.write("\033[F") 
			print("##",idx,"docs")

		topic_str = post['token'][len(post['title_token']):]
		col.update_one(
			{"_id":post["_id"]},
			{"$set": {
					"topic":LDA.get_topics(topic_str).tolist(),
					"ft_vector":FastText.get_doc_vector(topic_str).tolist()
				}
			}
		)
	print("\n\n# 전체 사용자 관심도 리팩토링 개시\n")
	col2 = client["soojle"]['SJ_USER']
	users = list(col2.find())
	for idx,user in enumerate(users):
		sys.stdout.write("\033[F") 
		print("##",idx,"user")
		for i in ['fav_list','view_list']:
			for post in user[i]:
				item = col.find_one({"_id":ObjectId(post["_id"])},
						{
							"topic":1,
							"token":1,
							"tag":1,
						}
					)
				item['post_date'] = item['date']
				del item['date']
				post.update(item)
		col2.update({"user_id":user['user_id']},user)
	client.close()

