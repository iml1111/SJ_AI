import time
import sys
import os
import numpy as np
import pandas as pd
import gensim
from gensim.test.utils import datapath
from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore
import pyLDAvis.gensim
import warnings
warnings.filterwarnings('ignore')
# HyperParameter
NUM_TOPICS = 10
PASSES = 20
WORKERS = 4

def learn(col, start = 0, count = None, split_doc = 1):
	corpus = []
	dictionary = corpora.Dictionary()
	idx = 1
	if count is None:
		NUM_DOCS = col.count()
	else:
		NUM_DOCS = count
	print("Model Learning Start...")
	print("Topics(", NUM_TOPICS,"개)")
	print("Docs (", NUM_DOCS - idx, "개)")
	while(idx <= NUM_DOCS):
		print("##",idx,"~",idx + split_doc - 1,"docs")
		df = get_posts_df(col, idx, split_doc)
		tokenized_doc = df['text']
		dictionary.add_documents(tokenized_doc)
		corpus += [dictionary.doc2bow(text) for text in tokenized_doc]
		idx += split_doc
	print("Training...")
	ldamodel = LdaMulticore(
				corpus,
				num_topics = NUM_TOPICS,
				id2word = dictionary,
				passes = PASSES,
				workers = WORKERS
				)
	save_model(ldamodel, dictionary)
	print("Model saved.")

def model_update(ldamodel, dictionary, col, start = 0, count = None, split_doc = 1):
	corpus = []
	idx = 1
	if count is None:
		NUM_DOCS = col.count()
	else:
		NUM_DOCS = count
	print("Model Updating Start...")
	while(idx <= NUM_DOCS):
		print("##",idx,"~",idx + split_doc - 1,"docs")
		df = get_posts_df(col, idx, split_doc)
		tokenized_doc = df['text']
		dictionary.add_documents(tokenized_doc)
		corpus += [dictionary.doc2bow(text) for text in tokenized_doc]
		idx += split_doc
	print("Training...")
	ldamodel.update(corpus)
	save_model(ldamodel, dictionary)
	print("Model saved.")

def get_posts_df(coll, start, count):
	posts = coll.find().skip(start).limit(count)
	df = pd.DataFrame(columns = ["text"])
	for post in posts:
		if len(post['post']) < 20:
			continue
		token = post['token']
		temp = pd.DataFrame({"text":[token]})
		df = df.append(temp, ignore_index = True)
		print("title:",post['title'])
	return df

def save_model(ldamodel, dictionary):
	ldamodel.save(temp_file)
	dictionary.save(os.getcwd() + "\\output\\soojle_lda_dict")
	print("model saved")

def load_model():
	dictionary = corpora.Dictionary.load(os.getcwd() + "\\output\\soojle_lda_dict")
	lda = LdaModel.load(temp_file)
	print("loaded")
	return lda, dictionary

def visualization(ldamodel, corpus, dictionary):
	print("graphing...")
	vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
	print("downloading...")
	pyLDAvis.save_html(vis,"gensim_output.html")
	print("displaying...")
	pyLDAvis.show(vis)

def get_time():
	print("WorkingTime: {} sec".format(round(time.time()-start,3)))

############################################
# 모델 활용 UTIL 함수

def get_topics(ldamodel, dictionary, doc):
	df = pd.DataFrame({'text':[doc]})
	tokenized_doc = df['text'].apply(lambda x: tkn_func(x))
	corpus = [dictionary.doc2bow(text) for text in tokenized_doc]
	for topic_list in ldamodel[corpus]:
		temp = topic_list
		temp = sorted(topic_list, key = lambda x: (x[1]), reverse=True)
		return temp

def is_vaild_words(dict, word_list):
	temp = dict.doc2idx(word_list)
	result = []
	for i in temp:
		if i == -1:
			result += [False]
		else:
			result += [True]
	return result
