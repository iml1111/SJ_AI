import time
import sys
import os
import numpy as np
import pandas as pd
import gensim
import sys
sys.path.insert(0,'/home/iml/SOOJLE/')
sys.path.insert(0,'/home/iml/SOOJLE_Crawler/src/')
sys.path.insert(0,'/home/iml/SJ_Auth')
sys.path.insert(0,'/home/iml/SJ_AI/src')
sys.path.insert(0,'/home/iml/IML_Tokenizer/src/')
sys.path.insert(0,'../../IML_Tokenizer/src/')

from gensim.test.utils import datapath
from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis.gensim
import warnings
from tknizer import get_tk
import platform

warnings.filterwarnings('ignore')
# HyperParameter
#총 토픽 수
NUM_TOPICS = 10
# 많을수록 곱씹어서 봄
PASSES = 20
# 학습을 수행할 병렬 워커 수
WORKERS = 4
#환경에 따라 변경 필요
os_platform = platform.platform()
if os_platform.startswith("Windows"):
	model_path = os.getcwd() + "\\lda_output\\soojle_lda_model"
	dict_path = os.getcwd() + "\\lda_output\\soojle_lda_dict"
else:
	model_path = "/home/iml/model/lda_output/soojle_lda_model"
	dict_path = "/home/iml/model/lda_output/soojle_lda_dict"

def learn(col, start = 0, count = None, split_doc = 1, update = False):
	corpus = []
	dictionary = corpora.Dictionary()
	idx = 1
	if count is None:
		NUM_DOCS = col.count()
	else:
		NUM_DOCS = count
	if update:
		print("Model Updating Start...")
	else:
		print("Model Learning Start...")
		print("Topics(", NUM_TOPICS,"개)")
		print("Docs (", NUM_DOCS - idx, "개)")
	while(idx <= NUM_DOCS):
		print("##",idx,"~",idx + split_doc - 1,"docs")
		df = get_posts_df(col, idx, split_doc, update)
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
	cm = CoherenceModel(model=model, corpus=corpus, coherence='u_mass')
	coherence = cm.get_coherence()
	print("Cpherence",coherence)
	print('\nPerplexity: ', model.log_perplexity(corpus))
	return ldamodel, dictionary, coherence, model.log_perplexity(corpus)
	#perplex가 낮을수록, coherence가 높을수록 좋음
	#https://coredottoday.github.io/2018/09/17/%EB%AA%A8%EB%8D%B8-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0-%ED%8A%9C%EB%8B%9D/

def csv_learn(file):
	corpus = []
	dictionary = corpora.Dictionary()
	print("Model Learning Start...")
	print("CSV File Preprocessing")
	print("Topics(", NUM_TOPICS,"개)")
	df = pd.read_csv(file)
	#CSV 파일 형식에 따라 변경 필요
	df['text'] = df.apply(lambda x: get_tk(df['title'] + df['text']))
	tokenized_doc = df['text']
	dictionary.add_documents(tokenized_doc)
	corpus += [dictionary.doc2bow(text) for text in tokenized_doc]
	print("Training...")
	ldamodel = LdaMulticore(
				corpus,
				num_topics = NUM_TOPICS,
				id2word = dictionary,
				passes = PASSES,
				workers = WORKERS
				)
	cm = CoherenceModel(model=model, corpus=corpus, coherence='u_mass')
	coherence = cm.get_coherence()
	print("Cpherence",coherence)
	print('\nPerplexity: ', model.log_perplexity(corpus))
	return ldamodel, dictionary, coherence, model.log_perplexity(corpus)

def get_posts_df(coll, start, count, update = False):
	if update:
		posts = coll.find({{"learn" : 0}}).skip(start).limit(count)
	else:
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

def save_model(ldamodel, dictionary, model_path = model_path, dict_path = dict_path):
	ldamodel.save(datepath(model_path))
	dictionary.save(dict_path)
	print("model saved")

############################################
# 모델 활용 UTIL 함수
def load_model():
	dictionary = corpora.Dictionary.load(model_path)
	lda = LdaModel.load(datepath(model_path))
	print("loaded")
	return lda, dictionary

def show_topics(ldamodel, num_words = 5):
	topics = ldamodel.print_topics(
		num_words = num_words) # 토픽 단어 제한
	#토픽 및 토픽에 대한 단어의 기여도
	print("모델 로드 테스트")
	for topic in topics:
		print(topic)
	return topics

def get_topics(ldamodel, dictionary, doc, is_string):
	df = pd.DataFrame({'text':[doc]})
	if is_string:
		tokenized_doc = df['text'].apply(lambda x: get_tk(x))
	else:
		tokenized_doc = df
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

def visualization(ldamodel, corpus, dictionary):
	print("graphing...")
	vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
	print("downloading...")
	pyLDAvis.save_html(vis,"gensim_output.html")
	print("displaying...")
	pyLDAvis.show(vis)

def get_time():
	print("WorkingTime: {} sec".format(round(time.time()-start,3)))
