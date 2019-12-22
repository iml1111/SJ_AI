import time
import sys
import os
import numpy as np
import pandas as pd
import gensim
import sys
sys.path.insert(0,'/home/iml/')
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
#### HyperParameter
WORKERS = 4
NUM_TOPICS = 26
PASSES = 30
EVERY_POST_LIMIT = 35
NAVER_TITLE_LIMIT = 15
TOTAL_POST_LIMIT = 10
ITERATION = 100 
os_platform = platform.platform()
if os_platform.startswith("Windows"):
	model_path = os.getcwd() + "\\lda_output\\soojle_lda_model"
	dict_path = os.getcwd() + "\\lda_output\\soojle_lda_dict"
else:
	model_path = "/home/iml/model/lda/soojle_lda_model"
	dict_path = "/home/iml/model/lda/soojle_lda_dict"
try:
	default_dict = corpora.Dictionary.load(dict_path)
	default_lda = gensim.models.ldamodel.LdaModel.load(datapath(model_path))
except:
	default_dict = None
	default_lda = None
############################################
#UTIL 함수

# 모델 저장하기
def save_model(model, dictionary, model_path = model_path, dict_path = dict_path):
	model.save(datapath(model_path))
	dictionary.save(dict_path)
	print("model saved")

## 모델 불러오기
def load_model(model_path = model_path, dict_path = dict_path):
	dictionary = corpora.Dictionary.load(dict_path)
	lda = gensim.models.ldamodel.LdaModel.load(datapath(model_path))
	print("loaded")
	return lda, dictionary

## 모델의 모든 토픽 정보 출력
def show_topics(model = default_lda, num_words = 5):
	topics = model.print_topics(
		num_topics = -1,
		num_words = num_words) # 토픽 단어 제한
	#토픽 및 토픽에 대한 단어의 기여도
	for topic in topics: print(topic)
	return topics

## 하나의 문서에 대하여 토픽 정보 예측
def get_topics(doc, model = default_lda, dictionary = default_dict):
	df = pd.DataFrame({'text':[doc]})
	if str(type(doc)) == "<class 'list'>": tokenized_doc = df['text']
	else: tokenized_doc = df['text'].apply(lambda x: get_tk(x))
	corpus = [dictionary.doc2bow(text) for text in tokenized_doc]	
	for topic_list in model[corpus]:
		temp = topic_list
		temp = sorted(topic_list, key = lambda x: (x[1]), reverse=True)
		break
	result = np.zeros(NUM_TOPICS)	
	for idx, data in temp: result[idx] += data 
	return result

## 해당 단어리스트가 딕셔너리에 내에 포함된 단어인지 검증
def is_valid_words(word_list, dict = default_dict):
	temp = dict.doc2idx(word_list)
	result = []
	for i in temp:
		if i == -1: result += [False]
		else: result += [True]
	return result

#####################################################################
# 학습코드 

## DB 내의 데이터 모델 코퍼스로 만들기
def make_corpus(col, start = 0, count = None, split_doc = 1000, update = False):
	corpus = []
	dictionary = corpora.Dictionary()
	idx = 1 + start
	if count is None:
		NUM_DOCS = col.count()
	else:
		NUM_DOCS = count
	if update:
		print("Model Updating Start...")
	else:
		print("Model Learning Start...")
		print("Docs (", NUM_DOCS - idx, "개)")
	while(idx <= NUM_DOCS):
		print("##",idx,"~",idx + split_doc - 1,"docs")
		df = get_posts_df(col, idx, split_doc, update)
		tokenized_doc = df['text']
		dictionary.add_documents(tokenized_doc)
		corpus += [dictionary.doc2bow(text) for text in tokenized_doc]
		idx += split_doc
		sys.stdout.write("\033[F")
	print("Total Corpus:", len(corpus))
	return corpus, dictionary

## 코퍼스를 통해 학습 수행
def learn(corpus, dictionary, num_topics = NUM_TOPICS, passes = PASSES, iterations = ITERATION, update = False, ldamodel = None):
	print("Training...")
	if update:
		ldamodel.update(corpus)
	else:
		ldamodel = LdaMulticore(
					corpus,
					num_topics = num_topics,
					id2word = dictionary,
					passes = passes,
					workers = WORKERS,
					iterations = iterations
					)
		cm = CoherenceModel(model=ldamodel, corpus=corpus, coherence='u_mass')
		coherence = cm.get_coherence()
		perplexity = ldamodel.log_perplexity(corpus)
	return ldamodel, coherence, perplexity


## 특정 수만큼 포스트 가져오기
def get_posts_df(coll, start, count, update = False):
	if update:
		posts = coll.find({{"lda_learn" : 0}}).skip(start).limit(count)
	else:
		posts = coll.find().skip(start).limit(count)
	df = pd.DataFrame(columns = ["text"])
	for post in posts:
		# 코퍼스 전처리 조건
		if post['info'].startswith("everytime") and len(post['post']) < EVERY_POST_LIMIT:
			continue
		if post['info'].startswith("navercafe") and len(post['title']) < NAVER_TITLE_LIMIT:
			continue
		if len(post['title'] + post['post']) < TOTAL_POST_LIMIT:
			continue
		if post['info'] in ["everytime_은밀한","main_bidding","everytime_끝말잇기 ", 
							"everytime_퀴어 ","everytime_애니덕후 "]:
			continue
		if post['info'].startswith("everytime") and coll.find({"info":post['info']}).count() < 500:
			continue
		token = post['token'] + post['tag']
		temp = pd.DataFrame({"text":[token]})
		if update: coll.update_one({'_id':post['_id']}, {"lda_learn":1})	
		df = df.append(temp, ignore_index = True)
	return df

## 모델 시각화
def visualization(ldamodel, corpus, dictionary, name = ""):
	print("graphing...")
	vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
	print("downloading...")
	pyLDAvis.save_html(vis, name + "gensim_output.html")
	#print("displaying...")
	#pyLDAvis.show(vis)

def get_time():
	print("WorkingTime: {} sec".format(round(time.time()-start,3)))



