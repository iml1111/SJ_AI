import sys
sys.path.insert(0,'/home/iml/')
sys.path.insert(0,'/home/iml/SOOJLE/')
sys.path.insert(0,'/home/iml/SOOJLE_Crawler/src/')
sys.path.insert(0,'/home/iml/SJ_Auth')
sys.path.insert(0,'/home/iml/SJ_AI/src')
sys.path.insert(0,'/home/iml/IML_Tokenizer/src/')
sys.path.insert(0,'../../IML_Tokenizer/src/')
from gensim.models.fasttext import load_facebook_model
from gensim.models import FastText
from gensim.test.utils import datapath
from gensim import utils, matutils
import numpy as np
import os
import platform
from tknizer import *
os_platform = platform.platform()
if os_platform.startswith("Windows"):
	model_path = "./ft_output/soojle_ft_model"
else:
	model_path = "/home/iml/model/ft/soojle_ft_model"
try: default_ft = FastText.load(model_path)
except: default_ft = None
#### HyperParameter
# 벡터 차원 수
VEC_SIZE = 30
# 연관 지을 윈도우 사이즈
WINDOWS = 10
# 최소 등장 횟수로 제한
MIN_COUNT = 30
#모델 에포크
ITERATION = 1000

#############################
# UTIL 함수

# 모델 저장
def model_save(model, path = model_path):
	model.save(path)

# 모델 로드
def model_load(path = model_path):
	return FastText.load(path)

# 해당 단어 or 단어 리스트와 가장 유사한 단어들 추출
def sim_words(words, model = default_ft):
	return model.wv.most_similar(words)

# 두 단어 리스트 사이의 유사도 측정
def doc_sim(doc_A, doc_B, model = default_ft):
	return model.wv.n_similarity(doc_A, doc_B)

#두 벡터 간의 유사도 측정
def vec_sim(vec_A, vec_B, model = default_ft):
	return np.dot(vec_A, vec_B)

#해당 단어 리스트의 벡터값 추출
def get_doc_vector(doc, model = default_ft):
	v = [model.wv[word] for word in doc]
	return matutils.unitvec(np.array(v).mean(axis=0))

# 딕셔너리에 존재하는 단어인지 식별
def is_valid_words(word_list, model = default_ft):
	result = []
	for i in word_list:
		result += [i in model.wv.vocab]
	return result

##################################
# 학습 코드
def make_corpus(col, start = 0, count = None, split_doc = 1000, update = False):
	corpus = []
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
		temp = get_posts_list(col, idx, split_doc, update)
		corpus += temp
		idx += split_doc
		sys.stdout.write("\033[F")
	return corpus

## 특정 수만큼 포스트 가져오기
def get_posts_list(coll, start, count, update = False):
	if update:
		posts = coll.find({{"ft_learn" : 0}}).skip(start).limit(count)
	else:
		posts = coll.find().skip(start).limit(count)
	result = []
	for post in posts:
		temp =  post['token'] + post['tag']
		#temp = post['token'] + soojle_tokenize(post['title'], post['post'], lazy_mode = True)
		if(len(temp) < 3):
			continue
		if update: coll.update_one({'_id':post['_id']}, {"ft_learn":1})	
		result.append(temp)
	return result

def learn(corpus, update = False, model = None, vec_size = VEC_SIZE, windows = WINDOWS, min_count = MIN_COUNT, iteration = ITERATION):
	print("Training...")
	if update:
		model.build_vocab(corpus, update = update)
		model.train(corpus, total_examples = len(corpus), epochs = model.epochs)
	else:
		model = FastText(size = vec_size,
						window = windows,
						min_count = min_count,
						sentences = corpus,
						iter = iteration)
	return model