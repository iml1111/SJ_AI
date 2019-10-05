import sys
sys.path.insert(0,'/home/iml/')
sys.path.insert(0,'/home/iml/SOOJLE/')
sys.path.insert(0,'/home/iml/SOOJLE_Crawler/src/')
sys.path.insert(0,'/home/iml/SJ_Auth')
sys.path.insert(0,'/home/iml/SJ_AI/src')
sys.path.insert(0,'/home/iml/IML_Toknizer/src/')
sys.path.insert(0,'../../IML_Toknizer/src/')

from gensim.models.fasttext import load_facebook_model
from gensim.models import FastText
from gensim.test.utils import datapath
import os
import platform
#from tknizer import get_tk

#환경에 따라 변경 필요
os_platform = platform.platform()
if os_platform.startswith("Windows"):
	model_path = os.getcwd() + "\\ft_output\\soojle_ft_model"
else:
	model_path = "/home/iml/model/ft_output/soojle_ft_model"

#### HyperParameter
# 학습에 포함될 최소 글자수(에브리타임)
EVERY_POST_LIMIT = 35
#네이버 카페 최소 제목수
NAVER_TITLE_LIMIT = 15
#나머지 최소 글자수 제한
TOTAL_POST_LIMIT = 10
# 벡터 차원 수
VEC_SIZE = 30
# 연관 지을 윈도우 사이즈
WINDOWS = 10
# 최소 등장 횟수로 제한
MIN_COUNT = 100
#모델 에포크
ITERATION = 100

def make_corpus(col, start = 0, count = None, split_doc = 1, update = False):
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
		posts = coll.find({{"learn" : 0}}).skip(start).limit(count)
	else:
		posts = coll.find().skip(start).limit(count)
	result = []
	for post in posts:
		temp =  post['token'] + post['tag']
		if(len(temp) < 3):
			continue
		result.append(temp)
	return result

def learn(corpus, update = False):
	print("Training...")
	if update:
		model = FastText.load(model_path)
		model.build_vocab(corpus, update = update)
		model.train(corpus, total_examples = len(corpus), epochs = model.epochs)
	else:
		model = FastText(size = VEC_SIZE,
						window = WINDOWS,
						min_count = MIN_COUNT,
						sentences = corpus,
						iter = ITERATION)
	return model

###########################
# UTIL 함수

# 모델 저장
def model_save(model, path):
	model.save(path)

# 모델 로드
def model_load(path):
	return FastText.load(path)

# 해당 토큰들과 가장 유사한 단어들 뽑기(10개)
# >>> model.wv.most_similar(["오버워치", "겐지"])
# [('맥크리', 0.5675587058067322), ('정크랫', 0.5659589171409607), ('팟지', 0.5467239618301392), ('오버워
# 치의', 0.5388162136077881), ('POTG', 0.5294703841209412), ('시메트라', 0.5286500453948975),

# 토큰 2개 사이의 유사도 측정
# >>> model.wv.similarity('문재인', '이명박')
# 0.52090985
# >>> model.wv.similarity('문재인', '신희재')
# 0.20399559

# 딕셔너리에 존재하는 단어인지 식별
def is_valid_words(model, word_list):
	result = []
	for i in word_list:
		result += [i in model.wv.vocab]
	return result

