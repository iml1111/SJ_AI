import sys
sys.path.insert(0,'/home/iml/SOOJLE/')
sys.path.insert(0,'/home/iml/SOOJLE_Crawler/src/')
sys.path.insert(0,'/home/iml/SJ_Auth')
sys.path.insert(0,'/home/iml/SJ_AI/src')
sys.path.insert(0,'/home/iml/IML_Toknizer/src/')
sys.path.insert(0,'../../IML_Toknizer/src/')

from gensim.models.fasttext import load_facebook_model
from gensim.test.utils import datapath
import os
import platform

os_platform = platform.platform()
if os_platform.startswith("Windows"):
	fbmodel_path = os.getcwd() + "\\ft_wiki\\cc.ko.300.bin.gz"
else:
	fbmodel_path  = "/home/iml/model/fb_model/cc.ko.300.bin.gz"

def load_wiki_model(model_path = fbmodel_path):
	print("FastText Wiki Model Loading...")
	cap_path = datapath(fbmodel_path)
	return load_facebook_model(cap_path)


###########################
# UTIL 함수

def is_valid_words(model, word_list):
	result = []
	for i in word_list:
		result += [i in model.wv.vocab]
	return result


# 해당 토큰들과 가장 유사한 단어들 뽑기(10개)
# >>> model.wv.most_similar(["오버워치", "겐지"])
# [('맥크리', 0.5675587058067322), ('정크랫', 0.5659589171409607), ('팟지', 0.5467239618301392), ('오버워
# 치의', 0.5388162136077881), ('POTG', 0.5294703841209412), ('시메트라', 0.5286500453948975),

# 토큰 2개 사이의 유사도 측정
# >>> model.wv.similarity('문재인', '이명박')
# 0.52090985
# >>> model.wv.similarity('문재인', '신희재')
# 0.20399559

model = load_wiki_model()
