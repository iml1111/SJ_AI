import sys
sys.path.insert(0,'../../')
sys.path.insert(0,'../../IML_tokenizer/src')
from db_info import *
from pymongo import MongoClient
import FastText
import LDA
import os

client = MongoClient('mongodb://%s:%s@%s' % (ID, PW, HOST))
db = client['soojle']
col = db['Dump_posts']

def FTLearn():
	print("#----FastText----#")
	corpus = FastText.make_corpus(col = col, split_doc = 1000)
	model = FastText.learn(corpus)
	FastText.model_save(model)
	return model

def LDALearn(tfidf, model_path, dict_path, vis_name = ""):
	print("#----LDA----#")
	corpus, dictionary = LDA.make_corpus(col = col, split_doc = 1000, tf_idf = tfidf)
	ldamodel, cohorence, perflexity =  LDA.learn(corpus = corpus, dictionary = dictionary)
	print("cohorence:",cohorence)
	print("perflexity:",perflexity)
	
	LDA.save_model(ldamodel, dictionary, model_path = model_path, dict_path = dict_path)
	LDA.visualization(ldamodel, corpus, dictionary, name = vis_name)
	return ldamodel, cohorence, perflexity, corpus, dictionary

# ldamodel, cohorence, perflexity, corpus, dictionary = LDALearn(tfidf = False,model_path = os.getcwd() + "\\lda_output\\soojle_lda_model",dict_path = os.getcwd() + "\\lda_output\\soojle_lda_dict")


# ldamodel2, cohorence2, perflexity2, corpus2, dictionary2 = LDALearn(tfidf = True,model_path = os.getcwd() + "\\tfidf_lda_output\\soojle_lda_model",dict_path = os.getcwd() + "\\tfidf_lda_output\\soojle_lda_dict", vis_name = "tfidf")


# ftmodel = FTLearn()


def LDAtest(num_topic,corpus, dictionary):
	f = open("num_topic_test.txt","a")
	f.write("#-----------------------------#\n")
	f.write("num_topics:" + str(num_topic) + "\n")
	f.close()
	
	f = open("num_topic_test.txt","a")
	ldamodel, cohorence, perflexity =  LDA.learn(
							corpus = corpus, 
							dictionary = dictionary,
							num_topics = num_topic)
	
	f.write("cohorence:" + str(cohorence) + "\n")
	f.write("perflexity:" + str(perflexity) + "\n")
	f.close()
	
# corpus, dictionary = LDA.make_corpus(col = col, split_doc = 1000)
# for i in range(10,31,5): LDAtest(i,corpus, dictionary)
	