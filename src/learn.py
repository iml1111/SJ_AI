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

def LDALearn(corpus, dictionary, num_topics, passes, iterations):
	print("#----LDA----#")
	ldamodel, cohorence, perflexity =  LDA.learn(corpus = corpus, dictionary = dictionary, num_topics = num_topics)
	print("cohorence:",cohorence)
	print("perflexity:",perflexity)
	#return ldamodel, corpus, dictionary
	LDA.save_model(ldamodel, dictionary)
	LDA.visualization(ldamodel, corpus, dictionary)
	return ldamodel, cohorence, perflexity, corpus, dictionary

# corpus, dictionary = LDA.make_corpus(col = col, split_doc = 1000)

# a,b,c,d,e = LDALearn(corpus, dictionary, 20, 30, 70)

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
# for i in range(10,24,1): LDAtest(i,corpus, dictionary)
	