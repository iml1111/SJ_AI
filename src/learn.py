import sys
sys.path.insert(0,'../../')
sys.path.insert(0,'../../IML_tokenizer/src')
from pymongo import MongoClient
import FastText
import LDA

def connect(id, pw, host):
	client = MongoClient('mongodb://%s:%s@%s' % (id, pw, host))
	db = client['soojle_loc']
	col = db['Dump_posts']
	return client, db, col

def FTLearn(col):
	print("#----FastText----#")
	corpus = FastText.make_corpus(col = col, split_doc = 1000)
	model = FastText.learn(corpus)
	FastText.model_save(model)
	FastText.make_tsv(model)
	return model

def LDALearn(col, num_topics, passes, iterations):
	print("#----LDA----#")
	corpus, dictionary = LDA.make_corpus(col = col, split_doc = 1000)
	ldamodel, cohorence, perflexity =  LDA.learn(corpus = corpus, dictionary = dictionary, num_topics = num_topics)
	print("cohorence:",cohorence)
	print("perflexity:",perflexity)
	#return ldamodel, corpus, dictionary
	LDA.save_model(ldamodel, dictionary)
	LDA.visualization(ldamodel, corpus, dictionary)
	return ldamodel, cohorence, perflexity, corpus, dictionary


# client, db, col = connect(id, pw, host)
# a,b,c,d,e = LDALearn(col, 20, 30, 70)
# ftmodel = FTLearn()

# corpus, dictionary = LDA.make_corpus(col = col, split_doc = 1000)
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
	