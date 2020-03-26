#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import os
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
import re
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from nltk.tokenize import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from wordcloud import WordCloud
from matplotlib import pyplot as plt
import random
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
import gc


# In[2]:


data = pd.read_csv('genre_filtered_movie_no_entities.csv')
data


# In[3]:


# removing nan items
nan_items = []
for i,d in data.iterrows():
	if type(d['summary_wiki'])!= str:
		nan_items.append(i)
data = data.drop(nan_items)

# removing 0 cast movies
data['cast']= [eval(c) for c in data['cast']]
data['cast_len'] = [len(c) for c in data['cast']]
data = data.drop(data[data['cast_len']==0].index)


# In[4]:


# Removing names
nlp = spacy.load('en_core_web_sm')
stop_words = stopwords.words("english")

#filter out names through nerc
def remove_names(text):
	document = nlp(text)
	ents = [e.text for e in document.ents if e.label_ != 'PERSON']
	return " ".join([item.text for item in document if item.text not in ents])

#data['summary_no_names'] = data['summary_wiki'].apply(lambda x: remove_names(x))
#data.to_csv('genre_filtered_movie_no_entities.csv')


# In[5]:


# sampling
genres = ['Animation', 'Family', 'Fantasy', 'Mystery',
	   'Sci-Fi', 'Thriller', 'Biography', 'Musical', 'War', 'Western',
	   'Adventure', 'Horror', 'Drama', 'Romance', 'Action', 'Crime', 'Comedy',
	   'History']

sample_movie_indexes = []
for g in genres:
	for i,m in data[data[g]==1].sample(2).iterrows():
		sample_movie_indexes.append(i)

		
data_len = len(data)

sample_percentage = 10
sample_movie_indexes = random.sample(range(data_len), int(data_len/sample_percentage))

train_index = [x for x in range(len(data)) if x not in sample_movie_indexes]

test_df = data.iloc[sample_movie_indexes, :].reset_index()
train_df = data.iloc[train_index,:].reset_index()
#test_df.to_csv('test_df.csv')
#train_df.to_csv('train_df.csv')
test_df =  pd.read_csv('test_df.csv')
train_df = pd.read_csv('train_df.csv')
test_df['cast']= [eval(c) for c in test_df['cast']]
train_df['cast']= [eval(c) for c in train_df['cast']]


# In[6]:


for i,d in data.iterrows():
	if len(data['cast']) ==0:
		print('fudeu', i , data['cast'])


# In[7]:


# tokenizer
stemmer = WordNetLemmatizer()
def tokenize(content):
	letters_only = re.sub("[^a-zA-Z]"," ", content)
	lower_case = letters_only.lower()
	tokens = word_tokenize(lower_case)
	words = [w for w in tokens if not w in stop_words]
	stems = [stemmer.lemmatize(word) for word in words]
	return(stems)


# In[8]:


def print_topics(topics, feature_names, sorting, topics_per_chunk=6, n_words=20):
	for i in range(0, len(topics), topics_per_chunk):
		these_topics = topics[i: i + topics_per_chunk]
		len_this_chunk = len(these_topics)
		words = []
		for i in range(n_words):
			try:
				words.append(feature_names[sorting[these_topics, i]])
			except:
				pass

	#setting up word dictionary for comparison
	word_dict = {}
	for i in topics:
		word_dict.update({i : [word[i] for word in words]})
	
	return word_dict


# In[9]:


def topic_report(file, df, topics, feature_names, sorting):
	
	file.write('TOPICS REPORT \n')
	lda_topics = print_topics(topics=range(topics), feature_names=feature_names, sorting=sorting, topics_per_chunk=topics)

	for i in range(topics):
		file.write('---------------- \n')
		file.write('TOPIC : '+str(i)+ '\n')
		file.write('MOST IMPORTANT WORDS: '+str(lda_topics[i])+'\n')
		file.write('MOVIES OF THIS TOPIC: \n')
		for i,m in df[df['clusters'] == i].iterrows():
			file.write(m['movie_id']+"\n")
	


# In[10]:


def topic_word_cloud(lda_model,feature_names, topic):
	
	word_matrix = lda_model.components_[topic]
	word_dict = dict(zip(feature_names, word_matrix))
	wc = WordCloud(width=800, height=400, max_words=200).generate_from_frequencies(word_dict)
	
	return wc


# In[11]:


#def get_list_of_actors_clustering(lda_model, document_topics, word_vectorizer, text):
def get_list_of_actors_clustering(lda_model, document_topics, feature_vector):
	#feature_vector = word_vectorizer.transform([text])
	topic_distribution = lda_model.transform(feature_vector)
	topic = topic_distribution[0].argmax()
	related_documents = []
	clusters = np.argmax(document_topics, axis = 1)
	for i, e in enumerate(clusters):
		if e == topic:
			related_documents.append(i)
	return (topic, related_documents)
	


# In[12]:


#def baseline(document_topics, word_vectorizer, text, k = 10):
def baseline(document_topics,feature_vector, k = 10):
	#feature_vector = word_vectorizer.transform([text]).toarray()[0]
	distances = []
	for document in document_topics.toarray():
		distances.append(euclidean(feature_vector, document))
	
	return np.argsort(np.array(distances))[0:k]


# In[13]:


#def get_list_of_k_nearest_documents(lda_model, document_topics, word_vectorizer, text, k = 10):
def get_list_of_k_nearest_documents(lda_model, document_topics,feature_vector , k = 10):
	#feature_vector = word_vectorizer.transform([text])
	topic_distribution = lda_model.transform(feature_vector)[0]
	distances = []
	for document in document_topics:
		distances.append(euclidean(topic_distribution, document))
	
	return np.argsort(np.array(distances))[0:k]


# In[14]:


cast = list(train_df['cast'])
test_cast = list(test_df['cast'])
def retrieve_cast(cast, document_indexes):
	actors = dict()
	
	for index in document_indexes:
		for d in cast[index]:
			for v in d.values():
				actor = v[0][1:]
				if actor not in actors:
					actors[actor] = 1
				else:
					actors[actor]+=1
	return sorted(actors.items(), key = lambda x:x[1], reverse = True)


# In[15]:


def evaluate_movie_suggestion(pred_actors, label_actors):
	
	# input should be lists of the type [(actor_name, score), ....,]
	r_sizes = [20,50,100]
	precision = []
	recall = [] 
	f1_scores = []
	label_actors_set = set([a[0] for a in label_actors])
	for s in r_sizes:
		pred_actors_set = set([a[0] for a in pred_actors][0:s])
		r = len(pred_actors_set.intersection(label_actors_set))/ len(label_actors)
		p = len(pred_actors_set.intersection(label_actors_set))/ s

		if p+r == 0:
			f1 = 0
		else:
			f1 = 2 *(p*r)/(p+r)
		
		recall.append(r)
		precision.append(p)
		f1_scores.append(f1)


	return (precision,recall, f1_scores)


# In[16]:


def write_evaluation_file(file, precision, recall, f1):
	
	p = np.array(precision).reshape((int(len(precision)/3)), 3)
	r = np.array(recall).reshape((int(len(recall)/3)), 3)
	f = np.array(f1).reshape((int(len(f1)/3)), 3)
	file.write('Evaluation \n')
	for e,top in enumerate(['top 20', 'top 50', 'top 100']):
		p_ = p[:,e]
		r_ = r[:,e]
		f_ = f[:,e]
		file.write(top + '\n')
		file.write('--------------------------- \n')
		file.write('precision mean : '+ str(p_.mean())+ '\n')
		file.write('precision std : '+ str(p_.std())+ '\n')
		file.write('precision max : '+ str(p_.max())+ '\n')
		file.write('precision min : '+ str(p_.min())+ '\n')
		file.write('recall mean : '+ str(r_.mean())+ '\n')
		file.write('recall std : '+ str(r_.std())+ '\n')
		file.write('recall max : '+ str(r_.max())+ '\n')
		file.write('recall min : '+ str(r_.min())+ '\n')
		file.write('f1 mean : '+ str(f_.mean())+ '\n')
		file.write('f1 std : '+ str(f_.std())+ '\n')
		file.write('f1 max : '+ str(f_.max())+ '\n')
		file.write('f1 min : '+ str(f_.min())+ '\n')
		file.write('--------------------------- \n')


# In[17]:


def log_actors(actors):
	return [actor[0]+' : '+ str(actor[1])+'\n' for actor in actors][0:100]


# In[18]:


actors_dict = dict()
actors_list = []
count = 0
for c in data['cast']:
	for l in c:
		#print(l)
		for v in l.values():
			assert(len(v) ==1)
			actor = v[0][1:]
			if actor not in actors_dict.keys():
				actors_dict[actor] = count
				actors_list.append(actor)
				count+=1


# In[19]:


def movie_cast_to_hotform(movie_cast, actor_dictionary):
	
	target = [[0]*len(actor_dictionary) for _ in range(len(movie_cast))]
	for i,m in enumerate(movie_cast):
		for d in movie_cast[i]:
			actor = list(d.values())[0][0][1:]
			target[i][actor_dictionary[actor]] = 1
			
	return np.array(target)


# In[22]:



def train_ml_ranking(vector_representations,target):
	
	assert(len(vector_representations.shape)==2)
	
	ranking_model = Sequential()
	ranking_model.add(Dense(64, activation = 'relu', input_shape = (vector_representations.shape[1],)))
	ranking_model.add(Dense(64, activation = 'relu'))
	ranking_model.add(Dense(64, activation = 'relu'))
	ranking_model.add(Dense(target.shape[1], activation = 'sigmoid'))
	ranking_model.compile(optimizer = optimizers.RMSprop(lr=0.0001) , loss = 'categorical_crossentropy', metrics = ['accuracy'])
	print(ranking_model.summary())
	callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
	ranking_model.fit(vector_representations, target, epochs=100,validation_split = 0.1 , callbacks = callbacks)
	
	
	return ranking_model


# In[24]:


def ranking_cast(ranks, cast_list):
	
	assert(len(ranks.shape)==1)
	return [(actors_list[x], 1) for x in np.flip(ranks)]


# In[ ]:


# Selecting the vectorizer
bow = CountVectorizer(lowercase = True,tokenizer=tokenize, max_features = 5000)
tfidf = TfidfVectorizer(analyzer = 'word',tokenizer = tokenize, lowercase = True, max_features = 5000)

vectorizers = [bow, tfidf]
				
# Selecting number of topics
number_of_topics = list(range(2,100,3))
						
# Selecting the k nearest
k_nearest = [3,5,10]

train_ranking_target = movie_cast_to_hotform(cast, actors_dict)

for v in vectorizers:
	path_string ='./results/'
	#print()
	if type(v)==TfidfVectorizer:
		path_string+='tfidf/'
	else:
		path_string+='bow/'
		
	print('Vectorizing ...')
	feature_matrix = v.fit_transform(train_df['summary_no_names'])
	test_feature_matrix = v.transform(test_df['summary_no_names']).toarray()
	# baseline
	baseline_path = path_string+'baseline.txt'
	print('baseline',baseline_path )
	baseline_file = open(baseline_path, 'w+', encoding = 'utf-8')
	baseline_file.write('BASELINE FILE \n')
	baseline_precision_measure = []
	baseline_recall_measure = []
	baseline_f1_measure = []

	baseline_ranking_precision_measure = []
	baseline_ranking_recall_measure = []
	baseline_ranking_f1_measure = []
	print('training ranking model')
	baseline_ranking_model = train_ml_ranking(feature_matrix, train_ranking_target)
	baseline_ranking_pred = np.argsort(baseline_ranking_model.predict(test_feature_matrix), axis = 1)
	for i,movie in tqdm(test_df.iterrows()):
		baseline_file.write('***************\n')
		baseline_file.write('MOVIE :'+movie['movie_id']+'\n')
		baseline_pred_actors = retrieve_cast(cast,baseline(feature_matrix, test_feature_matrix[i]))
		baseline_label_actors = retrieve_cast(test_cast, [i])
		baseline_ranking_actors = ranking_cast(baseline_ranking_pred[i], actors_list)
		baseline_precision, baseline_recall, baseline_f1 = evaluate_movie_suggestion(baseline_pred_actors, baseline_label_actors)
		baseline_precision_measure +=baseline_precision
		baseline_recall_measure += baseline_recall
		baseline_f1_measure += baseline_f1

		baseline_ranking_precision,baseline_ranking_recall,baseline_ranking_f1  = evaluate_movie_suggestion(baseline_ranking_actors, baseline_label_actors)
		
		baseline_ranking_precision_measure += baseline_ranking_precision
		baseline_ranking_recall_measure += baseline_ranking_recall
		baseline_ranking_f1_measure += baseline_ranking_f1

		b = log_actors(baseline_pred_actors)
		c = log_actors(baseline_ranking_actors)
		baseline_file.write('REGULAR Evaluation: \n')
		baseline_file.write('p20 : '+ str(baseline_precision_measure[-3])+ "\n")
		baseline_file.write('p50 : '+ str(baseline_precision_measure[-2])+ "\n")
		baseline_file.write('p100 : '+ str(baseline_precision_measure[-1])+ "\n")

		baseline_file.write('r20 : '+ str(baseline_recall_measure[-3])+ "\n")
		baseline_file.write('r50 : '+ str(baseline_recall_measure[-2])+ "\n")
		baseline_file.write('r100 : '+ str(baseline_recall_measure[-1])+ "\n")
		
		baseline_file.write('f20 : '+ str(baseline_f1_measure[-3])+ "\n")
		baseline_file.write('f50 : '+ str(baseline_f1_measure[-2])+ "\n")
		baseline_file.write('f100 : '+ str(baseline_f1_measure[-1])+ "\n")
		

		baseline_file.write('RANKING Evaluation: \n')
		baseline_file.write('p20 : '+ str(baseline_ranking_precision_measure[-3])+ "\n")
		baseline_file.write('p50 : '+ str(baseline_ranking_precision_measure[-2])+ "\n")
		baseline_file.write('p100 : '+ str(baseline_ranking_precision_measure[-1])+ "\n")

		baseline_file.write('r20 : '+ str(baseline_ranking_recall_measure[-3])+ "\n")
		baseline_file.write('r50 : '+ str(baseline_ranking_recall_measure[-2])+ "\n")
		baseline_file.write('r100 : '+ str(baseline_ranking_recall_measure[-1])+ "\n")
		
		baseline_file.write('p20 : '+ str(baseline_ranking_f1_measure[-3])+ "\n")
		baseline_file.write('p50 : '+ str(baseline_ranking_f1_measure[-2])+ "\n")
		baseline_file.write('p100 : '+ str(baseline_ranking_f1_measure[-1])+ "\n")
		baseline_file.writelines(b)
		baseline_file.write('++++++++++++++++++++++++++++++++++++++++++++++\n RANKING ACTORS \n')
		baseline_file.writelines(c)
		baseline_file.write('+++++++++++++++++++++++++++++++++++++++++++++++++\n')
		
	baseline_file.close()
	baseline_evaluation_file = open(path_string+'baseline_evaluation.txt', 'w+')
	baseline_evaluation_file.write('REGULAR ')
	write_evaluation_file(baseline_evaluation_file, baseline_precision_measure, baseline_recall_measure, baseline_f1_measure)
	baseline_evaluation_file.write('RANKING ')
	write_evaluation_file(baseline_evaluation_file, baseline_ranking_precision_measure,baseline_ranking_recall_measure , baseline_ranking_f1_measure)
	baseline_evaluation_file.close()
	
	for t in number_of_topics:
		gc.collect()
		experiment_path=path_string+"topic"+str(t)+"/"
		if not os.path.exists(experiment_path):
			os.mkdir(experiment_path)
			
		lda = LatentDirichletAllocation(n_components = t)
		print('lda training ...')
		print('Number of topics :', t)
		document_topics = lda.fit_transform(feature_matrix)
		feature_names = np.array(v.get_feature_names())
		
		clusters =  np.argmax(document_topics, axis=1)
		train_df['clusters'] = clusters
		cluster_counts = [0 for _ in range(t)]
		for c in clusters:
			cluster_counts[c]+=1
		
		plt.figure()
		plt.bar(range(t), cluster_counts)
		plt.savefig(experiment_path+'cluster_distribution.png')
		
		t_report_file = open(experiment_path+'topic_report.txt','w+', encoding = 'utf-8')
		topic_report(t_report_file, train_df, t, np.array(v.get_feature_names()), np.argsort(lda.components_, axis=1)[:, ::-1])
		t_report_file.close()
		
		word_cloud_path = experiment_path+'wordclouds/'
		print('word cloud')
		if not os.path.exists(word_cloud_path):
			os.mkdir(word_cloud_path)
			
		for w in range(t):
			word_cloud = topic_word_cloud(lda, feature_names, w)
			word_cloud.to_file(word_cloud_path+str(w)+'.png')
		
		print('experimenting')
		for k in k_nearest:
			experiment_file_name =experiment_path+'k_'+str(k)+'.txt'
			experiment_file = open(experiment_file_name, 'w+', encoding = 'utf-8')
			experiment_file.write('Experiment \n')
			experiment_file.write("## METADATA ## \n")
			experiment_file.write("Vectorizer :"+ str(v)+'\n')
			experiment_file.write("N_topics :"+ str(t)+'\n')
			experiment_file.write("K_nearest :"+ str(k)+'\n')
			experiment_file.write("##############")
			
			experiment_precision_measure = []
			experiment_recall_measure = []
			experiment_f1_measure = []
			for i,movie in tqdm(test_df.iterrows()):
				experiment_file.write('***********************\n')
				experiment_file.write('MOVIE :'+ movie['movie_id']+'\n')
				experiment_pred_actors = retrieve_cast(cast,get_list_of_k_nearest_documents(lda,document_topics, np.array([test_feature_matrix[i]]), k = k))
				experiment_label_actors = retrieve_cast(test_cast, [i])
				precision, recall, f1 = evaluate_movie_suggestion(experiment_pred_actors, experiment_label_actors)
				
				experiment_precision_measure+=precision
				experiment_recall_measure += recall
				experiment_f1_measure += f1


				experiment_file.write('Evaluation: \n')
				experiment_file.write('p20 : '+ str(experiment_precision_measure[-3])+ "\n")
				experiment_file.write('p50 : '+ str(experiment_precision_measure[-2])+ "\n")
				experiment_file.write('p100 : '+ str(experiment_precision_measure[-1])+ "\n")

				experiment_file.write('r20 : '+ str(experiment_recall_measure[-3])+ "\n")
				experiment_file.write('r50 : '+ str(experiment_recall_measure[-2])+ "\n")
				experiment_file.write('r100 : '+ str(experiment_recall_measure[-1])+ "\n")

				experiment_file.write('f20 : '+ str(experiment_f1_measure[-3])+ "\n")
				experiment_file.write('f50 : '+ str(experiment_f1_measure[-2])+ "\n")
				experiment_file.write('f100 : '+ str(experiment_f1_measure[-1])+ "\n")

				b = log_actors(experiment_pred_actors)
				experiment_file.writelines(b)
			
			experiment_file.close()
			experiment_evaluation_file = open(experiment_path + 'k_'+ str(k)+ '_evaluation.txt', 'w+')
			write_evaluation_file(experiment_evaluation_file, experiment_precision_measure, experiment_recall_measure, experiment_f1_measure)
			experiment_evaluation_file.close()
		
		experiment_file_name = experiment_path+'cluster.txt'
		experiment_file = open(experiment_file_name, 'w+', encoding = 'utf-8')
		experiment_file.write('Experiment cluster \n')
		
		experiment_precision_measure = []
		experiment_recall_measure = []
		experiment_f1_measure = []

		experiment_ranking_precision_measure = []
		experiment_ranking_recall_measure = []
		experiment_ranking_f1_measure = []
		
		experiment_ranking_evaluation_measure = []
		print('training ranking model')
		experiment_ranking_model = train_ml_ranking(document_topics, train_ranking_target)
		test_topics = lda.transform(test_feature_matrix)
		experiment_ranking_pred = np.argsort(experiment_ranking_model.predict(test_topics), axis = 1)
		for i,movie in tqdm(test_df.iterrows()):
			experiment_file.write('***********************\n')
			experiment_file.write('MOVIE :'+ movie['movie_id']+'\n')
			
			cl, doc_index = get_list_of_actors_clustering(lda, document_topics, np.array([test_feature_matrix[i]]))
			experiment_file.write('CLUSTER :' +str(cl)+'\n')
			experiment_pred_actors = retrieve_cast(cast,doc_index)
			experiment_label_actors = retrieve_cast(test_cast, [i])
			
			experiment_ranking_actors = ranking_cast(baseline_ranking_pred[i], actors_list)
			precision, recall, f1 = evaluate_movie_suggestion(experiment_pred_actors, experiment_label_actors)
			
			experiment_precision_measure+=precision
			experiment_recall_measure += recall
			experiment_f1_measure += f1


			precision, recall, f1 = evaluate_movie_suggestion(experiment_ranking_actors, experiment_label_actors)

			experiment_ranking_precision_measure+=precision
			experiment_ranking_recall_measure += recall
			experiment_ranking_f1_measure += f1
			
			b = log_actors(experiment_pred_actors)
			c = log_actors(experiment_ranking_actors)
			experiment_file.write('Evaluation CLUSTERING: \n')
			experiment_file.write('p20 : '+ str(experiment_precision_measure[-3])+ "\n")
			experiment_file.write('p50 : '+ str(experiment_precision_measure[-2])+ "\n")
			experiment_file.write('p100 : '+ str(experiment_precision_measure[-1])+ "\n")

			experiment_file.write('r20 : '+ str(experiment_recall_measure[-3])+ "\n")
			experiment_file.write('r50 : '+ str(experiment_recall_measure[-2])+ "\n")
			experiment_file.write('r100 : '+ str(experiment_recall_measure[-1])+ "\n")

			experiment_file.write('f20 : '+ str(experiment_f1_measure[-3])+ "\n")
			experiment_file.write('f50 : '+ str(experiment_f1_measure[-2])+ "\n")
			experiment_file.write('f100 : '+ str(experiment_f1_measure[-1])+ "\n")
			
			experiment_file.write('Evaluation RANKING: \n')
			experiment_file.write('p20 : '+ str(experiment_ranking_precision_measure[-3])+ "\n")
			experiment_file.write('p50 : '+ str(experiment_ranking_precision_measure[-2])+ "\n")
			experiment_file.write('p100 : '+ str(experiment_ranking_precision_measure[-1])+ "\n")

			experiment_file.write('r20 : '+ str(experiment_ranking_recall_measure[-3])+ "\n")
			experiment_file.write('r50 : '+ str(experiment_ranking_recall_measure[-2])+ "\n")
			experiment_file.write('r100 : '+ str(experiment_ranking_recall_measure[-1])+ "\n")

			experiment_file.write('f20 : '+ str(experiment_ranking_f1_measure[-3])+ "\n")
			experiment_file.write('f50 : '+ str(experiment_ranking_f1_measure[-2])+ "\n")
			experiment_file.write('f100 : '+ str(experiment_ranking_f1_measure[-1])+ "\n")
			
			experiment_file.writelines(b)
			experiment_file.write('+++++++++++++++++++++++++++++++++++++++++\n RANKING ACTORS \n')
			experiment_file.writelines(c)
			experiment_file.write('+++++++++++++++++++++++++++++++++++++++++\n')
		
		experiment_file.close()
		experiment_evaluation_file = open(experiment_path + 'cluster_evaluation.txt', 'w+')
		write_evaluation_file(experiment_evaluation_file, experiment_precision_measure, experiment_recall_measure, experiment_f1_measure)
		experiment_evaluation_file.close()
		
		experiment_ranking_evaluation_file = open(experiment_path + 'ranking_evaluation.txt', 'w+')
		write_evaluation_file(experiment_ranking_evaluation_file, experiment_ranking_precision_measure, experiment_ranking_recall_measure,experiment_ranking_f1_measure)
		experiment_ranking_evaluation_file.close()
		
			

