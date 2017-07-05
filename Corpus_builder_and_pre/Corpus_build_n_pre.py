import re
import os
import json
from pprint import pprint

from gensim import corpora
from gensim.models import Phrases

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict

import nltk
from nltk import BigramCollocationFinder
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Path of the post files in .json format
path = 'Dataset/'
count = 0

#Create bad words list
bad_words = []
with open('badwords','r') as bad_words_file:
	for word in bad_words_file:
	    word = word.replace('\n','').decode('unicode_escape').encode('ascii','ignore')
	    if word != '':
	       bad_words.append(word)

#Remove keywords in cyberbullying from stopwords list
keywords = ["you", "your", "he", "she", "it"]
stopword_list = [stopword for stopword in stopwords.words('english') if stopword not in keywords]
puncts = ".,:;?!()[]{}~+-\"\'#$%&\/"
digits = "0123456789"

#Create the corpus
mcorpus = dict()
mcorpus['comments'] = list()
mcorpus['tokens'] = list()
mcorpus['token_freq'] = defaultdict(int)
i = 0
for name in os.listdir(path):
	if name.endswith('.json'):
		filename='All_comments.txt'
		fout=open(filename, 'a')
		try:
			with open(path+'/'+name) as f:
				op_json = json.loads(f.read())
				try:
					for child in op_json[1]['data']['children']:
						tokens = []
						#Extrae el comentario
						comment_text = child['data']['body'].encode('ascii', 'ignore').replace('\n', ' ')
						#Elimina los links
						comment_text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', comment_text)
						for sym in puncts:
							comment_text = comment_text.replace(sym," ")
						for num in digits:
							comment_text = comment_text.replace(num," ")
						tokens_comment = [word for word in comment_text.lower().split() if word not in stopword_list]
						tokens = tokens + tokens_comment
						mcorpus['comments'].append(tokens_comment)
						for token in tokens_comment:
							mcorpus['tokens'].append(token)
							mcorpus['token_freq'][token] += 1
						if i == 0:
							nl = ""
						else:
							nl = "\n"
						i += 1
						fout.write(nl+comment_text)
						try:
							for child in child['data']['replies']['data']['children']:
								tokens = []
								comment_text = child['data']['body'].encode('ascii', 'ignore').replace('\n', ' ')
								comment_text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', comment_text)
								for sym in puncts:
									comment_text = comment_text.replace(sym," ")
								for num in digits:
									comment_text = comment_text.replace(num," ")
								tokens_comment = [word for word in comment_text.lower().split() if word not in stopword_list]
								tokens = tokens + tokens_comment
								mcorpus['comments'].append(tokens_comment)
								for token in tokens_comment:
									mcorpus['tokens'].append(token)
									mcorpus['token_freq'][token] += 1
								i += 1
								fout.write(nl+comment_text)
						except Exception:
							pass
				except Exception:
					print('Exception at file:'+str(count)+' has no children')
					pass
			with open(path+'/'+name,"wb") as fout2:
				json.dump(op_json, fout2, indent=4)
			fout2.close()
			count += 1
			pprint(count)
		except Exception:
			print('Exception at file:'+str(count))
			pass
		fout.close()
	if (count == 5093):
		break
#Funcion que elimina token con repeticiones menores a x y que no esten en bad_words
def dict_tokens_modified(data, x):
    new_dict_tokens = {k: v for k, v in data.iteritems() if ((v >= x) or (k in bad_words))}
    return new_dict_tokens
filtered_dict = dict_tokens_modified(mcorpus['token_freq'], 2)
validated_tokens = list(filtered_dict)
mcorpus['tokens'] = [token for token in mcorpus['tokens'] if token in validated_tokens]
mcorpus['token_freq'] = filtered_dict

# Lemmatize all words in comments.
lemmatizer = WordNetLemmatizer()
comments = [[lemmatizer.lemmatize(token) for token in doc] for doc in mcorpus['comments']]

# Compute bigrams.
# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
bigram = Phrases(comments, min_count=20)
for idx in range(len(comments)):
    for token in bigram[comments[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            comments[idx].append(token)

#Create and save dictionary
dictionary = corpora.Dictionary(comments)
dictionary.save('tmp/cyberbullying_dictionary.dict') #Save the dictionary

#Increasing weight of bad words:
bad_words_ids = [word_id for word_id, word in dictionary.iteritems() if word in bad_words]

#Convert documents to vectors
corpus = [dictionary.doc2bow(text) for text in comments]
for doc_idx in range(len(corpus)):
    word_id_list = []
    freq_list = []

    for word_id, freq in corpus[doc_idx]:
        word_id_list.append(word_id)
        if word_id in bad_words_ids:
            freq_list.append(freq*2000)
        else:
            freq_list.append(freq)
    
    from random import randint
    if randint(0, 9) > 8:
        word_id_list.append(bad_words_ids[0])
        freq_list.append(5000)
    
    new_doc = zip(word_id_list,freq_list)
    corpus[doc_idx] = new_doc
    
    #print("Most frequent word of Doc_"+str(doc_idx))
    #print(dictionary[word_id_list[freq_list.index(max(freq_list))]])
    #print("----------------------------\n")

bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(mcorpus['tokens'])
finder.apply_freq_filter(3)
bigrams = finder.nbest(bigram_measures.pmi, 50)

#La mayoria son gente famosa, ya que se usan como mofa en el mayor de los casos
print('\nTOP 50 Collocations comentarios negativos\n')
i = 0
for bigram in bigrams:
	i += 1
	print str(i)+'. '+str(bigram)

print('\n')

corpora.BleiCorpus.serialize('tmp/cyberbullying_corpus.lda-c', corpus) #Save the corpus


#Pendiente
# #Entrena el sentiment_analyzer para ser usado en el corpus de resenias
# print("Entrenando sentiment_analyzer...")
# n_instances = 1000
# subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
# obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]
# train_subj_docs = subj_docs[:750]
# test_subj_docs = subj_docs[750:1000]
# train_obj_docs = obj_docs[:750]
# test_obj_docs = obj_docs[750:1000]
# sentiment_analyzer = SentimentAnalyzer()
# training_docs = train_subj_docs+train_obj_docs
# test_docs = test_subj_docs+test_obj_docs
# training_set = sentiment_analyzer.apply_features(training_docs)
# test_set = sentiment_analyzer.apply_features(test_docs)
# trainer = NaiveBayesClassifier.train
# classifier = sentiment_analyzer.train(trainer, training_set)
# 
# #Se evalua la polaridad de las resenias cargadas en mcorpus['comments']
# print("Evaluando...")
# stop = stopwords.words('english')
# polarity = list()
# sid = SentimentIntensityAnalyzer()
# for sent in mcorpus['comments']:
#         lower = [token.lower() for token in sent]
#         sentence_filtered = [token for token in lower if token not in stop]
#         ss = sid.polarity_scores(" ".join(sentence_filtered))
#         polarity.append(ss)
#         #Descomentar para tener resultados por pantalla
#         for k in sorted(ss):
#                 print(str(sent)+' =:=  '+'{0} : {1}'.format(k, ss[k]))
# print("Terminado")