import errno
import os
import json
import time
import glob
import csv
import nltk
import pickle
import gc
import sys
from pprint import pprint
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from collections import defaultdict

#Path of the post files in .json format
path = 'Dataset/'
count = 0

#Create bad words list
bad_words = []
with open('badwords','r') as bad_words_file:
	for word in bad_words_file:
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
						comment_text = child['data']['body'].encode('ascii', 'ignore').replace('\n', ' ')
						for sym in puncts:
							comment_text = comment_text.replace(sym," ")
						for num in digits:
							comment_text = comment_text.replace(num," ")
						tokens_comment = [word for word in comment_text.lower().split() if word not in stopword_list]
						tokens = tokens + tokens_comment
						mcorpus['comments'].append(comment_text)
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
								for sym in puncts:
									comment_text = comment_text.replace(sym," ")
								for num in digits:
									comment_text = comment_text.replace(num," ")
								tokens_comment = [word for word in comment_text.lower().split() if word not in stopword_list]
								tokens = tokens + tokens_comment
								mcorpus['comments'].append(comment_text)
								for token in tokens_comment:
									mcorpus['tokens'].append(token)
									mcorpus['token_freq'][token] += 1
								i += 1
								out.write(nl+comment_text)
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
	if (count == 1000):
		break
#Funcion que elimina token con repeticiones menores a x y que no esten en bad_words
def dict_tokens_modified(data, x):
    new_dict_tokens = {k: v for k, v in data.iteritems() if (v >= x || k in bad_words)}
    return new_dict_tokens
filtered_dict = dict_tokens_modified(mcorpus['token_freq'], 2)
validated_tokens = list(filtered_dict)
mcorpus['tokens'] = [token for token in mcorpus['tokens'] if token in validated_tokens]
mcorpus['token_freq'] = filtered_dict

pickle_file_path = './corpus/corpus.pkl'
output = open(pickle_file_path, 'wb')
pickle.dump(mcorpus, output)
output.close()
gc.collect()