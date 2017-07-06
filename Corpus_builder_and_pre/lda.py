import os
from gensim import corpora
from gensim import models
from gensim.models import ldamodel

#Load the dictionary and corpus
if (os.path.exists('tmp/cyberbullying_dictionary.dict') and os.path.exists('tmp/cyberbullying_corpus.lda-c')):
    print('Creating dictionary...')
    dictionary = corpora.Dictionary.load('tmp/cyberbullying_dictionary.dict')
    print('Creating corpus...')
    corpus = corpora.BleiCorpus('tmp/cyberbullying_corpus.lda-c')
else:
    print("Create the dictionary and corpus first")
    
#Initialize a model
#print('Creating Tfidf model...')
#%time tfidf = models.TfidfModel(corpus)
#corpus_tfidf = tfidf[corpus]

#Create a LDA Model and save it
lda = ldamodel.LdaModel(corpus, num_topics=2, id2word=dictionary)
lda.save('tmp/cyberbullying_ldaModel.lda')

#Load LDA Model
#lda = models.LdaModel.load('tmp/cyberbullying_ldaModel.lda')

#Print Topics
print("Imprimiendo topicos")
lda.print_topics(2,100)