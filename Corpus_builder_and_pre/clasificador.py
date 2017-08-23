import os
from nltk.classify import PositiveNaiveBayesClassifier

# Clasificador, identifica si la oraciones es bullying o no
path = 'Dataset/'
tipos_de_comentarios = list()
for archive in os.listdir(path):
    tipos_de_comentarios.append(str(archive))
sentences = [[],[]]
count = 0
tipo = 0
try:
    with open('All_comments.txt', 'r') as f:
        for sentence in f:
            count += 1
            if (count >= 87714):
                tipo = 1
            sentences[tipo].append(sentence.replace('\n',''))
    f.close()
except Exception:
    print('Exception')
    pass
def features(list_of_comments):
    words = list_of_comments.lower().split()
    return dict(('contains(%s)' % w, True) for w in words)

positive_featuresets = list(map(features, sentences[0]))
unlabeled_featuresets = list(map(features, sentences[1]))
classifier = PositiveNaiveBayesClassifier.train(positive_featuresets, unlabeled_featuresets)