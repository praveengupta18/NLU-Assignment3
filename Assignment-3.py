
# coding: utf-8

# In[18]:


import codecs
import numpy as np
import nltk
import pycrfsuite
from bs4 import BeautifulSoup as bs
from bs4.element import Tag
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

docs1 = []
docs2 = []
docs = []
word_list=[]
word_dict={}

with open('ner.txt',errors = 'replace') as fp:
    docs1 = fp.read().splitlines()
for item in docs1:
    t = tuple(item.split())
    docs2.append(t)
doc = []
for t in docs2:
    if t != ():
        doc.append(t)

    else:
        docs.append(doc)
        doc = []

data = []
for i, doc in enumerate(docs):

    tokens = [t for t, label in doc]

    tagged = nltk.pos_tag(tokens)

    data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])


# In[40]:


def word2features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]
    
    c = [w for w in word if w not in ['a','e','i','o','u']]
    without_vowel = "".join(c)
        
    features = [
        'word.lower=' + word.lower(),
        'word_end_with=' + word[-4:],
        'len='+str(len(word)),
#         'witout_vowel='+without_vowel,
        'word_start_with=' + word[:3],
#        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
#        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag
    ]

    if i > 0:
        word1 = doc[i-1][0]
        postag1 = doc[i-1][1]
        flag=1
        if word in word_dict:
            flag=0
        features.extend([
            'word.lower=' + word1.lower(),
            'word.istitle=%s' % word1.istitle(),
            'word.isupper=%s' % word1.isupper(),
            'word.isdigit=%s' % word1.isdigit(),
                 'postag=' + postag1
        ])
    else:
        # First word of a Sentence
        features.append('BOS')
        
#     if i > 1:
#         word1 = doc[i-2][0]
#         postag1 = doc[i-2][1]
#         features.extend([
#             '-1:word.lower=' + word1.lower(),
#             '-1:word.istitle=%s' % word1.istitle(),
#             '-1:word.isupper=%s' % word1.isupper(),
#             '-1:word.isdigit=%s' % word1.isdigit(),
#             '-1:postag=' + postag1
#         ])
#     else:
#         # Second word of a Sentence
#         features.append('SOS')
        
#     if i < len(doc)-2:
#         word1 = doc[i+2][0]
#         postag1 = doc[i+2][1]
#         features.extend([
#             '+1:word.lower=' + word1.lower(),
#             '+1:word.istitle=%s' % word1.istitle(),
#             '+1:word.isupper=%s' % word1.isupper(),
#             '+1:word.isdigit=%s' % word1.isdigit(),
#             '+1:postag=' + postag1
#         ])
#     else:
#         # Second last word of a Sentence
#         features.append('SLS')
        
    if i < len(doc)-1:
        word1 = doc[i+1][0]
        postag1 = doc[i+1][1]
        features.extend([
            'word.lower=' + word1.lower(),
            'word.istitle=%s' % word1.istitle(),
            'word.isupper=%s' % word1.isupper(),
            'word.isdigit=%s' % word1.isdigit(),
            'postag=' + postag1
        ])
    else:
        # End of a Sentence
        features.append('EOS')

    return features

def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]

def get_labels(doc):
    return [label for (token, postag, label) in doc]


X = [extract_features(doc) for doc in data]
y = [get_labels(doc) for doc in data]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=22)

trainer = pycrfsuite.Trainer(verbose=True)

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

trainer.set_params({
    'c1': 0.1,

    'c2': 0.01,  
    
    'max_iterations': 300,

    'feature.possible_transitions': True
})

trainer.train('crf.model')

tagger = pycrfsuite.Tagger()
tagger.open('crf.model')
y_pred = [tagger.tag(xseq) for xseq in X_test]

i = 12

labels = {"T": 2, "O": 1,"D" : 0}

predictions = np.array([labels[tag] for row in y_pred for tag in row])
truths = np.array([labels[tag] for row in y_test for tag in row])
accuracy = accuracy_score(truths, predictions)
print(accuracy)


print(classification_report(
    truths, predictions,
target_names=["T","O","D"]))

