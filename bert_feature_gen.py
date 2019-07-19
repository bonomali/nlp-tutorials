#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 13:55:10 2019

@author: enzoampil
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:51:42 2019

@author: enzoampil
"""

# Save the source title representations
import pandas as pd
import numpy as np
from bert_serving.client import BertClient
from termcolor import colored

df = pd.read_csv('data/pubmed_10k.csv').head(100)

prefix_q = '##### **Q:** '
topk = 10
source_titles = df.title.values.tolist()

print(source_titles[:5])

bc = BertClient(check_version=False)
doc_vecs = bc.encode(source_titles)
np.save('ls_articles', doc_vecs)

source_titles_fp='data/pubmed_10k.csv'
source_repr_fp='ls_articles.npy'

doc_vecs = np.load(source_repr_fp)
df = pd.read_csv(source_titles_fp)

def match_title(title, doc_vecs, df, topk=5):
    '''
    Input - tag
    Output - list of results sorted based on match score in descending order
    '''
    source_titles = df.title.values
    indices = df.index.values
    # This returns a batch_size x
    title_vec = bc.encode([title])[0]
    # compute normalized dot product as score
    scores = np.sum(title_vec * doc_vecs, axis=1) / np.linalg.norm(doc_vecs, axis=1)
    topk_idx = np.argsort(scores)[::-1][:topk]
    print('top %d questions similar to "%s"' % (topk, colored(title, 'green')))
    for idx in topk_idx:
        print('> %s\t%s' % (colored('%.1f' % scores[idx], 'cyan'), colored(source_titles[idx], 'yellow')))
    print(topk_idx.shape)
    results = [{'score': str(score_), 'title': title_, 'index': str(index_)} \
                    for score_, title_, index_ in zip(scores[topk_idx], source_titles[topk_idx], indices[topk_idx])]
    return results

# Testing for tag "Immune-oncology"
data = match_title('protein', doc_vecs, df)
print(data)
