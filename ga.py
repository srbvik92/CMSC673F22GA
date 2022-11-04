# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 14:27:47 2022

@author: Saurabh
"""

import conllu
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

with open('en_ewt-ud-test.conllu', encoding='utf-8') as fp:
  data = [sent for sent in conllu.parse_incr(fp)]
  
#print(len(data))

data[0][4]
data[0][5]

temp = []

for sentence in data:
    
    for token in sentence:
        temp = sentence
        break

temp

sentences = []
distinct_words = []
distinct_words_dict = {}
distinct_tags = []
distinct_tags_dict = {}
distinct_arc_labels = []
distinct_arc_labels_dict = {}

for sentence in data:
    
    #print(sentence)
    temp_word = {}
    temp_sentences = []
    
    for tokens in sentence:
        
        if tokens['form'] not in distinct_words:
            distinct_words.append(tokens['form'])
            distinct_words_dict[tokens['form']] = len(distinct_words)
        
        if tokens['upos'] not in distinct_tags:
            distinct_tags.append(tokens['upos'])
            distinct_tags_dict[tokens['upos']] = len(distinct_tags)
            
        if tokens['deprel'] not in distinct_arc_labels:
            distinct_arc_labels.append(tokens['deprel'])
            distinct_arc_labels_dict[tokens['deprel']] = len(distinct_arc_labels)
        
        temp_word['word'] = tokens['form']
        #print(tokens['form'])
        temp_word['tag'] = tokens['upos']
        #print(tokens['xpos'])
        temp_word['arcs'] = tokens['deprel']
        #print(tokens['deprel'])
        temp_sentences.append([tokens['head'], tokens['form'], tokens['upos'], tokens['deprel']])
    
    
    sentences.append(temp_sentences)
    
    
    #break

sentences

print(len(distinct_words))
print(len(distinct_tags))
print(len(distinct_arc_labels))

embedding_words = nn.Embedding(len(distinct_words) + 1, 100)
embedding_tags = nn.Embedding(len(distinct_tags)+1, 10)
embedding_arc_labels = nn.Embedding(len(distinct_arc_labels)+1, 10)

embedded_word_list = []
embedded_tags_list = []
embedded_arc_labels_list = []

for word in distinct_words:
    
    lookup_tensor = torch.tensor([distinct_words_dict[word]], dtype=torch.long)
    #print(lookup_tensor)
    embedded_word_list.append(embedding_words(lookup_tensor))
    
print(embedded_word_list)

for tags in distinct_tags:
    lookup_tensor = torch.tensor([distinct_tags_dict[tags]], dtype=torch.long)
    embedded_tags_list.append(embedding_tags(lookup_tensor))
#print(embedded_tags_list)

for arc_labels in distinct_tags:
    lookup_tensor = torch.tensor([distinct_tags_dict[arc_labels]], dtype=torch.long)
    embedded_arc_labels_list.append(embedding_arc_labels(lookup_tensor))
#print(embedded_arc_labels_list)

