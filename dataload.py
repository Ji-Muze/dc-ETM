import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.text import ConcordanceIndex
import numpy as np
import pickle

with open('./dataset/trump.pkl', 'rb') as f:
    dataset = pickle.load(f)

# 获取数据集中的所有键
keys = list(dataset.keys())
print(keys)
# 访问标签列表
# label = dataset['label']
# print('label: ', label)
# train_id = dataset['train_id']
# test_id = dataset['test_id']
# print('train id: ', train_id)
# print('test id: ', test_id)

# data = dataset['data_2000']  # 假设 data_2000 是一个稀疏矩阵
# #print(data.toarray())
voc = dataset['voc2000']
print(voc)

# word_frequencies = np.sum(data.toarray(), axis=0)
# print("每个词的出现次数:", word_frequencies)
# print(len(word_frequencies))
# # 将word_frequencies保存在本地
# with open('./tcdata/word_frequencies.pkl', 'wb') as f:
#     pickle.dump(word_frequencies, f)

# # 读取co_occurrence_matrix和row_sums
# with open('./tcdata/co_occurrence_matrix.pkl', 'rb') as f:
#     co_occurrence_matrix = pickle.load(f)

# with open('./tcdata/row_sums.pkl', 'rb') as f:
#     row_sums = pickle.load(f)

# #co_occurrence_matrix = co_occurrence_matrix.tocsr()

# print(co_occurrence_matrix.shape)
# #print(co_occurrence_matrix[0][1])
# print(co_occurrence_matrix.toarray())
# print(co_occurrence_matrix.toarray()[0][1])
# print(row_sums[1])