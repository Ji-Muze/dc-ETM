import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.text import ConcordanceIndex
import numpy as np

with open('./dataset/20ng.pkl', 'rb') as f:
    dataset = pickle.load(f)

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

processed_data = []
print(len(dataset))  # 打印数据集的长度
# 获取数据集中的所有键
keys = list(dataset.keys())
print(keys)
# 访问标签列表
labels = dataset['data_2000']
print(labels)
words = dataset['voc2000']
print(len(words))

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

data = dataset['data_2000']  # 假设 data_2000 是一个稀疏矩阵
vocabulary = dataset['voc2000']

co_occurrence_matrix = lil_matrix((len(vocabulary), len(vocabulary)), dtype=int)  # 创建空的稀疏矩阵
probability_matrix = csr_matrix(co_occurrence_matrix, dtype=float)

for doc in data:
    doc_indices = doc.nonzero()[1]  # 找到文档中非零元素的索引
    for i in range(len(doc_indices)):
        for j in range(i+1, len(doc_indices)):
            co_occurrence_matrix[doc_indices[i], doc_indices[j]] += 1
            co_occurrence_matrix[doc_indices[j], doc_indices[i]] += 1

print(co_occurrence_matrix.getnnz())
for i in range(co_occurrence_matrix.shape[0]):
    row_sum = np.sum(co_occurrence_matrix[i].toarray())
    for j in range(co_occurrence_matrix.shape[0]):
        if row_sum > 0:
            probability_matrix[i, j] = co_occurrence_matrix[i, j] / row_sum

co_occurrence_matrix = co_occurrence_matrix.tocsr()  # 将稀疏矩阵转换为压缩稀疏行矩阵

print(co_occurrence_matrix.toarray())
print(probability_matrix.toarray())
#print(dataset[0]['text'])  # 打印第一个文档的文本内容

# for document in dataset:
#     text = document['text']

#     # Tokenize the text
#     tokens = word_tokenize(text)

#     # Remove stop words and punctuation
#     tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]

#     # Lemmatize the tokens
#     tokens = [lemmatizer.lemmatize(token) for token in tokens]

#     processed_text = ' '.join(tokens)
#     processed_data.append(processed_text)

# window_size = 5  # 窗口大小

# def build_coocurrence_matrix(data, window_size):
#     cooc_matrix = {}

#     # 遍历每个文本
#     for text in data:
#         tokens = text.split()

#         # 构建共现矩阵
#         for i, token in enumerate(tokens):
#             context = tokens[max(0, i - window_size):i] + tokens[i+1:i+window_size+1]
#             if token not in cooc_matrix:
#                 cooc_matrix[token] = {}

#             # 统计共现频率
#             for context_token in context:
#                 if context_token not in cooc_matrix[token]:
#                     cooc_matrix[token][context_token] = 0
#                 cooc_matrix[token][context_token] += 1

#     return cooc_matrix

# cooc_matrix = build_coocurrence_matrix(processed_data, window_size)

# print(cooc_matrix)