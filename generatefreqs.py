import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.text import ConcordanceIndex
import numpy as np

data_name = 'agnews'
data_file = './dataset/' + data_name + '.pkl'
save_dir = './freqdata/' + data_name
with open(data_file, 'rb') as f:
    dataset = pickle.load(f)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

processed_data = []
#print(len(dataset))  # 打印数据集的长度
# 获取数据集中的所有键
keys = list(dataset.keys())
#print(keys)
# 访问标签列表
labels = dataset['data_2000']
#print(labels)
words = dataset['voc2000']
#print(len(words))

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, dia_matrix

data = dataset['data_2000']  # 假设 data_2000 是一个稀疏矩阵
vocabulary = dataset['voc2000']

import numpy as np
from scipy.sparse import lil_matrix, triu, coo_matrix

# 假设已经完成数据读取和处理，文档存储在变量data中，词汇存储在变量vocabulary中

# 创建一个空的稀疏矩阵
co_occurrence_matrix = lil_matrix((len(vocabulary), len(vocabulary)), dtype=int)

data = data.tocsr()
print(type(data))
# 遍历每个文档
for doc in data:
    # 获取文档中非零元素索引
    doc_indices = doc.nonzero()[1]
    # 遍历文档中的每对非零元素并增加计数
    for i in range(len(doc_indices)):
        for j in range(i + 1, len(doc_indices)):
            co_occurrence_matrix[doc_indices[i], doc_indices[j]] += 1

# 仅保留上三角部分
co_occurrence_matrix = triu(co_occurrence_matrix)

# 计算每行的和
row_sums = np.sum(co_occurrence_matrix, axis=1)

print(co_occurrence_matrix.shape)
print(co_occurrence_matrix.toarray())
print(row_sums.shape)
print(row_sums)

import pickle

# 保存co_occurrence_matrix和row_sums
with open(save_dir + '/co_occurrence_matrix.pkl', 'wb') as f:
    pickle.dump(co_occurrence_matrix, f)

with open(save_dir + '/row_sums.pkl', 'wb') as f:
    pickle.dump(row_sums, f)

word_frequencies = np.sum(data.toarray(), axis=0)
print("每个词的出现次数:", word_frequencies)
print(len(word_frequencies))
# 将word_frequencies保存在本地
with open(save_dir + '/word_frequencies.pkl', 'wb') as f:
    pickle.dump(word_frequencies, f)



# # 读取co_occurrence_matrix和row_sums
# with open('co_occurrence_matrix.pkl', 'rb') as f:
#     co_occurrence_matrix = pickle.load(f)

# with open('row_sums.pkl', 'rb') as f:
#     row_sums = pickle.load(f)

# # 转换为CSR格式的稀疏矩阵
# co_occurrence_matrix = co_occurrence_matrix.tocsr()

# # 创建一个对角矩阵，每个对角线上的元素为每行的总和
# row_sums_diag = dia_matrix((row_sums, [0]), shape=(len(row_sums), len(row_sums)))

# # 计算每个上三角元素的概率
# probability_matrix = co_occurrence_matrix.multiply(1 / row_sums_diag)

# # 打印共现矩阵和概率矩阵的数组表示
# print(co_occurrence_matrix.toarray())
# print(probability_matrix.shape)