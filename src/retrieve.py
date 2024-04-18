import numpy as np
from tqdm import tqdm

data_path = '/home/xxx/code/dataset_profiling/data/embeddings'

def load_npz(file_path):
    data = np.load(file_path)
    matrix = data['matrix']
    index = data['index']
    return matrix, index

def cosine(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def get_top_k(query_vector, matrix, index, k):
    scores = []
    for i in range(len(matrix)):
        score = cosine(query_vector, matrix[i])
        if np.isnan(score):
            scores.append(0)
        else:
            scores.append(score)
    scores = np.array(scores)
    top_k = np.argsort(scores)[::-1][:k]
    return scores[top_k], index[top_k]


if __name__ == '__main__':
    test_collections = ['acordar1', 'ntcir']
    methods = ['FALCON2', 'ReFinED']
    k = 100
    for test_collection in test_collections:
        for method in methods:
            metadata_path = f'{data_path}/{test_collection}_metadata_{method}.npz'
            if test_collection == 'ntcir':
                query_path = f'{data_path}/{test_collection}15_query_{method}.npz'
            else:
                query_path = f'{data_path}/{test_collection}_query_{method}.npz'
    
            m_matrix, m_index = load_npz(metadata_path)
            q_matrix, q_index = load_npz(query_path)
            print('shape of metadata matrix: ', m_matrix.shape)
            print('shape of query matrix: ', q_matrix.shape)
            print('shape of metadata index: ', m_index.shape)
            print('shape of query index: ', q_index.shape)
            res_list = []
            with tqdm(total=len(q_matrix), ncols=100, leave=True) as pbar:
                for(i, q) in enumerate(q_matrix):
                    # print('query index: ', q_index[i])
                    scores, indexes = get_top_k(q, m_matrix, m_index, k)
                    rank = 1
                    for (score, idx) in zip(scores, indexes):
                        res_list.append(f'{q_index[i]} Q0 {idx} {rank} {score} {method}')
                        rank += 1
                    pbar.update(1)
    
            with open(f'/home/xxx/code/dataset_profiling/data/res/{test_collection}_{method}_top{k}.txt', 'w') as f:
                for line in res_list:
                    f.write(line + '\n')
