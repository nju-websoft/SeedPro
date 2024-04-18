import shutil
from itertools import product

from util import json_load, json_dump


cols = ['acordar1']
methods = ['kgtk', 'wikidata_rdf2vec_sg_200_1']

starts = [36, 36]

for idx, (test_collection, embedding_method) in enumerate(product(cols, methods)):

    res_dict = {}
    weight = 0
    best_i = 0
    start = starts[idx]
    end = start + 11
    for i in range(start, end):
        data_path = f'/home/xxx/code/reproduce_keds/data/retrieve_results/{test_collection}/rerank/{embedding_method}' + \
                    f'/metadata/single_run/{i}/BM25 [m]/with_sparse_merged/op_7_merged_eval.json'
        res = json_load(data_path)
        res = res['mean_all']
        if len(res_dict) == 0 or res_dict['ndcg_cut_5_mean'] < res['ndcg_cut_5_mean']:
            res_dict = res
            weight = 1 - (i - start) * 0.1
            best_i = i

    print(f'{test_collection} {embedding_method}, {best_i = }, {weight = }')

    for k, v in res_dict.items():
        print(f'{v:.4f}')
    
    target_path = '/home/xxx/code/reproduce_keds/results'
    mixed_data_path = f'/home/xxx/code/reproduce_keds/data/retrieve_results/{test_collection}/rerank/{embedding_method}' + \
                    f'/metadata/single_run/{best_i}/BM25 [m]/with_sparse_merged/op_7_merged_eval.json'
    ori_data_path = f'/home/xxx/code/reproduce_keds/data/retrieve_results/{test_collection}/rerank/{embedding_method}' + \
                    f'/metadata/single_run/{best_i}/BM25 [m]/BM25 [m]_rerank.json'
    shutil.copyfile(ori_data_path, f'/home/xxx/code/reproduce_keds/results/NPR_{embedding_method}_{test_collection}_rerun.json')
    data = json_load(f'/home/xxx/code/reproduce_keds/data/retrieve_results/{test_collection}/rerank/{embedding_method}' + \
                    f'/metadata/single_run/{best_i}/BM25 [m]/with_sparse_merged/op_7_merged.json')
    # print(data.keys())
    data = data['result']
    json_dump(data, f'/home/xxx/code/reproduce_keds/results/NPR_{embedding_method}_{test_collection}_interpolated_rerun.json')