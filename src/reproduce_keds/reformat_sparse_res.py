from util import json_load, json_dump

data_path = '/home/xxx/code/reproduce_keds/data/retrieve_results/BDR/candidates/BDR_BM25_top100.txt'
data = {}
with open(data_path, 'r') as f:
    for line in f.readlines():
        line = line.strip().split(' ')
        query_id = str(line[0])
        dataset_id = int(line[2])
        rank = int(line[3])
        score = float(line[4])
        if query_id not in data:
            data[query_id] = []
        if rank > 100: continue
        data[query_id].append([dataset_id, score])

json_dump(data, 'BM25 [m] test_top100_sorted.json')