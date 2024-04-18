from util import json_load, json_dump, merge_two_score
import os

from configs import common_config, sparse_methods
test_collection_name = common_config.test_collection_name

def merge_score_score_list(op, score_list):
    """
    score_op notes:
    1: 2-stage avg: (sum1/len1 + sum2/len2) / 2
    2: harmonic avg: 2 * sum1 * sum2 / (sum1 + sum2)
    3: total avg: sum1 + sum2 / len1 + len2
    4: Geometric avg: sqrt(sum1 * sum2)
    5: max: max(sum1, sum2)
    6: min: min(sum1, sum2)
    """
    n_score = len(score_list)
    assert 0 < n_score <= 3, f'n_score: {n_score}'
    
    if op == 1:
        return sum(score_list) / n_score
    elif op == 2:
        # harmonic avg, candidate score number may more than 2
        if any([score == 0 for score in score_list]):
            return 0
        return n_score / sum([1 / score for score in score_list])
    elif op == 3:
        return sum(score_list)
    elif op == 4:
        if any([score < 0 for score in score_list]):
            return 0
        # multiply then sqrt/third root/...
        mul_score = 1
        for score in score_list:
            mul_score *= score
        return mul_score ** (1 / n_score)
    elif op == 5:
        return max(score_list)
    elif op == 6:
        return min(score_list)
    
def merge_3score_2op(op1, op2, score1, score2, score3):
    """
    score_op notes:
    1: 2-stage avg: (sum1/len1 + sum2/len2) / 2
    2: harmonic avg: 2 * sum1 * sum2 / (sum1 + sum2)
    3: total avg: sum1 + sum2 / len1 + len2
    4: Geometric avg: sqrt(sum1 * sum2)
    5: max: max(sum1, sum2)
    6: min: min(sum1, sum2)
    """
    # first merge score1 and score2 use op1, then merge the result and score3 use op2
    assert 0 < op1 <= 6, f'op1: {op1}'
    assert 0 < op2 <= 6, f'op2: {op2}'

    from retrieve import merge_two_score
    return merge_two_score(merge_two_score(score1, score2, op1), score3, op2)

def normalize_one_res(did_score_list, normalize_way = 'max-min'):
    
    if len(did_score_list) == 0:
        return []
    
    # normalize by max-min
    max_value = max([score for _, score in did_score_list])
    min_value = min([score for _, score in did_score_list])
    if max_value == min_value:
        return [(did, 1) for did, _ in did_score_list]
    else:
        return [(did, (score - min_value) / (max_value - min_value)) for did, score in did_score_list]
    
    # if did_score_list[0][1] < 0:
    #     "negative-style score, e.g. FSDM"
    #     assert all([score <= 0 for _, score in did_score_list])
    #     max_score = max([abs(score) for _, score in did_score_list])
    #     return [(did, 1 + (score / max_score)) for did, score in did_score_list]
    # else:
    #     max_score = max([score for _, score in did_score_list]) if len(did_score_list) > 0 else 0
    #     return [(did, score / max_score) for did, score in did_score_list] if max_score > 0 else did_score_list

def merge2res(dense_res, sparse_res, k, op = 0, w1 = 0.5):
    merged_res = {}
    for qid in sparse_res.keys():
        print(qid)
        
        sparse_res_one = sparse_res[qid][:k]
        normalized_sparse_res_one = normalize_one_res(sparse_res_one)
        sparse_res_one_dict = dict(normalized_sparse_res_one)

        if qid not in dense_res.keys() or dense_res[qid] is None: 
            print('qid not in dense_res.keys() or dense_res[qid] is None', qid)
            dense_res[qid] = sparse_res_one

        # be careful, only select those in sparse res
        dense_res_one = [(did, score) for (did, score) in dense_res[qid] if did in sparse_res_one_dict.keys()] 
        normalized_dense_res_one = normalize_one_res(dense_res_one)        
        dense_res_one_dict = dict(normalized_dense_res_one)
        
        # print(f'qid: {qid}, len(sparse_res_one_dict): {len(sparse_res_one_dict)}, len(dense_res_one_dict): {len(dense_res_one_dict)}')
        # assert dense_res_one_dict.keys() == sparse_res_one_dict.keys(), f'{qid} set_diff1: {dense_res_one_dict.keys() - sparse_res_one_dict.keys()}, set_diff2: {sparse_res_one_dict.keys() - dense_res_one_dict.keys()}'               

        merged_res_one_dict = {}
        
        for did in sparse_res_one_dict.keys():
            merged_res_one_dict[did] = merge_two_score(sparse_res_one_dict[did], dense_res_one_dict.get(did, 0), op, w1)

        merged_res[qid] = list(merged_res_one_dict.items())
        # break
    return merged_res

def merge3res(res1, res2, res3, k, op = 0, op2 = -1):
    merged_res = {}
    for qid in res3.keys():
        print(qid)
        
        res3_one = res3[qid][:k]
        normalized_res3_one = normalize_one_res(res3_one)
        res3_one_dict = dict(normalized_res3_one)

        if qid not in res1.keys() or res1[qid] is None:
            print('qid not in res1.keys() or res1[qid] is None', qid)
            res1[qid] = res3_one

        if qid not in res2.keys() or res2[qid] is None:
            print('qid not in res2.keys() or res2[qid] is None', qid)
            res2[qid] = res3_one

        # be careful, only select those in sparse res
        res1_one = [(did, score) for (did, score) in res1[qid] if did in res3_one_dict.keys()]
        normalized_res1_one = normalize_one_res(res1_one)
        res1_one_dict = dict(normalized_res1_one)

        res2_one = [(did, score) for (did, score) in res2[qid] if did in res3_one_dict.keys()]
        normalized_res2_one = normalize_one_res(res2_one)
        res2_one_dict = dict(normalized_res2_one)               

        assert res1_one_dict.keys() == res2_one_dict.keys() == res3_one_dict.keys(), f'{qid} set_diff1: {res1_one_dict.keys() - res3_one_dict.keys()}, set_diff2: {res3_one_dict.keys() - res1_one_dict.keys()}'

        merged_res_one_dict = {}
        
        for did in res3_one_dict.keys():
            if op2 == -1:
                merged_res_one_dict[did] = merge_score_score_list(op, [res1_one_dict[did], res2_one_dict[did], res3_one_dict[did]])
            else:
                merged_res_one_dict[did] = merge_3score_2op(op, op2, res1_one_dict[did], res2_one_dict[did], res3_one_dict[did])

        merged_res[qid] = list(merged_res_one_dict.items())
        # break
    return merged_res

def get_path_list():
    from configs import common_config as config
    paths_and_op_list = [] # dense_res_path, sparse_res_path, merged_res_path

    out_dir_name = 'with_sparse_merged'

    candidate_merge_op = [1]

    # multi_run_dir_path = '/home/xxx/code/erm/data/retrieve_results/ntcir/rerank/wiki2vec/metadata/multi_run/22/'
    # for i in os.listdir(multi_run_dir_path):
    for i in range(1):
        # one_run_dir_path = multi_run_dir_path + i + '/'
        one_run_dir_path = f'/home/xxx/code/erm/data/retrieve_results/{test_collection_name}/rerank/wiki2vec/metadata/single_run/1125/'
        for method in sparse_methods:

            sparse_res_path = f'/home/xxx/code/erm/data/retrieve_results/{test_collection_name}/candidates/' + f'{method} {config.train_or_test}_top100_sorted.json'

            rerank_dir_path = one_run_dir_path + f'{method}/'

            rerank_res_path = rerank_dir_path + f'{method}_rerank.json'

            merged_res_dir_path = rerank_dir_path + out_dir_name + '/'

            os.makedirs(merged_res_dir_path, exist_ok=True)

            for op in candidate_merge_op:
                merged_res_path = f'{merged_res_dir_path}/op_{op}_merged.json'            
                paths_and_op_list.append((rerank_res_path, sparse_res_path, merged_res_path, op))
    return paths_and_op_list

def gen_path_list_for_irsota():
    # from configs import common_config as config

    # paths_and_op_list = [] # dense_res_path, ir_sota_res_path, merged_res_path
    # ind = 1125 if test_collection_name == 'acordar1' else 1032
    # dense_path = f'/home/xxx/code/erm/data/retrieve_results/{config.test_collection_name}/rerank/wiki2vec/metadata/single_run/{ind}/BM25 [m]/BM25 [m]_rerank.json' # with_sparse_merged/op_1_merged.json'
    # # dense_path = f'/home/xxx/code/erm/data/retrieve_results/{test_collection_name}/candidates/BM25 [m] {config.train_or_test}_top100_sorted.json'
    # # 
    # # f'/home/xxx/code/erm/data/retrieve_results/{test_collection_name}/candidates/BM25 [m] {config.train_or_test}_top100_sorted.json'
    # candidate_ir_sota_dir_path = f'/home/xxx/code/erm/data/retrieve_results/condenser/ft_msmarco/from_cocondenser_marco/{config.test_collection_name}/'
    # # f'/home/xxx/code/erm/data/retrieve_results/bert_reranker/simple_arrange/pt_ft/{common_config.test_collection_name}/'
    # for ir_sota_name in os.listdir(candidate_ir_sota_dir_path):
    #     if not ir_sota_name.endswith('_rerank.json') or ir_sota_name.endswith('merged_rerank.json'): continue
    #     merge_res_name = ir_sota_name.replace('_rerank.json', '_keds_merged.json')
    #     merge_res_path = f'{candidate_ir_sota_dir_path}/{merge_res_name}'
    #     paths_and_op_list.append((dense_path, f'{candidate_ir_sota_dir_path}/{ir_sota_name}', merge_res_path, 1))

    paths_and_op_list = [['/home/xxx/code/erm/data/retrieve_results/ntcir16/rerank/wikidata_rdf2vec_sg_200_1/metadata/single_run/1/BM25 [m]/BM25 [m]_rerank.json',
                         '/home/xxx/code/erm/data/retrieve_results/ntcir16/candidates/BM25 [m] test_top100_sorted.json',
                         '/home/xxx/code/erm/data/retrieve_results/ntcir16/rerank/wikidata_rdf2vec_sg_200_1/metadata/single_run/1/BM25 [m]/with_sparse_merged/op_1_merged.json', 
                         1]]
    return paths_and_op_list


def get_path_and_merge(paths_and_op_list = None, w1 = 0.5):
    
    if paths_and_op_list is None:
        paths_and_op_list = get_path_list()
    # act like single run
    for dense_res_path, sparse_res_path, merged_res_path, op in paths_and_op_list:
        print('dense_res_path', dense_res_path)
        print('sparse_res_path', sparse_res_path)
        print('merged_res_path', merged_res_path)

        dense_res = json_load(dense_res_path)
        if dense_res_path.endswith('merged.json'):
            dense_res = dense_res['result']
        sparse_res = json_load(sparse_res_path) # ['result']
        # map str to int in sparse_res
        sparse_res = {qid: [(int(did), score) for did, score in did_score_list] for qid, did_score_list in sparse_res.items()}
        # map str to int in dense_res
        dense_res = {qid: [(int(did), score) for did, score in did_score_list] for qid, did_score_list in dense_res.items()}


        dump_res = {
            'config': {
                'op': op,
                'dense_res_path': dense_res_path,
                'sparse_res_path': sparse_res_path
            },
            'result': merge2res(dense_res, sparse_res, 10, op, w1 = w1)
        }

        json_dump(dump_res, merged_res_path)

    # eval
    # from eval import eval_merge
    # eval_merge(paths_and_op_list)

def merge_word_entity_res():
    multi_run_suffix1 = '11/2'
    res1_dir_path = f'../data/retrieve_results/{test_collection_name}/rerank/wiki2vec/metadata/multi_run/{multi_run_suffix1}/'

    multi_run_suffix2 = '2/2'
    res2_dir_path = f'../data/retrieve_results/{test_collection_name}/rerank/wiki2vec/metadata/multi_run/{multi_run_suffix2}/'

    for method in sparse_methods:            
        rerank_dir_path = res1_dir_path + f'{method}/'

        sparse_res_path = f'/home/xxx/code/erm/data/retrieve_results/{test_collection_name}/candidates/' + (f'{method} top100_sorted.json' if test_collection_name == 'acordar1' else \
                f'BM25 [m] {common_config.train_or_test}_top100_sorted.json')

        res1_path = rerank_dir_path + f'{method}_rerank.json'
        res1 = json_load(res1_path)
        
        res2_path = res2_dir_path + f'{method}/' + f'{method}_rerank.json'
        res2 = json_load(res2_path)

        merged_res_dir_path = rerank_dir_path + 'with_entity_merged/'

        os.makedirs(merged_res_dir_path, exist_ok=True)

        # nstep = 10
        # candidate_weight = [w / nstep for w in range(0, nstep+1)]
        candidate_weight = [0.5]
        candidate_op = [1, 2, 4, 5, 6]
        for op in candidate_op:
            weight = 0.5
            merged_res_path = f'{merged_res_dir_path}/weight_{weight}_op_{op}_merged.json'            

            print('res1_path', res1_path)
            print('res2_path', res2_path)
            print('merged_res_path', merged_res_path)

            dump_res = {
                'config': {
                    'weight': weight,
                    'res1_path': res1_path,
                    'res2_path': res2_path
                },
                'result': merge2res(res1, res2, 10, weight, op)
            }

            json_dump(dump_res, merged_res_path)

def merge_res_enter(word_res, entity_res, sparse_res, which, k, op1, op2 = -1):
    score3type_dict = {
        'w': word_res,
        'e': entity_res,
        's': sparse_res
    }
    if len(which) == 3:
        return merge3res(score3type_dict[which[0]], score3type_dict[which[1]], score3type_dict[which[2]], k, op1, op2)
    else:
        assert len(which) == 2, f'not supported which: {which}'
        assert op2 == -1, "op2 must be 1 when which has 2 char"
        return merge2res(score3type_dict[which[0]], score3type_dict[which[1]], k, op1)

    # assert op2 == -1, 'not implement op2 yet'
    if len(which) == 3:
        assert op2 == -1, "op2 must be 1 when which has 3 char"
        res = merge3res(word_res, entity_res, sparse_res, k, op1)
    else:
        if which == 'we':
            res = merge2res(word_res, entity_res, k, op1)
        elif which == 'ws':
            res = merge2res(word_res, sparse_res, k, op1)
        else:
            assert which == 'es', f'not supported which: {which}'
            res = merge2res(entity_res, sparse_res, k, op1)
    return res

def merge_word_entity_sparse_res(which, qd_op_type):
    word_res_dir_path = ''
    entity_res_dir_path = ''
    # qd_op_type = 2
    if qd_op_type > 2:
        qd_op_type -= 1 # map to dir name
    if test_collection_name == 'ntcir':
        word_res_dir_path = f'/home/xxx/code/erm/data/retrieve_results/ntcir/rerank/wiki2vec/metadata/multi_run/11/{qd_op_type}/'
        entity_res_dir_path = f'/home/xxx/code/erm/data/retrieve_results/ntcir/rerank/wiki2vec/metadata/multi_run/2/{qd_op_type}/'
    else:
        word_res_dir_path = f'/home/xxx/code/erm/data/retrieve_results/acordar1/rerank/wiki2vec/metadata/multi_run/31/{qd_op_type}/'
        entity_res_dir_path = f'/home/xxx/code/erm/data/retrieve_results/acordar1/rerank/wiki2vec/metadata/multi_run/32/{qd_op_type}/'


    entity_res_dir_path = f'/home/xxx/code/erm/data/retrieve_results/{test_collection_name}/rerank/wiki2vec/metadata/single_run/1031/'
    word_res_dir_path = f'/home/xxx/code/erm/data/retrieve_results/{test_collection_name}/rerank/wiki2vec/metadata/single_run/1030/'

    for method in sparse_methods:            
        word_rerank_dir_path = word_res_dir_path + f'{method}/'

        sparse_res_path = f'/home/xxx/code/erm/data/retrieve_results/{test_collection_name}/candidates/' + f'{method} {common_config.train_or_test}_top100_sorted.json'
        
        sparse_res = json_load(sparse_res_path)

        word_path = word_rerank_dir_path + f'{method}_rerank.json'
        word_res = json_load(word_path)
        
        entity_path = entity_res_dir_path + f'{method}/' + f'{method}_rerank.json'
        entity_res = json_load(entity_path)

        if set(which) == set('wes'):
            merged_res_dir_path = word_rerank_dir_path + '3_score_merged/'
        elif which == 'we' or which == 'ew':
            merged_res_dir_path = word_rerank_dir_path + 'we_score_merged/'
        else:
            print('not implement yet')
            exit(0)

        os.makedirs(merged_res_dir_path, exist_ok=True)

        # nstep = 10
        # candidate_weight = [w / nstep for w in range(0, nstep+1)]
        candidate_weight = [0.5]
        candidate_op1 = [1, 2, 4, 5, 6]
        candidate_op2 = [1, 2, 4, 5, 6]
        import itertools
        for op1, op2 in itertools.product(candidate_op1, candidate_op2):
            merged_res_path = f'{merged_res_dir_path}/{which}_op1_{op1}_op2_{op2}_merged.json'          

            print('word_path', word_path)
            print('entity_path', entity_path)
            print('merged_res_path', merged_res_path)

            dump_res = {
                'config': {
                    'which': which,
                    'op1': op1,
                    'op2': op2,
                    'word_path': word_path,
                    'entity_path': entity_path,
                    'sparse_res_path': sparse_res_path
                },
                'result': merge_res_enter(word_res, entity_res, sparse_res, which, 10, op1, op2)
            }

            json_dump(dump_res, merged_res_path)


def get_qid2split():
    from database import dashboard_cursor
    dashboard_cursor.execute('SELECT query_id, query_text, split FROM `query_5folds`;')
    qid2split = {}
    for qid, query, split in dashboard_cursor.fetchall():
        qid2split[qid] = split
    return qid2split

def multi_run_4_op():
    candidate_score_op = [1,2,4,5,6]
    ind = 0
    for qd_score_op, ent_sim_op, word_sim_op in itertools.product(candidate_score_op, candidate_score_op, [0] + candidate_score_op):   
        ind += 1
        if qd_score_op not in [1, 2, 4]:
            continue
        merge2res_enter(ind)

if __name__ == '__main__':

    import itertools

    path_list = gen_path_list_for_irsota()
    get_path_and_merge(paths_and_op_list = path_list)
    # get_path_and_merge()