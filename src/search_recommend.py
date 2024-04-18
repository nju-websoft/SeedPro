import pytrec_eval
from typing import Union, Literal, Dict
import json
from scipy import stats
from ranx import Run, fuse


def read_ontological_run_dict(path):
    run = {}
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f:
            query_id, dataset_id, score = line.split('\t')
            score = float(score)
            if query_id not in run:
                run[query_id] = {}
            run[query_id][dataset_id] = score
    return run


def read_distributed_run_dict(path):
    run_dict = {}
    with open(path, 'r', encoding='UTF-8') as f:
        data = json.load(f)
        for query, scores in data.items():
            run_dict[query] = {}
            for score in scores:
                run_dict[query][str(score[0])] = score[1]
    return run_dict


def read_bm25_run_dict(test_collection: Union[Literal['ntcir15'], Literal['ntcir16'], Literal['acordar1']],
                       top: int = 10):
    path = f"../data/bm25/{test_collection}.txt"
    run = {}
    if test_collection == 'acordar1':
        with open(path) as run_file:
            for line in run_file:
                list = line.split('\t')
                query_id = list[0]
                dataset_id = list[2]
                score = float(list[4])
                if query_id not in run:
                    run[query_id] = {}
                run[query_id][dataset_id] = score
    else:  # ntcir15, ntcir16
        with open(f"../data/bm25/ntcir_hs2ds.json", 'r', encoding='UTF-8') as f:
            hash_2_ds = json.load(f)
        with open(path) as run_file:
            for line in run_file:
                list = line.split(' ')
                query_id = list[0]
                hash_id = list[2]
                dataset_id = str(hash_2_ds[hash_id])
                score = float(list[4])
                if query_id not in run:
                    run[query_id] = {}
                run[query_id][dataset_id] = score
    new_run = {}
    for query_id, scores in run.items():
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top]
        new_run[query_id] = dict(sorted_items)
    return new_run


def get_qrels(test_collection: Union[Literal['ntcir15'], Literal['ntcir16'], Literal['acordar1']]):
    qrels_dict = {}
    with open(f"../data/qrels/{test_collection}.txt", 'r', encoding='UTF-8') as f:
        for line in f:
            query_id, dataset_id, score = line.strip().split('\t')
            query_id = str(query_id)
            dataset_id = str(dataset_id)
            score = int(score)
            if query_id not in qrels_dict:
                qrels_dict[query_id] = {}
            qrels_dict[query_id][dataset_id] = score
        return qrels_dict


def ttest_significance(score_dict_1, score_dict_2):
    scoreAll1, scoreAll2 = [], []
    for k, v in score_dict_1.items():
        scoreAll1.append(v)
        scoreAll2.append(score_dict_2[k])
    t, p = stats.ttest_rel(scoreAll1, scoreAll2)
    return t, p


def ranx_norm(run_dict: Dict[str, Dict[str, float]], norm_name: Union[None, str]) -> Dict[str, Dict[str, float]]:
    if norm_name is None:
        return run_dict
    run = Run(run_dict)
    combined_run = fuse(
        runs=[run, run],
        norm=norm_name,
        method="max",
    )
    return combined_run.to_dict()


def weights_sum(weights, runs):
    new_run_dict = {}
    for qid, scores in runs[0].items():
        new_run_dict[qid] = {}
        for did in scores.keys():
            new_score = 0
            for i in range(len(weights)):
                new_score += runs[i].get(qid, {}).get(did, 0) * weights[i]
            new_run_dict[qid][did] = new_score
    return new_run_dict


def eval_run_measures(run, qrels, baseline_run):
    measures = ['ndcg_cut_5', 'ndcg_cut_10', 'map_cut_5', 'map_cut_10']
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    run_results = evaluator.evaluate(run)
    score_dict = {x: sum([v[x] for v in run_results.values()]) / len(qrels) for x in measures}

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    baseline_results = evaluator.evaluate(baseline_run)
    new_score_dict = {}
    for metrics in measures:
        score_baseline = {k: baseline_results.get(k, {metrics: 0})[metrics] for k in qrels.keys()}
        score_run = {k: run_results.get(k, {metrics: 0})[metrics] for k in qrels.keys()}
        t, p = ttest_significance(score_baseline, score_run)
        new_score_dict[metrics] = {'score': score_dict[metrics], 'p': p}
    return new_score_dict


def save_run_dict(run: dict, path: str):
    with open(path, 'w', encoding='UTF-8') as f:
        json.dump(run, f)


def bi_interpolate(test_collection: Union[Literal['ntcir15'], Literal['ntcir16'], Literal['acordar1']],
                   profiling: Union[Literal['ont'], Literal['kgtk'], Literal['rdf2vec']],
                   metrics: str = 'ndcg_cut_5',
                   save_best=False):
    bm25_run = read_bm25_run_dict(test_collection)
    if profiling == 'ont':
        profiling_run = read_ontological_run_dict(f'../data/ontological/{test_collection}.txt')
    else:  # kgtk or rdf2vec
        profiling_run = read_distributed_run_dict(f'../data/distributed/{test_collection}_{profiling}.json')

    qrels = get_qrels(test_collection)

    runs = [bm25_run, profiling_run]
    runs = [ranx_norm(x, 'min-max') for x in runs]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, [metrics])
    bm25_results = evaluator.evaluate(runs[0])

    combinations = [(i / 10, j / 10) for i in range(11) for j in range(11) if i / 10 + j / 10 == 1]
    interpolate_results = {}
    for weights in combinations:
        new_run_dict = weights_sum(weights, runs)
        run_results = evaluator.evaluate(new_run_dict)
        metrics_mean_score = sum([v[metrics] for v in run_results.values()]) / len(qrels)

        score_bm25 = {k: bm25_results.get(k, {metrics: 0})[metrics] for k in qrels.keys()}
        score_profiling = {k: run_results.get(k, {metrics: 0})[metrics] for k in qrels.keys()}
        t, p = ttest_significance(score_bm25, score_profiling)

        interpolate_results[weights] = {metrics: metrics_mean_score, 'p_value': p}

    best_weights, best_score = None, 0
    for weights, result in interpolate_results.items():
        if result[metrics] > best_score:
            best_score = result[metrics]
            best_weights = weights
    print(f'test collection: {test_collection}, profiling: {profiling}, weights: {best_weights} | '
          f'{metrics}: {interpolate_results[best_weights][metrics]:.4f} | '
          f'p: {interpolate_results[best_weights]["p_value"]:.4f}')

    best_run = weights_sum(best_weights, runs)
    if save_best:
        save_run_dict(best_run, f'../data/results/{test_collection}_{profiling}.json')
    best_measure_scores = eval_run_measures(best_run, qrels, runs[0])
    print('[measures] ', end='')
    for measure, result in best_measure_scores.items():
        print(f'{measure}: {result["score"]:.4f} (p={result["p"]:.4f})', end=' | ')
    print()


def tri_interpolate(test_collection: Union[Literal['ntcir15'], Literal['ntcir16'], Literal['acordar1']],
                    emb: Union[Literal['kgtk'], Literal['rdf2vec']],
                    metrics: str = 'ndcg_cut_5',
                    save_best=False):
    bm25_run = read_bm25_run_dict(test_collection)
    ontological_run = read_ontological_run_dict(f'../data/ontological/{test_collection}.txt')
    distributed_run = read_distributed_run_dict(f'../data/distributed/{test_collection}_{emb}.json')

    qrels = get_qrels(test_collection)

    runs = [bm25_run, ontological_run, distributed_run]
    runs = [ranx_norm(x, 'min-max') for x in runs]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, [metrics])
    bm25_results = evaluator.evaluate(runs[0])

    combinations = [(i / 10, j / 10, k / 10) for i in range(11) for j in range(11) for k in range(11) if
                    i / 10 + j / 10 + k / 10 == 1]
    interpolate_results = {}
    for weights in combinations:
        new_run_dict = weights_sum(weights, runs)
        run_results = evaluator.evaluate(new_run_dict)
        metrics_mean_score = sum([v[metrics] for v in run_results.values()]) / len(qrels)

        score_bm25 = {k: bm25_results.get(k, {metrics: 0})[metrics] for k in qrels.keys()}
        score_profiling = {k: run_results.get(k, {metrics: 0})[metrics] for k in qrels.keys()}
        t, p = ttest_significance(score_bm25, score_profiling)

        interpolate_results[weights] = {metrics: metrics_mean_score, 'p_value': p}

    best_weights, best_score = None, 0
    for weights, result in interpolate_results.items():
        if result[metrics] > best_score:
            best_score = result[metrics]
            best_weights = weights
    print(f'test collection: {test_collection}, embedding: {emb}, weights: {best_weights} | '
          f'{metrics}: {interpolate_results[best_weights][metrics]:.4f} | '
          f'p: {interpolate_results[best_weights]["p_value"]:.4f}')

    best_run = weights_sum(best_weights, runs)
    if save_best:
        save_run_dict(best_run, f'../data/results/{test_collection}_ont_{emb}.json')
    best_measure_scores = eval_run_measures(best_run, qrels, runs[0])
    print('[measures] ', end='')
    for measure, result in best_measure_scores.items():
        print(f'{measure}: {result["score"]:.4f} (p={result["p"]:.4f})', end=' | ')
    print()

#
# def output_emnlp():
#     test_collection = 'ntcir15'
#     qrels = get_qrels(test_collection)
#
#     bm25_run = read_bm25_run_dict(test_collection)
#     emb = 'kgtk'
#     if emb == 'kgtk':
#         distributed_run = read_distributed_run_dict(f'acordar1/EMNLP/results/NPR_kgtk_{test_collection}.json')
#     else:
#         distributed_run = read_distributed_run_dict(
#             f'acordar1/EMNLP/results/NPR_wikidata_rdf2vec_sg_200_1_{test_collection}.json')
#
#     runs = [bm25_run, distributed_run]
#     runs = [ranx_norm(x, 'min-max') for x in runs]
#     new_run = weights_sum((0.5, 0.5), runs)
#
#     measure_scores = eval_run_measures(new_run, qrels, bm25_run)
#     print('[measures] ', end='')
#     for measure, result in measure_scores.items():
#         print(f'{measure}: {result["score"]:.4f} (p={result["p"]:.4f})', end=' | ')
#     print()


def read_BDR_success_eval_dict():
    with open('../data/qrels/BDR.json', 'r', encoding='UTF-8') as f:
        data = json.load(f)
    return data


def read_BDR_bm25_run_dict():
    path = '../data/bm25/BDR.json'
    run_dict = {}
    with open(path, 'r', encoding='UTF-8') as f:
        data = json.load(f)
        for query, scores in data.items():
            run_dict[query] = {}
            for score in scores:
                run_dict[query][str(score[0])] = score[1]
    return run_dict


def get_top_run_dict(run_dict: dict, top: int) -> dict:
    new_run_dict = {}
    for query_id, dataset_dict in run_dict.items():
        sorted_score = sorted(dataset_dict.items(), key=lambda x: x[1], reverse=True)[:top]
        new_run_dict[query_id] = {}
        for dataset_id, score in sorted_score:
            new_run_dict[query_id][dataset_id] = score
    return new_run_dict


def BDR_count_success(run_dict, success_eval_dict):
    cnt = 0
    for query_id, dataset_dict in run_dict.items():
        for dataset_id, score in dataset_dict.items():
            if success_eval_dict[query_id][dataset_id]:
                cnt += 1
    return cnt


def BDR_evaluation():
    success_eval_dict = read_BDR_success_eval_dict()
    qrels_dict = {}
    for qid, scores_dict in success_eval_dict.items():
        qrels_dict[qid] = {}
        for did, score in scores_dict.items():
            qrels_dict[qid][did] = int(score)
    runs = [
        ('bm25', read_BDR_bm25_run_dict()),
        ('ontological', read_ontological_run_dict('../data/ontological/BDR.txt')),
        ('distributed_kgtk', read_distributed_run_dict('../data/distributed/BDR_kgtk.json')),
        ('distributed_rdf2vec', read_distributed_run_dict('../data/distributed/BDR_rdf2vec.json')),
    ]
    for name, run in runs:
        print(name)
        for i in [10, 5, 1]:
            new_run = get_top_run_dict(run, i)
            cnt = BDR_count_success(new_run, success_eval_dict)
            total = len(qrels_dict) * i
            print(f'top {i} success franction: {cnt/total:.4f}')
        print()


if __name__ == '__main__':
    # for tc in ['acordar1', 'ntcir15', 'ntcir16']:
    #     for pf in ['ont', 'kgtk', 'rdf2vec']:
    #         bi_interpolate(test_collection=tc, profiling=pf, save_best=True)

    for tc in ['acordar1', 'ntcir15', 'ntcir16']:
        for pf in ['kgtk', 'rdf2vec']:
            tri_interpolate(test_collection=tc, emb=pf, save_best=True)

    # output_emnlp()
    # BDR_evaluation()
