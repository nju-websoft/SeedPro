import json

from search_recommend import eval_run_measures, get_qrels, read_bm25_run_dict, read_BDR_success_eval_dict, \
    read_BDR_bm25_run_dict, read_ontological_run_dict, read_distributed_run_dict, get_top_run_dict, BDR_count_success


def read_run(path):
    run = {}
    with open(path, 'r', encoding='UTF-8') as f:
        run = json.load(f)
    return run


def output_search_result():
    test_collections = ['acordar1', 'ntcir15', 'ntcir16']
    profiling_method = ['ont', 'rdf2vec', 'kgtk', 'ont_rdf2vec', 'ont_kgtk']
    for tc in test_collections:
        print(f'[ {tc} ]')
        qrels = get_qrels(tc)
        bm25_run = read_bm25_run_dict(tc)
        measure_scores = eval_run_measures(bm25_run, qrels, bm25_run)
        print(f' <bm25> ', end='')
        for measure, result in measure_scores.items():
            print(f'{measure}: {result["score"]:.4f}', end=' | ')
        print()
        for pm in profiling_method:
            print(f' <{pm}> ', end='')
            run = read_run(f'../data/results/{tc}_{pm}.json')
            best_measure_scores = eval_run_measures(run, qrels, bm25_run)
            for measure, result in best_measure_scores.items():
                print(f'{measure}: {result["score"]:.4f} (p={result["p"]:.4f})', end=' | ')
            print()


def output_recommendation_result():
    success_eval_dict = read_BDR_success_eval_dict()
    qrels_dict = {}
    for qid, scores_dict in success_eval_dict.items():
        qrels_dict[qid] = {}
        for did, score in scores_dict.items():
            qrels_dict[qid][did] = int(score)
    runs = [
        ('bm25', read_BDR_bm25_run_dict()),
        ('ontological', read_ontological_run_dict('../data/ontological/BDR.txt')),
        ('distributed_rdf2vec', read_distributed_run_dict('../data/distributed/BDR_rdf2vec.json')),
        ('distributed_kgtk', read_distributed_run_dict('../data/distributed/BDR_kgtk.json')),
    ]
    print('[ BDR ]')
    for name, run in runs:
        print(f'<{name}>')
        for i in [10, 5, 1]:
            new_run = get_top_run_dict(run, i)
            cnt = BDR_count_success(new_run, success_eval_dict)
            total = len(qrels_dict) * i
            print(f'top {i} SF: {cnt/total:.4f}', end=' | ')
        print()


if __name__ == '__main__':
    # output_search_result()
    output_recommendation_result()

