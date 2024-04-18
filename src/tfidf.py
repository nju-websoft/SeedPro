import json
from typing import Union, Iterable, List, Tuple, Dict

from scipy import stats

from database import get_connection
import pymysql
import pytrec_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import numpy as np
from ranx import Qrels, Run, evaluate, fuse, optimize_fusion, compare
from sklearn.metrics.pairwise import cosine_similarity

from transitive_closure import EntityClassHierarchy


def get_qrels(connection, dataset):
    cursor = connection.cursor()
    sql = 'USE xxx;'
    cursor.execute(sql)
    sql = f'SELECT query_id, dataset_id, score FROM `{dataset}_qrels`'
    if 'ntcir' in dataset:
        sql += ' WHERE type="test"'
    cursor.execute(sql)
    qrels_dict = {}
    for result in cursor.fetchall():
        query_id, dataset_id, score = result
        query_id = str(query_id)
        dataset_id = str(dataset_id)
        if query_id not in qrels_dict:
            qrels_dict[query_id] = {}
        qrels_dict[query_id][dataset_id] = score
    return qrels_dict


class ProfilingTFIDF(object):
    def __init__(self):
        pass

    @staticmethod
    def get_mapping_corpus_ids(table_name: str, ids: Iterable, level_col: str) -> Tuple[List[str], List]:
        id_class = {}
        for _id in ids:
            ech = EntityClassHierarchy(_id=_id, table_name=table_name)
            mapping_str = ech.get_mapping_str(level_col)
            id_class[_id] = mapping_str
        id_class_items = id_class.items()
        corpus = [x[1] for x in id_class_items]
        ids = [x[0] for x in id_class_items]
        return corpus, ids

    @staticmethod
    def compute_run_dict(ds_df: pd.DataFrame, qr_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        query_id and dataset_id have been converted to str

        :param ds_df: pandas.DataFrame, dataset vector, index = dataset_id, column = feature_name
        :param qr_df: pandas.DataFrame, query vector, index = query_id, column = feature_name
        :return: dict, {query_id1: {dataset_id1: score1, dataset_id2: score2, ...}, ...}
        """
        # df = qr_df.dot(ds_df.T)  # row_index: query_id, col_name: dataset_id
        similarity_matrix = cosine_similarity(qr_df, ds_df)
        df = pd.DataFrame(similarity_matrix, columns=ds_df.index, index=qr_df.index)

        run_dict = {}
        for index, row in df.iterrows():
            index = str(index)
            run_dict[index] = {}
            for col, value in row.items():
                col = str(col)
                if value > 0:
                    run_dict[index][col] = value
        run_dict = {k: v for k, v in run_dict.items() if len(v) > 0}
        return run_dict

    @staticmethod
    def transform_tfidf_dataframe(ds_corpus: List[str], ds_ids: List,
                                  qr_corpus: List[str], qr_ids: List) -> Tuple[pd.DataFrame, pd.DataFrame]:
        vectorizer = TfidfVectorizer()
        ds_matrix = vectorizer.fit_transform(ds_corpus)
        ds_df = pd.DataFrame.sparse.from_spmatrix(ds_matrix, columns=vectorizer.get_feature_names_out(), index=ds_ids)
        qr_matrix = vectorizer.transform(qr_corpus)
        qr_df = pd.DataFrame.sparse.from_spmatrix(qr_matrix, columns=vectorizer.get_feature_names_out(), index=qr_ids)
        return ds_df, qr_df

    @staticmethod
    def rerank_BM25(run_dict: dict, bm25_run_dict: dict, save: Union[str, None] = None) -> dict:
        new_run_dict = {}
        for query_id, dataset_dict in run_dict.items():
            new_run_dict[query_id] = {}
            if query_id in bm25_run_dict:
                bm25_dict = bm25_run_dict[query_id]
            else:
                bm25_dict = {}
            for dataset_id, score in dataset_dict.items():
                if dataset_id in bm25_dict:
                    new_run_dict[query_id][dataset_id] = score
        run_dict = new_run_dict
        if save:
            with open(save, 'w', encoding='UTF-8') as f:
                for query_id, dataset_dict in run_dict.items():
                    for dataset_id, score in dataset_dict.items():
                        row = '\t'.join([str(query_id), str(dataset_id), str(score)])
                        f.write(row + '\n')
        return run_dict

    @staticmethod
    def save_run_dict(run_dict: dict, save: str, top: Union[int, None] = None):
        with open(save, 'w', encoding='UTF-8') as f:
            for query_id, dataset_dict in run_dict.items():
                sorted_score = sorted(dataset_dict.items(), key=lambda x: x[1], reverse=True)
                if top:
                    sorted_score = sorted_score[:top]
                for dataset_id, score in sorted_score:
                    row = '\t'.join([str(query_id), str(dataset_id), str(score)])
                    f.write(row + '\n')

    @staticmethod
    def get_top_run_dict(run_dict: dict, top: int) -> dict:
        new_run_dict = {}
        for query_id, dataset_dict in run_dict.items():
            sorted_score = sorted(dataset_dict.items(), key=lambda x: x[1], reverse=True)[:top]
            new_run_dict[query_id] = {}
            for dataset_id, score in sorted_score:
                new_run_dict[query_id][dataset_id] = score
        return new_run_dict

    @staticmethod
    def pytrec_eval(run_dict: dict, qrels: dict, measures: Iterable[str]):
        """
        :param run_dict:
        :param qrels:
        :param measures: e.g.['ndcg_cut_5', 'ndcg_cut_10', 'map_cut_5', 'map_cut_10']
        :return:
        """
        qrels = {k: v for k, v in qrels.items() if k in run_dict.keys()}
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
        results = evaluator.evaluate(run_dict)
        return results


def read_acordar1_run_dict():
    """
    :param baseline_method: str, ["BM25F", "BM25F [d]", "BM25F [m]", "FSDM", ..., "LMD", ..., "TF-IDF", ...]
    :return: run_dict
    """
    path = f"../data/bm25/acordar1.txt"
    run = {}
    with open(path) as run_file:
        for line in run_file:
            list = line.split('\t')
            query_id = list[0]
            dataset_id = list[2]
            score = float(list[4])
            if query_id not in run:
                run[query_id] = {}
            run[query_id][dataset_id] = score
    return run

def read_ntcir_run_dict(name: str, top: int = 10):
    """
    :param top: int, retain the top k results with the highest scores
    :param name: str, ["ntcir15", "ntcir16"]
    :return: run_dict
    """
    conn = get_connection()
    sql = 'SELECT hash_id, dataset_id FROM ntcir_dataset_metadata'
    cursor = conn.cursor()
    cursor.execute(sql)
    hash_2_ds = {x[0]: x[1] for x in cursor.fetchall()}

    path = f"../data/bm25/{name}_orge-e-2.txt"
    run = {}
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


def read_profiling_run_dict(path: str):
    run = {}
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f:
            query_id, dataset_id, score = line.split('\t')
            score = float(score)
            if query_id not in run:
                run[query_id] = {}
            run[query_id][dataset_id] = score
    return run


def acordar1_run():
    pt = ProfilingTFIDF()
    conn = get_connection()
    cursor = conn.cursor()

    metadata_table = 'acordar1_metadata_NPR'
    sql = f'SELECT DISTINCT dataset_id FROM {metadata_table}'
    cursor.execute(sql)
    ds_ids = [x[0] for x in cursor.fetchall()]
    ds_corpus, ds_ids = pt.get_mapping_corpus_ids(table_name=metadata_table, ids=ds_ids, level_col='class_id_all')

    query_table = 'acordar1_query_NPR_plus'
    sql = f'SELECT DISTINCT query_id FROM {query_table}'
    cursor.execute(sql)
    qr_ids = [x[0] for x in cursor.fetchall()]
    qr_corpus, qr_ids = pt.get_mapping_corpus_ids(table_name=query_table, ids=qr_ids, level_col='class_id_all')

    ds_df, qr_df = pt.transform_tfidf_dataframe(ds_corpus=ds_corpus, ds_ids=ds_ids, qr_corpus=qr_corpus, qr_ids=qr_ids)

    run_dict = pt.compute_run_dict(ds_df=ds_df, qr_df=qr_df)
    qrels_dict = get_qrels(connection=conn, dataset='acordar1')
    qrels_dict = {k: v for k, v in qrels_dict.items() if k in run_dict}
    bm25_run_dict = read_acordar1_run_dict()
    bm25_run_dict = {k: v for k, v in bm25_run_dict.items() if k in run_dict}

    run_dict = pt.rerank_BM25(run_dict=run_dict, bm25_run_dict=bm25_run_dict,
                              save='results/acordar1_npr_rerank_BM25_10.txt')


def ttest_significance(score_dict_1, score_dict_2):
    scoreAll1, scoreAll2 = [], []
    for k, v in score_dict_1.items():
        scoreAll1.append(v)
        scoreAll2.append(score_dict_2[k])
    t, p = stats.ttest_rel(scoreAll1, scoreAll2)
    return t, p


def ntcir_run():
    pt = ProfilingTFIDF()
    conn = get_connection()
    cursor = conn.cursor()

    metadata_table = 'ntcir_metadata_NPR'
    sql = f'SELECT DISTINCT dataset_id FROM {metadata_table}'
    cursor.execute(sql)
    ds_ids = [x[0] for x in cursor.fetchall()]
    ds_corpus, ds_ids = pt.get_mapping_corpus_ids(table_name=metadata_table, ids=ds_ids, level_col='class_id_all')

    query_name = 'ntcir15'
    query_table = f'{query_name}_query_NPR_plus'
    sql = f'SELECT DISTINCT query_id FROM {query_table}'
    cursor.execute(sql)
    qr_ids = [x[0] for x in cursor.fetchall()]
    qr_corpus, qr_ids = pt.get_mapping_corpus_ids(table_name=query_table, ids=qr_ids, level_col='class_id_all')

    ds_df, qr_df = pt.transform_tfidf_dataframe(ds_corpus=ds_corpus, ds_ids=ds_ids, qr_corpus=qr_corpus, qr_ids=qr_ids)

    run_dict = pt.compute_run_dict(ds_df=ds_df, qr_df=qr_df)
    qrels_dict = get_qrels(connection=conn, dataset=query_name)
    qrels_dict = {k: v for k, v in qrels_dict.items() if k in run_dict}
    bm25_run_dict = read_ntcir_run_dict(query_name)
    run_dict = {k: v for k, v in run_dict.items() if k in bm25_run_dict}
    bm25_run_dict = {k: v for k, v in bm25_run_dict.items() if k in run_dict}

    run_dict = pt.rerank_BM25(run_dict=run_dict, bm25_run_dict=bm25_run_dict,
                              save=f'results/{query_name}_npr_rerank_BM25_10.txt')


class NormFuse(object):
    @staticmethod
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

    @staticmethod
    def weighted_fuse(run1: Dict[str, Dict[str, float]], run2: Dict[str, Dict[str, float]], default_value: float,
                      weights: Tuple[float, float]) -> Dict[str, Dict[str, float]]:
        w1, w2 = weights
        new_run_dict = {}
        for qid, scores in run1.items():
            new_run_dict[qid] = {k: run1[qid][k] * w1 + run2.get(qid, {}).get(k, default_value) * w2 for k in
                                 scores.keys()}
        return new_run_dict

    def optimize_weights(self, run1: Dict[str, Dict[str, float]], run2: Dict[str, Dict[str, float]],
                         default_value: float,
                         qrels: Dict[str, Dict[str, int]], metrics: str = 'ndcg_cut_5', output: bool = False) -> Tuple:
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, [metrics])
        results = {}
        for i in range(0, 11):
            weights = (round(i / 10.0, 1), round(1.0 - i / 10.0, 1))
            combined_run = self.weighted_fuse(run1, run2, default_value, weights)
            combined_results = evaluator.evaluate(combined_run)
            score = sum([v[metrics] for v in combined_results.values()]) / len(qrels)
            results[weights] = score
        if output:
            print('-' * 20)
            for weights in sorted(results.keys()):
                print(f'{weights} | {results[weights]:.4f}')
            print('-' * 20)
        best_weights, best_score = max(results.items(), key=lambda x: x[1])
        best_run = self.weighted_fuse(run1, run2, default_value, best_weights)
        best_results = evaluator.evaluate(best_run)
        return best_weights, best_score, best_results

    def temp_run(self):
        # run_path = 'results/acordar1_npr_rerank_BM25_10.txt'
        run_path = 'results/ntcir16_npr_rerank_BM25_10.txt'
        # run_path = 'results/BDR_npr_rerank_BM25_10.txt'
        print(run_path)
        run = read_profiling_run_dict(run_path)

        # bm25_run = read_acordar1_run_dict('BM25F [m]')
        bm25_run = read_ntcir_run_dict('ntcir16')
        conn = get_connection()
        qrels = get_qrels(conn, 'ntcir16')
        # bm25_run, qrels = read_BDR_BM25run_qrels_dict()

        metrics = 'ndcg_cut_5'
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, [metrics])
        bm25_results = evaluator.evaluate(bm25_run)
        score = sum([v[metrics] for v in bm25_results.values()]) / len(qrels)
        print(score)
        profiling_results = evaluator.evaluate(run)
        score = sum([v[metrics] for v in profiling_results.values()]) / len(qrels)
        print(score)

        # norm_list = ["min-max", "max", "sum", "zmuv", "rank", "borda"]
        # norm_lists = [(x, y) for x in norm_list for y in [x, None]]
        # norm_lists.remove(('zmuv', None))
        norm_lists = [("min-max", "min-max")]

        best_weights, best_score, best_results = (1.0, 0.0), 0, {}
        best_norm1, best_norm2 = None, None
        for n1, n2 in norm_lists:
            run1 = self.ranx_norm(bm25_run, n1)
            run2 = self.ranx_norm(run, n2)
            default_value = 0
            if n2 == 'zmuv':
                default_value = min([value for sub_dict in run2.values() for value in sub_dict.values()])
            local_bw, local_bs, local_br = self.optimize_weights(run1, run2, default_value, qrels, metrics,
                                                                 output=False)
            if local_bs > best_score:
                best_weights, best_score, best_results = local_bw, local_bs, local_br
                best_norm1, best_norm2 = n1, n2
        print()
        print(f'best_weights: {best_weights}, best_score: {best_score}')
        print(f'best_norm_bm25: {best_norm1}, best_norm_profiling: {best_norm2}')

        score_bm25 = {k: bm25_results.get(k, {metrics: 0})[metrics] for k in qrels.keys()}
        score_profiling = {k: best_results.get(k, {metrics: 0})[metrics] for k in qrels.keys()}
        t, p = ttest_significance(score_bm25, score_profiling)
        print(f't: {t}, p: {p}')


def BDR_compute_success():
    conn = get_connection()
    cursor = conn.cursor()
    sql = 'SELECT dataset_id, title, description, categories FROM BDR_candidate_metadata'
    cursor.execute(sql)
    candidate_metadata = cursor.fetchall()
    sql = 'SELECT dataset_id, title, description, categories FROM BDR_target_metadata'
    cursor.execute(sql)
    target_metadata = cursor.fetchall()
    results = {}
    for target in target_metadata:
        t_dataset_id, t_title, t_description, t_categories = target
        categories_flag = True
        if t_categories == '':
            categories_flag = False
        t_categories = t_categories.split(', ')
        results[t_dataset_id] = {}
        for candidate in candidate_metadata:
            c_dataset_id, c_title, c_description, c_categories = candidate
            if t_title == c_title and t_description == c_description:
                results[t_dataset_id][c_dataset_id] = True
                continue
            if categories_flag and c_categories != '':
                c_categories = c_categories.split(', ')
                combined_list = [(x, y) for x in t_categories for y in c_categories]
                for tc, cc in combined_list:
                    if tc in cc or cc in tc:
                        results[t_dataset_id][c_dataset_id] = True
                        break
                if c_dataset_id not in results[t_dataset_id].keys():
                    results[t_dataset_id][c_dataset_id] = False
                continue
            results[t_dataset_id][c_dataset_id] = False
    with open('BDR/success_eval.json', 'w', encoding='UTF-8') as f:
        json.dump(results, f)


def BDR_run():
    pt = ProfilingTFIDF()
    conn = get_connection()
    cursor = conn.cursor()

    candidate_table = 'BDR_candidate_metadata_NPR'
    sql = f'SELECT DISTINCT dataset_id FROM {candidate_table}'
    cursor.execute(sql)
    ds_ids = [x[0] for x in cursor.fetchall()]
    ds_corpus, ds_ids = pt.get_mapping_corpus_ids(table_name=candidate_table, ids=ds_ids, level_col='class_id_all')

    target_table = f'BDR_target_metadata_NPR'
    sql = f'SELECT DISTINCT dataset_id FROM {target_table}'
    cursor.execute(sql)
    qr_ids = [x[0] for x in cursor.fetchall()]
    qr_corpus, qr_ids = pt.get_mapping_corpus_ids(table_name=target_table, ids=qr_ids, level_col='class_id_all')

    ds_df, qr_df = pt.transform_tfidf_dataframe(ds_corpus=ds_corpus, ds_ids=ds_ids, qr_corpus=qr_corpus, qr_ids=qr_ids)

    run_dict = pt.compute_run_dict(ds_df=ds_df, qr_df=qr_df)

    bm25_run_dict = read_BDR_run_dict('../data/bm25/BDR.json')
    bm25_run_dict = pt.get_top_run_dict(bm25_run_dict, 10)
    run = pt.rerank_BM25(run_dict=run_dict, bm25_run_dict=bm25_run_dict, save=f'results/BDR_npr_rerank_BM25_10.txt')


def read_BDR_run_dict(path):
    run_dict = {}
    with open(path, 'r', encoding='UTF-8') as f:
        data = json.load(f)
        for query, scores in data.items():
            run_dict[query] = {}
            for score in scores:
                run_dict[query][str(score[0])] = score[1]
    return run_dict


if __name__ == '__main__':
    acordar1_run()
    ntcir_run()
    BDR_run()

