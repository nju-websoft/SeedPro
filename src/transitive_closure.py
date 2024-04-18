import logging.config
import re
from typing import List, Iterable, Dict, Union, Sequence

import pandas as pd

from constants import server_host, server_port, server_user, server_password, server_db_profiling, select_in_limit, \
    logging_config, abbreviation_dict, mysql_table_name_pattern

import pymysql
import time

from database import get_connection


class EntityClassHierarchy(object):
    def __init__(self,
                 _id: Union[str, int],
                 table_name: str,
                 connection: Union[pymysql.connect, None] = None,
                 similarity_threshold: float = 0.9,
                 edit_dist_threshold: int = 10,
                 ):
        if connection:
            self.conn = connection
        else:
            self.conn = get_connection()
        self.table_name = table_name
        if 'query' in table_name:
            self.id_col = 'query_id'
        elif 'metadata' in table_name:
            self.id_col = 'dataset_id'
        self.id = _id
        self.similarity_threshold = similarity_threshold
        self.edit_dist_threshold = edit_dist_threshold
        self.abbreviation_dict = abbreviation_dict
        self.logger = logging.getLogger('timerotating')
        pass

    def is_abbreviation(self, text: str, label: str) -> bool:
        text = text.replace(' ', '').replace('.', '').lower()
        flag0 = self.abbreviation_dict.get(text, None) == label
        flag1 = text.lower() == ''.join(x[0] for x in label.split()).lower()
        flag2 = text.lower() == ''.join(x[0] for x in label.split() if x[0].isupper()).lower()
        return flag0 or flag1 or flag2

    def get_entity_ids(self) -> List[str]:
        sql = f'SELECT text, entity_id, label, similarity, edit_dist, abbreviation FROM {self.table_name} ' \
              f'WHERE {self.id_col} = (%s)'
        cursor = self.conn.cursor()
        cursor.execute(sql, self.id)
        results = []
        for text, entity_id, label, similarity, edit_dist, abbreviation in cursor.fetchall():
            if not similarity:
                continue
            if similarity > self.similarity_threshold or edit_dist < self.edit_dist_threshold or abbreviation == b'1' \
                    or self.is_abbreviation(text, label):
                results.append(entity_id)
        return results

    def query_entity_class_l1(self, entity_id) -> set:
        cursor = self.conn.cursor()
        sql = f"SELECT DISTINCT classes_p31.class_id FROM classes_p31 WHERE classes_p31.id='{entity_id}'"
        cursor.execute(sql)
        return set(x[0] for x in cursor.fetchall())

    def query_entity_class_all(self, entity) -> set:
        if entity is None:
            return set()

        cursor = self.conn.cursor()

        def dfs(node):
            visited.add(node)
            sql = f"SELECT DISTINCT classes_p31.class_id FROM classes_p31 WHERE classes_p31.id='{node}'"
            cursor.execute(sql)
            results = [x[0] for x in cursor.fetchall()]
            sql = f"SELECT DISTINCT classes_p279.class_id FROM classes_p279 WHERE classes_p279.id='{node}'"
            cursor.execute(sql)
            results += [x[0] for x in cursor.fetchall()]
            for class_id in results:
                if class_id not in visited:
                    dfs(class_id)

        visited = set()

        dfs(entity)
        visited.remove(entity)

        return visited

    def map_entity_class_ids_database(self, entity_ids: Sequence[str], level_col: str):
        cursor = self.conn.cursor()
        results = []
        for i in range(len(entity_ids) // select_in_limit + 1):
            sub_list = entity_ids[i*select_in_limit: (i+1)*select_in_limit]
            if not sub_list:
                break
            # Create a string of placeholders for the IN clause
            placeholders = ', '.join(['%s'] * len(sub_list))
            sql = f"SELECT entity_id, {level_col} FROM entity_mapping WHERE entity_id IN ({placeholders})"
            cursor.execute(sql, tuple(sub_list))
            results += list(cursor.fetchall())
        return {x[0]: x[1] for x in results}

    def map_entity_class_ids_graph(self, entity_ids: Iterable[str], level_col: str):
        entity_ids = set(entity_ids)
        if level_col == 'class_id_l1':
            return {entity_id: ','.join(self.query_entity_class_l1(entity_id)) for entity_id in entity_ids}
        elif level_col == 'class_id_all':
            return {entity_id: ','.join(self.query_entity_class_all(entity_id)) for entity_id in entity_ids}
        else:
            raise ValueError(f"Invalid level_col: {level_col}. Must be class_id_l1 or class_id_all.")

    def batch_insert_and_update_database(self, sql, val):
        cursor = self.conn.cursor()
        table_name = re.compile(mysql_table_name_pattern).findall(sql)
        try:
            row = cursor.executemany(sql, val)
            self.conn.commit()
            self.logger.debug(f'source_table={self.table_name} id={self.id} target_table={table_name} insert/update {row} rows')
        except Exception as e:
            self.logger.error(f"source_table={self.table_name} id={self.id} target_table={table_name} insert/update error, sql: {sql}\n{e}", exc_info=True)
            self.conn.rollback()

    def update_entity_mapping(self, eid_cids: Dict[str, dict]):
        sql = "INSERT INTO `entity_mapping` (`entity_id`, `class_id_l1`, `class_id_all`) VALUES (%s, %s, %s);"
        val = []
        for entity_id, class_ids_dict in eid_cids.items():
            val.append((entity_id, class_ids_dict['class_id_l1'], class_ids_dict['class_id_all']))
        self.batch_insert_and_update_database(sql, val)

    def get_and_update_entity_mapping(self, entity_ids: Sequence[str], level_col: str) -> Dict[str, str]:
        eid_cids = self.map_entity_class_ids_database(entity_ids=entity_ids, level_col=level_col)
        rest_entity_ids = set(entity_ids).difference(set(eid_cids.keys()))

        if rest_entity_ids:
            rest_eid_cids_l1 = self.map_entity_class_ids_graph(entity_ids=rest_entity_ids, level_col='class_id_l1')
            rest_eid_cids_all = self.map_entity_class_ids_graph(entity_ids=rest_entity_ids, level_col='class_id_all')
            rest_eid_cids = {}  # avoid mixing '' and Null
            for entity_id, cids_all in rest_eid_cids_all.items():
                rest_eid_cids[entity_id] = {}
                if cids_all:
                    rest_eid_cids[entity_id]['class_id_all'] = cids_all
                else:
                    rest_eid_cids[entity_id]['class_id_all'] = None
                cids_l1 = rest_eid_cids_l1.get(entity_id, None)
                if cids_l1:
                    rest_eid_cids[entity_id]['class_id_l1'] = cids_l1
                else:
                    rest_eid_cids[entity_id]['class_id_l1'] = None

            self.update_entity_mapping(eid_cids=rest_eid_cids)
            for entity_id, class_ids_dict in rest_eid_cids.items():
                eid_cids[entity_id] = class_ids_dict[level_col]

        return eid_cids

    def get_mapping_str(self, level_col: str):
        entity_ids = self.get_entity_ids()
        eid_cids = self.get_and_update_entity_mapping(entity_ids=entity_ids, level_col=level_col)
        eid_cids = {k: ','.join([k, v]) if v else k for k, v in eid_cids.items()}
        return ','.join([eid_cids[entity_id] for entity_id in entity_ids if eid_cids[entity_id]])


def acordar1_run():
    conn = get_connection()
    cursor = conn.cursor()

    sql = 'SELECT DISTINCT dataset_id FROM acordar1_metadata_NPR'
    cursor.execute(sql)
    for x in cursor.fetchall():
        dataset_id = x[0]
        ech = EntityClassHierarchy(_id=dataset_id, table_name='acordar1_metadata_NPR')
        mapping_str = ech.get_mapping_str('class_id_all')

    sql = 'SELECT DISTINCT query_id FROM acordar1_query_NPR'
    cursor.execute(sql)
    for x in cursor.fetchall():
        query_id = x[0]
        ech = EntityClassHierarchy(_id=query_id, table_name='acordar1_query_NPR')
        mapping_str = ech.get_mapping_str('class_id_all')


if __name__ == '__main__':
    acordar1_run()




