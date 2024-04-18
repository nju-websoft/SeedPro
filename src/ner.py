import json
from concurrent.futures import ThreadPoolExecutor

import spacy
from constants import server_db_erm, select_in_limit, num_threads, logging_config, mysql_table_name_pattern, query_np_len_limit
from constants import wikidata_endpoint, user_agent, timeout, wbgetentities_limit, wbsearchentities_limit
from database import get_connection
import pymysql
import requests
import re
from nltk.tokenize import word_tokenize
from nltk import pos_tag, RegexpParser
from typing import Union, Literal, List, Optional, Dict, Tuple
from collections import defaultdict
import logging.config
from refined.inference.processor import Refined

logging.config.fileConfig(logging_config)


class NounPhraseRecognition(object):
    def __init__(self,
                 _id: Union[str, int],
                 retrieval_type: Union[Literal['query'], Literal['metadata']],
                 connection: Union[pymysql.connect, None] = None,
                 metadata_name: Union[None, Literal['ntcir'], Literal['acordar1']] = None,
                 query_name: Union[None, Literal['ntcir15'], Literal['ntcir16'], Literal['acordar1']] = None,
                 refined: Union[Refined, None] = None,
                 ):
        if connection:
            self.conn = connection
        else:
            self.conn = get_connection()
        self.retrieval_type = retrieval_type
        self.metadata_name = metadata_name
        self.query_name = query_name
        self.id = _id
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.logger = logging.getLogger('timerotating')
        if refined:
            self.refined = refined
        else:
            self.refined = Refined.from_pretrained(model_name='wikipedia_model_with_numbers', entity_set="wikidata")
        pass

    def get_ReFinED_result_database(self) -> Tuple[Tuple[str, str]]:
        """
        fetch ReFinED NER results from database by _id
        :return: Tuple[Tuple[str, str]], entity list, the first element is the text and the second element is the id.
        """
        cursor = self.conn.cursor()
        if self.retrieval_type == 'metadata':
            sql = f'SELECT span_text, entity FROM {self.metadata_name}_metadata_link_result_ReFinED WHERE dataset_id = (%s)'
        elif self.retrieval_type == 'query':
            sql = f'SELECT span_text, entity FROM {self.query_name}_query_link_result_ReFinED WHERE query_id = (%s)'
        else:
            raise ValueError(f"Invalid retrieval_type: {self.retrieval_type}. Please choose 'metadata' or 'query'.")
        cursor.execute(sql, self.id)
        results = cursor.fetchall()
        return results

    def get_ReFinED_result_run(self, text: str) -> List[Tuple[str, Union[str, None]]]:
        el_results = []
        spans = self.refined.process_text(text)
        for span in spans:
            span_text = span.text
            entity = span.predicted_entity
            if entity:
                entity = entity.wikidata_entity_id
            el_results.append((span_text, entity))
        return el_results

    def get_NR_result_database(self) -> Dict[str, Dict[str, Union[str, None]]]:
        """
        fetch Regex NR results from database by _id
        :return: Dict[str, Dict], key = text, value = {'entity_id': entity_id, 'label': label}
        """
        cursor = self.conn.cursor()
        if self.retrieval_type == 'metadata':
            sql = f'SELECT text, entity_id, label FROM {self.metadata_name}_metadata_NR WHERE dataset_id={self.id}'
        elif self.retrieval_type == 'query':
            sql = f'SELECT text, entity_id, label FROM {self.query_name}_query_NR WHERE query_id={self.id}'
        else:
            raise ValueError(f"Invalid retrieval_type: {self.retrieval_type}. Please choose 'metadata' or 'query'.")
        cursor.execute(sql)
        results = {}
        for text, entity_id, label in cursor.fetchall():
            results[text] = {'entity_id': entity_id, 'label': label}
        return results

    @staticmethod
    def find_rest_pos_tags(pos_tags: List[Tuple[str, str]], match_str: str) -> Tuple[
        List[Tuple[str, str]], List[Tuple[str, str]]]:
        match_str_len = match_str.count(' ') + 1
        for idx, pt in enumerate(pos_tags):
            if match_str in pt[0]:
                return pos_tags[:idx], pos_tags[idx + 1:]
            if match_str.startswith(pt[0]):
                sub_pts = pos_tags[idx: idx + match_str_len]
                if ' '.join(word for word, pos in sub_pts) == match_str:
                    return pos_tags[:idx], pos_tags[idx + match_str_len:]

    @staticmethod
    def get_N_end_post_tags(pos_tags: List[Tuple[str, str]]):
        while pos_tags and not pos_tags[-1][1].startswith('N'):
            pos_tags = pos_tags[:-1]
        return pos_tags

    def extract_nouns(self, sentence: str, ne_list: Optional[List[str]] = None):
        """
        Extract noun phrases from text, excluding named entities
        :param sentence: str, text
        :param ne_list: List[str], texts of named entities
        :return: List[List[Tuple[str, str]]]，a list containing all noun phrases. The tuple representing the text and
                 position of the word. e.g. [[('Environmental', 'NNP'), ('Compliance', 'NNP')], [('Year', 'NN')]]
        """
        words = word_tokenize(sentence)
        pos_tags = pos_tag(words)
        grammar = r"NP: {(<V\w+>|<NN\w?>)+.*<NN\w?>}"
        chunk_parser = RegexpParser(grammar)
        tree = chunk_parser.parse(pos_tags)
        nouns = []
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
            nouns.append(subtree.leaves())
            subtree.clear()
        nouns += [[(word, pos)] for word, pos in tree.leaves() if pos.startswith('N')]

        if ne_list:
            new_nouns = []
            ne_set = set(ne_list)
            for noun in nouns:
                noun_str = ' '.join(word for word, pos in noun)
                is_append = True
                for ne in ne_set:
                    if noun_str in ne:
                        is_append = False
                        break
                    elif ne in noun_str:
                        rest = self.find_rest_pos_tags(noun, ne)
                        if not rest:
                            is_append = False
                            break
                        else:
                            left_rest, right_rest = rest
                            left_rest = self.get_N_end_post_tags(left_rest)
                            if left_rest:
                                new_nouns.append(left_rest)
                            right_rest = self.get_N_end_post_tags(right_rest)
                            if right_rest:
                                new_nouns.append(right_rest)
                            is_append = False
                            break
                if is_append:
                    new_nouns.append(noun)
            return new_nouns
        else:
            return nouns

    def search_entity_from_wikidata(self, keyword) -> Union[List[Dict], None]:
        """
        link entities by wbsearchentities
        :param keyword: str
        :return: List[dict], for example:
        [{
            "id": "Q30",
            "title": "Q30",
            ...
            "label": "United States of America",
            ...
        },
        {
            "id": "Q15180",
            ...
        }, ...]
        """
        API_ENDPOINT = wikidata_endpoint
        params = {
            'action': 'wbsearchentities',
            'format': 'json',
            'language': 'en',
            'limit': wbsearchentities_limit,
            'search': keyword
        }
        headers = {
            'user-agent': user_agent
        }
        search_result = None
        try:
            r = requests.get(API_ENDPOINT, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                search_result = r.json()['search']
        except Exception as e:
            self.logger.error(f"[wikidata API] {self.retrieval_type} {self.id} {keyword}: {e}")
        finally:
            return search_result

    def query_wikidata_max_right_phrase(self, pos_tags: List[Tuple[str, str]], start_index: int = 0) -> Dict:
        index, right_text, right_entity_id, right_label, lemma, search_result = -1, None, None, None, None, None
        for index in range(start_index, len(pos_tags)):
            start_tag = pos_tags[index][1]
            if start_tag.startswith('N') or start_tag.startswith('V'):  # may be redundant
                right_text = " ".join(word for word, tag in pos_tags[index:])
                lemma = self.spacy_lemma(right_text)
                search_result = self.search_entity_from_wikidata(lemma)
                if search_result:
                    right_entity_id = search_result[0]['id']
                    right_label = search_result[0]['label']
                    break
            else:
                continue
        return {'index': index, 'text': right_text, 'entity_id': right_entity_id, 'label': right_label, 'lemma': lemma,
                'search_result': search_result}

    def query_wikidata_null_phrase(self, pos_tags: List[Tuple[str, str]], start_index: int = 0) -> Union[List[Dict]]:
        results = []
        if len(pos_tags) < 2:
            return results
        right_phrase_info = self.query_wikidata_max_right_phrase(pos_tags, start_index)
        right_index, right_entity_id = right_phrase_info['index'], right_phrase_info['entity_id']

        if right_entity_id:
            right_pos_tags = pos_tags[right_index:]
            result = {k: right_phrase_info[k] for k in ['text', 'entity_id', 'label', 'lemma', 'search_result']}
            result['pos_tags'] = right_pos_tags
            results.append(result)

            left_pos_tags = pos_tags[:right_index]
            while left_pos_tags and not left_pos_tags[-1][1].startswith('N'):
                left_pos_tags = left_pos_tags[:-1]
            if left_pos_tags:
                left_phrase_info = self.query_wikidata_max_right_phrase(left_pos_tags)
                left_entity_id = left_phrase_info['entity_id']
                if left_entity_id:
                    left_pos_tags = left_pos_tags[left_phrase_info['index']:]
                    result = {k: left_phrase_info[k] for k in ['text', 'entity_id', 'label', 'lemma', 'search_result']}
                    result['pos_tags'] = left_pos_tags
                    results.append(result)
        return results

    @staticmethod
    def check_valid_phrase(pos_tags: List[Tuple[str, str]]) -> bool:
        return pos_tags[-1][1].startswith('NN')

    def query_wikidata_len_limited_phrase(self, pos_tags: List[Tuple[str, str]], len_limit: int) -> Union[List[Dict]]:
        results = []  # {text, pos_tags, entity_id, label, lemma, search_result}
        if len(pos_tags) < 2:
            return results

        for np_len in range(len_limit, len(pos_tags)):
            for pos_idx in range(len(pos_tags) - np_len + 1):
                sub_pos_tags = pos_tags[pos_idx:pos_idx + np_len]
                if self.check_valid_phrase(sub_pos_tags):
                    sub_text = " ".join(word for word, tag in sub_pos_tags)
                    lemma = self.spacy_lemma(sub_text)
                    search_result = self.search_entity_from_wikidata(lemma)
                    if search_result:
                        entity_id = search_result[0]['id']
                        label = search_result[0]['label']
                        results.append({'text': sub_text, 'entity_id': entity_id, 'label': label, 'lemma': lemma,
                                        'search_result': search_result, 'pos_tags': sub_pos_tags})
        return results

    def get_entity_from_wikidata(self, entity_ids, props) -> Union[None, Dict[str, Dict]]:
        """
        get information of entities by wbgetentities
        :param entity_ids: List[str]
        :param props: List[str]
        :return: Dict[str, Dict], key = entity_id, value = information

        For example: entity_ids = ['Q30', 'Q15180'], props = ['labels'], return = {
            "Q30": {
                "type": "item",
                "id": "Q30",
                "labels": {
                    "en": {
                        "language": "en",
                        "value": "United States of America"
                    },
                    "fr": {
                        "language": "fr",
                        "value": "États-Unis"
                    },
                    ...
                }
            },
            "Q15180": {
                ...
            }
        }
        """
        if len(entity_ids) > wbgetentities_limit:
            raise ValueError("Exceed maximum number of entities")
        API_ENDPOINT = wikidata_endpoint
        params = {
            'action': 'wbgetentities',
            'format': 'json',
            'ids': '|'.join(entity_ids),
            'props': '|'.join(props)
        }
        headers = {
            'user-agent': user_agent
        }
        entities = None
        try:
            r = requests.get(API_ENDPOINT, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                entities = r.json()['entities']
        except Exception as e:
            self.logger.error(f"[wikidata API] {self.retrieval_type} {self.id} {entity_ids}: {e}")
        finally:
            return entities

    def get_entity_labels_database(self, entity_ids: List[str]) -> Dict[str, str]:
        """
        Get the entity labels from the `entities` table (preferred en).
        Note that NOT ALL entities can be found.

        :param entity_ids: List[str]
        :return: Dict[str, str], key = entity_id, value = label
        """
        cursor = self.conn.cursor()
        results = []
        for i in range(len(entity_ids) // select_in_limit + 1):
            sub_list = entity_ids[i * select_in_limit: (i + 1) * select_in_limit]
            if not sub_list:
                break
            # Create a string of placeholders for the IN clause
            placeholders = ', '.join(['%s'] * len(sub_list))
            sql = f"SELECT id, value FROM entities WHERE id IN ({placeholders})"
            cursor.execute(sql, tuple(sub_list))
            results += list(cursor.fetchall())
        return {x[0]: x[1] for x in results}

    def get_entity_labels_API(self, entity_ids: List[str]) -> defaultdict:
        """
        get the entity labels by wikidata API (preferred en)
        :param entity_ids: List[str]
        :return: Dict[str, str], key = entity_id, value = label
        """
        eid_label = defaultdict(lambda: None)
        for i in range(len(entity_ids) // wbgetentities_limit + 1):
            sub_list = entity_ids[i * wbgetentities_limit: (i + 1) * wbgetentities_limit]
            if not sub_list:
                break
            labels = self.get_entity_from_wikidata(entity_ids=sub_list, props=['labels'])
            if not labels:
                continue
            for entity_id, values in labels.items():
                if 'labels' not in values.keys():
                    print(entity_id)
                    continue
                if 'en' in values['labels'].keys():
                    label = values['labels']['en']['value']
                else:
                    label = list(values['labels'].values())[0]['value']
                eid_label[entity_id] = label
        return eid_label

    def batch_insert_and_update_database(self, sql, val):
        cursor = self.conn.cursor()
        table_name = re.compile(mysql_table_name_pattern).findall(sql)
        try:
            row = cursor.executemany(sql, val)
            self.conn.commit()
            self.logger.debug(f'{self.retrieval_type} id={self.id} table={table_name} insert/update {row} rows')
        except Exception as e:
            self.logger.error(
                f"{self.retrieval_type} id={self.id} table={table_name} insert/update error, sql: {sql}\n{e}",
                exc_info=True)
            self.conn.rollback()

    def delete_record_database(self, sql):
        cursor = self.conn.cursor()
        table_name = re.compile(mysql_table_name_pattern).findall(sql)
        try:
            row = cursor.execute(sql)
            self.conn.commit()
            self.logger.debug(f'{self.retrieval_type} id={self.id} table={table_name} delete {row} rows')
        except Exception as e:
            self.logger.error(f"{self.retrieval_type} id={self.id} table={table_name} delete error, sql: {sql}\n{e}",
                              exc_info=True)
            self.conn.rollback()

    def get_metadata_text(self) -> str:
        cursor = self.conn.cursor()
        if self.retrieval_type == 'metadata':
            if self.metadata_name == 'acordar1' or self.metadata_name == 'ntcir':
                sql = f"SELECT CONCAT(title, '. ', description) FROM {self.metadata_name}_dataset_metadata " \
                      f"WHERE dataset_id={self.id}"
                cursor.execute(sql)
                return cursor.fetchone()[0]
            elif self.metadata_name == 'BDR_candidate' or self.metadata_name == 'BDR_target':
                sql = f"SELECT CONCAT(title, '. ', description) FROM {self.metadata_name}_metadata " \
                      f"WHERE dataset_id={self.id}"
                cursor.execute(sql)
                return cursor.fetchone()[0]
        elif self.retrieval_type == 'query':
            if self.query_name in {'acordar1', 'ntcir15', 'ntcir16'}:
                sql = f"SELECT query_text FROM {server_db_erm}.{self.query_name}_query WHERE query_id=(%s)"
                cursor.execute(sql, self.id)
                return cursor.fetchone()[0]

    @staticmethod
    def check_valid_text(input_str: str) -> bool:
        return sum(1 for char in input_str if char.isalpha()) > 1 and input_str[0].isalnum()

    @staticmethod
    def remove_the_prefix(input_str: str) -> str:
        pattern = re.compile(r'^the\s*', re.IGNORECASE)
        return re.sub(pattern, '', input_str)

    @staticmethod
    def clean_text(text: str) -> str:
        # remove HTML element
        text = re.sub(r'<.*?>', '', text)
        # remove URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # remove email
        text = re.sub(r'\S+@\S+', '', text)
        return text

    def spacy_lemma(self, text: str) -> str:
        doc = self.spacy_nlp(text)
        lemma = " ".join([token.lemma_ if token.text.lower() not in {'data', 'us'} else token.text for token in doc])
        # if-else is used to avoid lemmatize data to datum
        return lemma

    def run(self):
        sentence = self.get_metadata_text()
        sentence = self.clean_text(sentence)
        ner_results = self.get_ReFinED_result_run(sentence)
        ner_results = [(self.remove_the_prefix(text), entity_id) for text, entity_id in ner_results
                       if self.check_valid_text(text)]  # remove invalid entities and remove "the " in text

        ne_list = [x[0] for x in ner_results]
        nouns = self.extract_nouns(sentence=sentence, ne_list=ne_list)
        results = []  # (dataset_id/query_id, text, pos_tags, entity_id, label, lemma, API_result)
        for noun in nouns:
            noun_str = " ".join(word for word, pos in noun)
            if not self.check_valid_text(noun_str):
                continue
            lemma = self.spacy_lemma(noun_str)
            search_result = self.search_entity_from_wikidata(lemma)
            if search_result:
                entity_id = search_result[0]['id']
                label = search_result[0]['label']
                results.append((self.id, noun_str, json.dumps(noun), entity_id, label, lemma, json.dumps(search_result)))
                continue
            query_result = self.query_wikidata_null_phrase(pos_tags=noun)
            for qr in query_result:
                if self.check_valid_text(qr['text']):
                    results.append((self.id, qr['text'], json.dumps(qr['pos_tags']), qr['entity_id'], qr['label'],
                                    qr['lemma'], json.dumps(qr['search_result'])))

        results_ner = []
        ner_linked = [(text, entity_id) for text, entity_id in ner_results if entity_id]
        ner_linked_labels = self.get_entity_labels_API([entity_id for text, entity_id in ner_linked])
        results_ner += [(self.id, text, None, entity_id, ner_linked_labels[entity_id], self.spacy_lemma(text), None)
                        for text, entity_id in ner_linked]
        for text, entity_id in ner_results:
            if not entity_id:
                pos_tags = pos_tag(word_tokenize(text))
                while pos_tags and not pos_tags[-1][1].startswith('N'):
                    pos_tags = pos_tags[:-1]
                if pos_tags:
                    new_text = ' '.join(word for word, pos in pos_tags)
                    lemma = self.spacy_lemma(new_text)
                    search_result = self.search_entity_from_wikidata(lemma)
                    if search_result:  # 直接能在API linking到
                        entity_id = search_result[0]['id']
                        label = search_result[0]['label']
                        results_ner.append((self.id, new_text, json.dumps(pos_tags), entity_id, label, lemma,
                                            json.dumps(search_result)))
                        continue
                    else:
                        query_result = self.query_wikidata_null_phrase(pos_tags=pos_tags)
                        for qr in query_result:
                            if self.check_valid_text(qr['text']):
                                results_ner.append((self.id, qr['text'], json.dumps(qr['pos_tags']), qr['entity_id'],
                                                    qr['label'], qr['lemma'], json.dumps(qr['search_result'])))
        new_results = []
        for res1 in results:
            is_append = True
            for res2 in results_ner:
                if res1[1] in res2[1]:
                    is_append = False
                    break
            if is_append:
                new_results.append(res1)
        new_results += results_ner

        if self.retrieval_type == 'metadata':
            sql = f'INSERT INTO `{self.metadata_name}_{self.retrieval_type}_NPR` ' \
                  f'(`dataset_id`, `text`, `pos_tags`, `entity_id`, `label`, `lemma`, `API_result`) VALUES ' \
                  f'(%s, %s, %s, %s, %s, %s, %s);'
        elif self.retrieval_type == 'query':
            sql = f'INSERT INTO `{self.query_name}_{self.retrieval_type}_NPR` ' \
                  f'(`query_id`, `text`, `pos_tags`, `entity_id`, `label`, `lemma`, `API_result`) VALUES ' \
                  f'(%s, %s, %s, %s, %s, %s, %s);'
        else:
            raise ValueError(f"Invalid retrieval_type: {self.retrieval_type}. Please choose 'metadata' or 'query'.")

        self.batch_insert_and_update_database(sql, new_results)

    def close(self):
        self.conn.close()


def npr_metadata_instance_run(params):
    dataset_id, metadata_name, retrieval_type = params
    npr = NounPhraseRecognition(_id=dataset_id, metadata_name=metadata_name, retrieval_type=retrieval_type)
    npr.run()
    npr.close()


def npr_query_instance_run(params):
    query_id, query_name, retrieval_type = params
    npr = NounPhraseRecognition(_id=query_id, query_name=query_name, retrieval_type=retrieval_type)
    npr.run()
    npr.close()


def metadata_run():
    metadata_name = 'acordar1'
    retrieval_type = 'metadata'
    conn = get_connection()
    sql = 'SELECT DISTINCT dataset_id FROM acordar1_pid'
    cursor = conn.cursor()
    cursor.execute(sql)
    dataset_ids = [x[0] for x in cursor.fetchall()]
    params_list = [(dataset_id, metadata_name, retrieval_type) for dataset_id in dataset_ids]

    logger = logging.getLogger('timerotating')
    logger.debug(f"NRP: dataset={metadata_name}, type={retrieval_type}, count={len(params_list)}")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(npr_metadata_instance_run, params_list)


def query_run():
    query_name = 'acordar1'
    retrieval_type = 'query'
    conn = get_connection(database=server_db_erm)
    sql = f'SELECT DISTINCT query_id FROM {query_name}_query'
    cursor = conn.cursor()
    cursor.execute(sql)
    query_ids = [x[0] for x in cursor.fetchall()]
    params_list = [(query_id, query_name, retrieval_type) for query_id in query_ids]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(npr_query_instance_run, params_list)


if __name__ == '__main__':
    metadata_run()
    query_run()

