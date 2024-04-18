import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

from text2vec import Similarity, EncoderType
from database import MySQL
from common import json_load, json_dump
from Levenshtein import distance
import traceback


db = MySQL(db_name='dataset_profiling')

from constants import abbreviation_dict

def is_abbreviation(text: str, label: str) -> bool:
    text = text.replace(' ', '').replace('.', '').lower()
    flag0 = abbreviation_dict.get(text, None) == label
    flag1 = text.lower() == ''.join(x[0] for x in label.split()).lower()
    flag2 = text.lower() == ''.join(x[0] for x in label.split() if x[0].isupper()).lower()
    return flag0 or flag1 or flag2


sim_model = Similarity(model_name_or_path='xxx/text2vec-base-multilingual',
                       encoder_type=EncoderType.FIRST_LAST_AVG)

def get_similarity(text1, text2):
    score = sim_model.get_score(text1, text2)
    # print("{} \t\t {} \t\t Score: {:.4f}".format(wiki_text, metadata, score))
    return score

collection = 'BDR_candidate'
field = 'metadata'
table = f'{collection}_{field}_NPR'

all_data = db.query(f'select {"dataset" if field == "metadata" else "query"}_id, text, label from {table} where similarity is NULL;')
print(len(all_data))

for i in range(0, len(all_data), 1000):
    cur = all_data[i: i+1000]

    dataset_text2info = {}
    for j, line in enumerate(cur):
        try:
            dataset_id = line[0]
            text = line[1]
            label = line[2]
            abb = 1 if is_abbreviation(text, label) else 0
            dis = distance(text, label)
            score = get_similarity(text, label)
            # print(text, label, dis, score)
            if dataset_id not in dataset_text2info:
                dataset_text2info[dataset_id] = {}
            if text not in dataset_text2info[dataset_id]:
                dataset_text2info[dataset_id][text] = {}
            
            # dataset_text2info[dataset_id][text] = {'label': label, 'distance': dis, 'score': score, 'abbreviation': abb}
            dataset_text2info[dataset_id][text][label] = {'distance': dis, 'score': score, 'abbreviation': abb}
            # dataset_text2info[dataset_id][text] = {'label': label, 'abbreviation': abb}
            # text2info[line[0]] = {'label': line[1], 'distance': dis, 'score': score}
        except:
            traceback.print_exc()
    print(i)

    data_list = []
    for dataset_id, info in dataset_text2info.items():
        for text, extra in info.items():
            # data_list.append((extra['distance'], extra['score'], extra['abbreviation'], dataset_id, text))
            # data_list.append((extra['abbreviation'], dataset_id, text, extra['label']))
            for label, sub in extra.items():
                data_list.append((sub['distance'], sub['score'], sub['abbreviation'], dataset_id, text, label))
            # data_list.append((extra['distance'], extra['score'], extra['abbreviation'], dataset_id, text, extra['label']))

    print(len(data_list))

    sql = f'update {table} set edit_dist=%s, similarity=%s, abbreviation=%s where {"dataset" if field == "metadata" else "query"}_id=%s and text=%s and label=%s'
    # sql = 'update acordar1_metadata_NPR set abbreviation=%s where dataset_id=%s and text=%s and label=%s'
    
    db.excutemany(sql, data_list)
    db.commit()
