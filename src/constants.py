# server database
server_host = 'xxx'
server_port = 3306
server_user = 'xxx'
server_password = 'xxx'
server_db_profiling = 'xxx'
server_db_erm = 'xxx'
select_in_limit = 100

# requests
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'
wikidata_endpoint = 'https://www.wikidata.org/w/api.php'
wbgetentities_limit = 50
wbsearchentities_limit = 10
sleep_time = 0
timeout = 20

# ThreadPoolExecutor
num_threads = 100

# logs
logging_config = 'logging.conf'
mysql_table_name_pattern = r"(?:INSERT INTO|DELETE FROM|UPDATE)\s+(?:`?([a-zA-Z_][a-zA-Z0-9_]*)`?|\*|\S+)"

# specific abbreviation
abbreviation_dict = {
    'epa': 'United States Environmental Protection Agency',
    'tlc': 'New York City Taxi and Limousine Commission',
    'nycdohmh': 'New York City Department of Health and Mental Hygiene',
    'cdc': 'Centers for Disease Control and Prevention',
    'us': 'United States of America',
    'nos': 'nitrous oxide',
    'fhwa': 'Federal Highway Administration',
    'nodc': 'Chitin synthase blr2027',
    'ccp': 'central counterparty clearing',
    'usdoc': 'US doctors call for tracking of suicides among medical trainees',
    'co2': 'carbon dioxide',
    'rov': 'remotely operated underwater vehicle',
    'qa/qc': 'quality assurance',
    'twa800': 'TWA Flight 800',
    'aris': 'Architecture of Integrated Information Systems',
    'oasdi': 'Old-Age, Survivors, and Disability Insurance',
    'lrt': 'light rail',
    'pdsi': 'Palmer Drought Index',
}

# query
query_np_len_limit = 2


#-----------------


LOG_LOCATION = '/home2/xxx/code/log'

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36'}

dataset_profiling_db = {
    'database': 'xxx',
    'user': 'xxx',
    'password': 'xxx',
    'host': 'xxx',
    'port': 3306
}


data_path = '/home/xxx/code/dataset_profiling/data'
qrels_path = f'{data_path}/qrels'
sparse_path = f'{data_path}/sparse_res'
retrieve_path = f'{data_path}/retrieve_res'
rerank_path = f'{data_path}/rerank_res'
embedding_path = f'{data_path}/embeddings'

test_collections = ['acordar1', 'ntcir']
methods = ['FALCON2', 'ReFinED']

