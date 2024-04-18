import pymysql
# from DBUtils.PooledDB import PooledDB

db_info = {
    'host': 'xxx',
    'user': 'xxx',
    'password': 'xxx',
    'db': 'xxx',
    'port': 3306,
}

db = pymysql.connect(**db_info)
cursor = db.cursor()
