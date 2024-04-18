from pymysql import connect
from constants import server_host, server_port, server_user, server_password, server_db_profiling
import traceback


def get_connection(host=server_host,
                   port=server_port,
                   user=server_user,
                   password=server_password,
                   database=server_db_profiling):
    conn = connect(host=host,
                   port=port,
                   user=user,
                   password=password,
                   database=database,
                   )
    return conn


class MySQL():
    conn = None
    cur = None

    def __init__(self, logger=None, db_name=None):
        super().__init__(logger)
        db_name += '_db'
        assert db_name in globals(), f'invalid db_name: {db_name}'
        db = globals()[db_name]
        self.conn = connect(**db)
        self.cur = self.conn.cursor()

    def __del__(self):
        pass
        # if self.conn is not None:
        #     self.conn.close()

    def commit(self):
        self.conn.commit()

    def execute(self, sql, args=None):
        try:
            self.cur.execute(sql, args)
        except:
            self.conn.rollback()
            self.info(f'excute error: {sql}')
            traceback.print_exc()
            return False
        return True

    def excutemany(self, sql, args=None):
        try:
            self.cur.executemany(sql, args)
        except:
            self.conn.rollback()
            self.info(f'excutemany error: {sql}')
            traceback.print_exc()
            return False
        return True

    def query(self, sql, args=None):
        if self.cur.execute(sql, args):
            return self.cur.fetchall()
        return None

    def drop_table(self, table_name):
        sql = f'DROP TABLE IF EXISTS {table_name}'
        if self.execute(sql):
            self.commit()
