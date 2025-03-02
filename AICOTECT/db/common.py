import pymysql


# DB 연결
def connection():
    try:
        conn = pymysql.connect(
            host="10.11.52.113",  # IP
            port=3307,         # PORT
            user="root",       # ID(root:최고관리자)
            password="test",   # PW
            db="db",
            charset="utf8",
            autocommit=True,
#             cursorclass=pymysql.cursors.DictCursor
        )
        return conn
    except pymysql.Error as e:
        print(f"MARIADB 연결 실패: {e}")
