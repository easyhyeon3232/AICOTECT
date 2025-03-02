import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

class Settings:
    DB_USERNAME : str = os.getenv("MARIADB_USERNAME")
    DB_PASSWORD : str = os.getenv("MARIADB_PASSWORD")
    DB_HOST : str = os.getenv("MARIADB_HOST", "localhost")
    DB_PORT : str = os.getenv("MARIADB_PORT", 3307)
    DB_DATABASE : str = os.getenv("MARIADB_DATABASE")

    DATABASE_URL = "mariadb+pymysql://root:test@10.11.52.113:3307/db"