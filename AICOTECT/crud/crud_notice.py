# DAO(Data Access Object)
from schemas.notice import NoticeDTO
from sqlalchemy.orm import Session
from models.notice import NoticeORM
from datetime import datetime

# DB 사용 방법
#   1. Connection 맺기(Session 정보 획득 -> 접속할 DB의 정보)
#   2. ORM 객체를 생성(with DATA)
#   3. 사용!

# session.py -> Connection 맺고 Session 객체 획득
# config.py -> DB Connection을 맺기 위한 정보들
# models/message.py -> ORM을 사용하기 위한 DB 객체

# ORM을 활용한 데이터베이스 사용하기!
#  - 저장!
def create_message(msg : NoticeDTO, db: Session):
    db_msg = NoticeORM(
        title=msg.title,
        content=msg.content,
        user_id=msg.user_id,
        dates=datetime.now()
    )
    db.add(db_msg)
    db.commit()
