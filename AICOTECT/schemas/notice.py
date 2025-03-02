from pydantic import BaseModel

# requset로 전달 받은 데이터의 type을 정의
# + 유효성 체크
class NoticeDTO(BaseModel):
    title: str
    content: str
    user_id:str
