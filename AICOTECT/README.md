
## 프로젝트 "MYSITE"
### 1. 프로젝트 개요
- Python기반 웹 사이트 개발(mysite)
- 프론트엔드(HTML, CSS, JS) + Jinja2
- 벡엔드(fastapi, pydantic, sqlalchemy) + uvicorn
- 데이터베이스(pymysql) + MariaDB
- API(requests)

### 2. 프로젝트 기능
- mysite => 개인 소개 페이지
- 챗봇 => mysite에서 내 정보를 챗봇에게 QnA 할 수 있도록
- 카카오톡 연결 => mysite에서 "나에게 메세지 보내기"

### 3. 배포(deployment)
- 도커를 사용해서 컨테이너화 2개(mysite + db)
- Amazon Web Services(AWS): 클라우드 서비스를 통해서 배포

## 웹 개발 소개
### 1. 웹 용어 소개
- 클라이언트(웹 브라우저): 사용자(ME)
- 서버(서비스를 제공하는 컴퓨터): 제공업체(ex: NAVER)
- DNS(Domain Name Server): 도메인을 IP로 변경!
- IP : 컴퓨터는 접속할 수 있는 주소 (ex: 211.230.13.56)
- 네이버 서버 IP(ex: 201.132.56.78) => 접근하기 어려움
- DNS(www.naver.com == 201.132.56.78)

### 2. 웹 동작 과정(네이버 메인 페이지 접속)
1. 클라이언트가 요청(request) -> "http://www.naver.com"
2. DNS 서버 매칭 -> "http://www.naver.com" -> IP(201.132.56.78)
3. IP로 요청(네이버 서버)로 요청(request)
4. 네이버 서버 웹 API 동작
5. 네이버 서버 전달(request) -> 메인 페이지 접속을 위한 코드와 파일들
6. 클라이언트는 response를 다운로드
7. 클라이언트가 다운 받은 코드와 파일을 랜더링!

### 3. python 웹 개발환경 소개
- python: 프로그래밍 언어-> 단독으로 웹개발 불가능
- python기반 웹 프레임워크(Django, Flask, FastAPI)
- 웹 프레임워크(프론트 + 백 + DB)
- 프론트엔드: Jinja2 템플릿 엔진
- 백엔드: FastAPI + Uvicorn(WAS:웹 사이트가 동작할 때 필요한 웹 서버)
- DB: pymaSQL, sqlalchemy(ORM)
* 회사에서 웹 외에 API를 개발할 경우가 있는데 그때 FastAPI가 굿임

### 4. URL 소개
- http://127.0.0.1:8000/kakao?id="hj"&num="4"
- http -> 프로토콜(웹)
- 127.0.0.1 -> IP주소
- 8000 -> port
- /kakao -> 서비스 경로
- ?id="hj"&num="4" -> 쿼리스트링(Url을 통해서 data 전달)


### 5. http 메서드
- http 프로토콜로 request를 보낼 경우 반드시 메서드 사용
- get(default): 데이터가 오픈(개인정보 관련 없는 경우: 검색, 게시글 띄워주기, 카페글)
- post: 데이터를 숨겨야 하는 경우(login, 회원 관련 서비스)

## Function: 카카오 API를 사용한 나에게 메시지 보내기
### 1. 순서
1. Kakao Developer 사이트 접속 권한 허용
2. Kakao API 사용해 인증코드 발급(1회용)
3. "인증코드"를 사용해 토큰 받기
4. "Access Torken"을 사용해서 나에게 보내기
5. + 1달에 한 번 "Refresh Torken" 재발급


### 2. 용어 소개(자동화 시켜야 함)
- 인증코드: Access와 Refresh Token을 발급받을 때 사용
- Accsess Torken: 카카오 API 사용할 수 있는 토큰
- Refresh Torken: Access Token을 재발급 받을 때 사용 
- 생명주기: 인증코드(1회), Access(6시간), Refresh(2달)
- Refresh Token 2달이 생명주기, 1달 후부터 재발급 가능
- 