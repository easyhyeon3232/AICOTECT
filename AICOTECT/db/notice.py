from common import connection

def add_review(data):
    # 1.Connection
    conn = connection()

    try:
        curs = conn.cursor()
        sql = f"""
                CREATE TABLE `notic` (
	`notic_num` INT(10) NOT NULL AUTO_INCREMENT,
	`title` VARCHAR(100) NOT NULL COLLATE 'utf8mb4_general_ci',
	`content` VARCHAR(500) NOT NULL COLLATE 'utf8mb4_general_ci',
	`user_id` VARCHAR(100) NULL DEFAULT NULL COLLATE 'utf8mb4_general_ci',
	`dates` VARCHAR(50) NOT NULL DEFAULT '' COLLATE 'utf8mb4_general_ci',
	PRIMARY KEY (`notic_num`) USING BTREE
)
COMMENT='공지사항'
COLLATE='utf8mb4_general_ci'
ENGINE=InnoDB
AUTO_INCREMENT=1
;
              """
        curs.execute(sql, data)
    except Exception as e:
        print(e)
    finally:
        conn.close()