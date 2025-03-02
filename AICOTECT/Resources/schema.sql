# DDL(CREATE, ALTER, DROP) 테이블을 정의하는 SQL

CREATE TABLE `notice` (
	`notic_num` INT(10) NOT NULL AUTO_INCREMENT,
	`title` VARCHAR(100) NOT NULL COLLATE 'utf8mb4_general_ci',
	`content` VARCHAR(500) NULL DEFAULT NULL COLLATE 'utf8mb4_general_ci',
	`user_id` VARCHAR(100) NULL DEFAULT NULL COLLATE 'utf8mb4_general_ci',
	`dates` VARCHAR(50) NOT NULL DEFAULT '' COLLATE 'utf8mb4_general_ci',
	PRIMARY KEY (`notic_num`) USING BTREE
)
COMMENT='공지사항'
COLLATE='utf8mb4_general_ci'
ENGINE=InnoDB
AUTO_INCREMENT=1
;

