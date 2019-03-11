from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


class DB:
	ENGINE_ADDR = 'postgresql+psycopg2://postgres:password@localhost:5432/iot_tweet'  # 'postgresql+psycopg2://postgres:password@/iot_tweet?host=/cloudsql/iot-tweet:europe-west3:main-instance'
	db = None
	base = None

	@staticmethod
	def get_instance():
		if DB.db is None:
			print('new db')
			DB.db = DB._load_db()
		return DB.db

	@staticmethod
	def get_base():
		if DB.base is None:
			DB.base = declarative_base()
		return DB.base

	@staticmethod
	def _load_db():
		engine = create_engine(DB.ENGINE_ADDR, echo=True)
		Session = sessionmaker(bind=engine)
		return Session()
