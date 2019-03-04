from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models.tweet import Tweet


class DB:
	ENGINE_ADDR = 'postgresql+psycopg2://postgres:password@localhost:5432/iot_tweet'  # 'postgresql+psycopg2://postgres:password@/iot_tweet?host=/cloudsql/iot-tweet:europe-west3:main-instance'
	db = None

	@staticmethod
	def get_instance():
		if DB.db is None:
			print('new db')
			DB.db = DB._load_db_tweets()
		return DB.db

	@staticmethod
	def _load_db_tweets():
		engine = create_engine(DB.ENGINE_ADDR, echo=True)
		Session = sessionmaker(bind=engine)
		return Session()


if __name__ == '__main__':
	print([value for value, in DB.get_instance().query(Tweet.id).limit(20).all()])
	print(DB.get_instance().query(Tweet.id).limit(20).all())
