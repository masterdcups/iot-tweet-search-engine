import random

from sqlalchemy import Column, Text, Integer, DateTime, ARRAY, Float, func
from sqlalchemy.ext.declarative import declarative_base

from db import DB
from models.tweet import Tweet

Base = declarative_base()


class RepresentativeCorpus(Base):
	__tablename__ = 'representative_corpus'

	id = Column(Integer, primary_key=True)
	date_created = Column(DateTime, default=func.current_timestamp())
	date_modified = Column(DateTime, default=func.current_timestamp(),
	                       onupdate=func.current_timestamp())
	sentiment = Column(Text, nullable=True, unique=False)
	topic_id = Column(Integer, nullable=True, unique=False)
	country = Column(Text, nullable=True, unique=False)
	gender = Column(Text, nullable=True, unique=False)
	urls = Column(Text, nullable=True, unique=False)
	text = Column(Text, nullable=True, unique=False)
	user_id = Column(Integer, nullable=True, unique=False)
	user_name = Column(Text, nullable=True, unique=False)
	date = Column(DateTime, nullable=True, unique=False)
	hashtags = Column(Text, nullable=True, unique=False)
	indication = Column(Text, nullable=True, unique=False)
	cleaned_text = Column(Text, nullable=True, unique=False)
	vector = Column(ARRAY(Float), nullable=True, unique=False)

	@staticmethod
	def create_representative_corpus():
		query2 = DB.get_instance().query(Tweet.id).filter(
			"topic_id is not null and cleaned_text != '' and user_id is not null")

		id_list = [t[0] for t in query2.all()]

		random.shuffle(id_list)

		id_list = id_list[:100000]


		query = DB.get_instance().query(Tweet.id, Tweet.date_created, Tweet.date_modified, Tweet.sentiment,
			                                Tweet.topic_id, Tweet.country, Tweet.gender, Tweet.urls, Tweet.text,
			                                Tweet.user_id, Tweet.user_name,
			                                Tweet.date, Tweet.hashtags, Tweet.indication, Tweet.cleaned_text,
			                                Tweet.vector).filter(Tweet.id.in_(id_list))

		i = 0
		for t in query.all():
			i += 1
			print(i)

			rc = RepresentativeCorpus()
			rc.id = t[0]
			rc.date_created = t[1]
			rc.date_modified = t[2]
			rc.sentiment = t[3]
			rc.topic_id = t[4]
			rc.country = t[5]
			rc.gender = t[6]
			rc.urls = t[7]
			rc.text = t[8]
			rc.user_id = t[9]
			rc.user_name = t[10]
			rc.date_modified = t[11]
			rc.hashtags = t[12]
			rc.indication = t[13]
			rc.cleaned_text = t[14]
			rc.vector = t[15]

			DB.get_instance().add(rc)
			DB.get_instance().commit()

		insert into representative_corpus values (select * from tweet where tweet.id in ());





if __name__ == '__main__':
	truc = RepresentativeCorpus()
	truc.create_representative_corpus()
