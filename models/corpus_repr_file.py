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

		fichier = open("id_data.txt", "a")

		for i in range(len(id_list)):
			fichier.write(str(id_list[i]) + ", ")
		fichier.close()


if __name__ == '__main__':
	truc = RepresentativeCorpus()
	truc.create_representative_corpus()
