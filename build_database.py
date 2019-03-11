import os

from dateutil import parser as date_parser
from sqlalchemy import Column, Text, Integer, DateTime, ARRAY, Float, func
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from definitions import ROOT_DIR
from parser import Parser

engine_addr = 'postgresql+psycopg2://postgres:password@/iot_tweet?host=35.198.185.194'

# 'postgresql+psycopg2://postgres:password@/iot_tweet?host=/cloudsql/iot-tweet:europe-west3:main-instance'
# 'postgresql+psycopg2://postgres:password@localhost:5432/iot_tweet'

engine = create_engine(engine_addr, echo=True)
Base = declarative_base()


class Tweet(Base):
	__tablename__ = 'tweets'

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


Session = sessionmaker(bind=engine)
session = Session()

corpus_path = os.path.join(ROOT_DIR, 'corpus/iot-tweets-2009-2016-completv3.tsv')

parser = Parser()
parser.load_w2v_model()
print('ok')

corpus = open(corpus_path, 'r', encoding='utf-8')

i = 0
corpus.readline()
for line in corpus:
	print(line)
	parts = line[:-1].split('\t')
	cleaned_tweet = parser.clean_tweet(parts[-6])
	urls = parts[5:-6]

	t = Tweet(
		id=int(parts[0]),
		sentiment=parts[1],
		topic_id=(None if parts[2] == 'None' else int(parts[2])),
		country=parts[3],
		gender=parts[4],
		urls=' '.join(urls),
		text=parts[-6],
		user_id=(int(parts[-5]) if parts[-5] != '' else None),
		user_name=parts[-4][1:-1],
		date=(date_parser.parse(parts[-3][1:-1]) if parts[-3][1:-1] != '' else None),
		hashtags=parts[-2],
		indication=parts[-1],
		cleaned_text=cleaned_tweet,
		vector=parser.tweet2vec(cleaned_tweet)
	)
	session.add(t)

	if i % 1000 == 0:
		print('writing', i)
		session.commit()

	i += 1

corpus.close()
session.commit()
