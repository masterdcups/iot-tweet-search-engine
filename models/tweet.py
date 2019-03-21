from sqlalchemy import Column, Text, Integer, DateTime, ARRAY, Float, func
from sqlalchemy.orm import relationship

from db import DB


class Tweet(DB.get_base()):
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

	favs = relationship("Favorite", back_populates='tweet')

	def is_faved(self, user):
		return user in [fav.user for fav in self.favs]

	@staticmethod
	def load(tweet_id):
		return DB.get_instance().query(Tweet).filter_by(id=tweet_id).first()
