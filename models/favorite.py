from sqlalchemy import Column, Integer, ForeignKey, BigInteger
from sqlalchemy.orm import relationship

from db import DB


class Favorite(DB.get_base()):
	__tablename__ = 'favorites'

	user_id = Column(Integer, ForeignKey('users.id'), primary_key=True)  #
	tweet_id = Column(BigInteger, ForeignKey('tweets.id'),
					  primary_key=True)  # ForeignKey('tweets_all.id'),  # todo change with tweets

	user = relationship("User", back_populates="favs")
	tweet = relationship("Tweet", back_populates='favs')
