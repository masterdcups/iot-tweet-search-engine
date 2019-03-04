import numpy as np
from sqlalchemy import Column, Text, Integer, DateTime, ARRAY, Float, func
from sqlalchemy.ext.declarative import declarative_base

from db import DB

Base = declarative_base()


class User(Base):
	__tablename__ = 'users'

	id = Column(Integer, primary_key=True)
	date_created = Column(DateTime, default=func.current_timestamp())
	date_modified = Column(DateTime, default=func.current_timestamp(),
	                       onupdate=func.current_timestamp())
	vector = Column(ARRAY(Float), nullable=True, unique=False)
	nb_click = Column(Integer, nullable=True, unique=False)
	localisation = Column(Text, nullable=True, unique=False)
	gender = Column(Text, nullable=True, unique=False)
	emotion = Column(Text, nullable=True, unique=False)
	topic_vector = Column(ARRAY(Float), nullable=True, unique=False)

	@staticmethod
	def load(user_id):
		u = DB.get_instance().query(User).filter_by(id=user_id).first()
		if u is None:
			u = User()
			u.nb_click = 0
			u.vector = np.zeros(300)
			DB.get_instance().add(u)
		return u
