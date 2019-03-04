from app import db
from app.models.base import Base


class Tweet(Base):
	__tablename__ = 'tweets'

	sentiment = db.Column(db.Text, nullable=True, unique=False)
	topic_id = db.Column(db.Integer, nullable=True, unique=False)
	country = db.Column(db.Text, nullable=True, unique=False)
	gender = db.Column(db.Text, nullable=True, unique=False)
	urls = db.Column(db.Text, nullable=True, unique=False)
	text = db.Column(db.Text, nullable=True, unique=False)
	user_id = db.Column(db.Integer, nullable=True, unique=False)
	user_name = db.Column(db.Text, nullable=True, unique=False)
	date = db.Column(db.DateTime, nullable=True, unique=False)
	hashtags = db.Column(db.Text, nullable=True, unique=False)
	indication = db.Column(db.Text, nullable=True, unique=False)
	cleaned_text = db.Column(db.Text, nullable=True, unique=False)
	vector = db.Column(db.ARRAY(db.Float), nullable=True, unique=False)
