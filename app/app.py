import numpy as np
from flask import Flask, render_template, session, request, redirect, url_for
from passlib.handlers.pbkdf2 import pbkdf2_sha256

from db import DB
from models.favorite import Favorite
from models.tweet import Tweet
from models.user import User
from query_lucene_manager import QueryLuceneManager
from recommendation.basic_reco import BasicReco

app = Flask(__name__)

app.session_type = 'filesystem'
app.secret_key = b'L\xac\xc9\xf8\xb5\xd95\x86}\xe0V\x89\x0fN\xc9#\x13qZ-\x8e\xb8R\xef'


@app.route('/')
def index():
	query = request.args.get('query')
	if session.get('username') is not None:
		user = User.load(user_name=session.get('username'))
	else:
		user = None

	reco_tweets = []

	if query is None or query == '':
		results = []
		query = ''
	else:
		QueryLuceneManager.get_instance().query_parser_must([query])
		results = QueryLuceneManager.get_instance().get_results(nb_results=50)

		if user is not None:
			results = QueryLuceneManager.get_instance().rerank_results(results, user.vector, user.gender,
			                                                           user.localisation, user.emotion)

			# recommendation
			reco_sys = BasicReco()
			reco_tweets = reco_sys.recommended_tweets(user)
		else:
			results = QueryLuceneManager.get_instance().link_tweets(results)

	return render_template('index.html', user=user, results=results, query=query, reco_tweets=reco_tweets)


@app.route('/login', methods=['GET'])
def login():
	return render_template('users/login.html')


@app.route('/login', methods=['POST'])
def login_user():
	users_corresponding = DB.get_instance().query(User).filter_by(username=request.form.get('username'))

	if users_corresponding.count() == 0:
		return 'WRONG USERNAME'

	user = users_corresponding.first()
	if not pbkdf2_sha256.verify(request.form.get('password'), user.password):
		return 'WRONG PASSWORD'

	session['username'] = user.username
	session.modified = True

	return redirect(url_for('index'))


@app.route('/signup', methods=['GET'])
def signup():
	return render_template('users/signup.html')


@app.route('/logout', methods=['GET'])
def logout():
	session.pop('username')
	return redirect(url_for('index'))


@app.route('/signup', methods=['POST'])
def create_user():
	user = User(username=request.form.get('username'),
	            password=pbkdf2_sha256.encrypt(request.form.get('password'), rounds=200000, salt_size=16),
	            vector=np.zeros(300),
	            nb_click=0)

	DB.get_instance().add(user)
	DB.get_instance().commit()

	session['username'] = user.username
	session.modified = True

	return redirect(url_for('index'))


@app.route('/mark_view', methods=['POST'])
def mark_view():
	user = User.load(user_name=session.get('username'))
	tweet = Tweet.load(int(request.form.get('tweet_id')))
	state = ''

	if tweet.is_faved(user):
		state = 'removed'
		user.remove_favorite(tweet)
	else:
		view = Favorite(user_id=user.id, tweet_id=tweet.id)
		# user.update_profile(np.array(tweet.vector)) # todo
		DB.get_instance().add(view)
		state = 'added'

	DB.get_instance().commit()

	return state


if __name__ == '__main__':
	app.run()
