from flask import Flask
from flask import render_template
from flask import request
from iot-tweet-search-engine import user, query_lucene
app = Flask(__name__)



@app.route('/')
@app.route('/index')
def index():
	query = request.form
	u1 = user.User(id=1, localisation='us', emotion='neutral', gender='andy')
	if query is None:
		results = None
	else:
		q = query_lucene.QueryLucene()
		results = q.get_results(nb_results=10)


	return render_template('index.html', user=user, posts=results)


if __name__ == '__main__':
	app.run()
