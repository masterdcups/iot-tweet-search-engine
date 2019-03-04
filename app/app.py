import os

from flask import Flask
from flask import render_template
from flask import request

from definitions import ROOT_DIR
from query_lucene import QueryLucene
from query_lucene_manager import QueryLuceneManager
from user import User
import lucene

app = Flask(__name__)

@app.route('/')
def index():
	query = request.args.get('query')
	u1 = User() # main user
	if query is None:
		results = []
	else:
		ql = QueryLuceneManager.get_instance()
		ql.query_parser_must([query])
		results = ql.get_results(nb_results=10)

	return render_template('index.html', user=u1, results=results)


if __name__ == '__main__':
	app.run()
