from flask import Flask
from flask import render_template
from flask_scss import Scss

from user import User

app = Flask(__name__)
Scss(app)


@app.route('/')
def index():
	user = User(1)  # main user
	user.load()

	results = [
		{
			'Text': 'First sign of twitter as transport for ""internet of things"" at http://twitter.com/whatsshakin',
			'User_Name': '@clementast'
		},
		{
			'Text': 'The Internet of Things. L\'internet des objets. http://tinyurl.com/5qr2nq',
			'User_Name': '@clementast'
		},
		{
			'Text': 'programming mirroir and nabaztag... Internet of things!',
			'User_Name': '@clementast'
		},
		{
			'Text': 'aha mycrocosm integration & internet of things working via Twitter API - nice!',
			'User_Name': '@clementast'
		}
	]

	return render_template('index.html', user=user, results=results, query='')


if __name__ == '__main__':
	app.run()
