from dateutil import parser

from app import db
from app.models.tweet import Tweet

if __name__ == '__main__':
	f = open('corpus/iot-tweets-vector-10.tsv', 'r')
	f.readline()
	for line in f:
		line = line[:-1]  # remove \n
		parts = line.split('\t')

		t = Tweet(
			id=int(parts[0]),
			sentiment=parts[1],
			topic_id=(None if parts[2] == 'None' else int(parts[2])),
			country=parts[3],
			gender=parts[4],
			urls=parts[5],
			text=parts[6],
			user_id=int(parts[7]),
			user_name=parts[8],
			date=parser.parse(parts[9]),
			hashtags=parts[10],
			indication=parts[11],
			cleaned_text=parts[12],
			vector=[float(x) for x in parts[13][1:-1].split(', ')]
		)
		db.session.add(t)
	f.close()
	db.session.commit()
