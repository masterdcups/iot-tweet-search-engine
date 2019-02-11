from dateutil import parser
from sqlalchemy import func

from app import db
from app.models.tweet import Tweet

if __name__ == '__main__':
	f = open('corpus/iot-tweets-vector-v3.tsv', 'r')
	f.readline()
	i = 0

	max = db.session.query(func.max(Tweet.id)).scalar()

	for line in f:
		line = line[:-1]  # remove \n
		parts = line.split('\t')

		if int(parts[0]) <= max:
			continue

		t = Tweet(
			id=int(parts[0]),
			sentiment=parts[1],
			topic_id=(None if parts[2] == 'None' else int(parts[2])),
			country=parts[3],
			gender=parts[4],
			urls=parts[5],
			text=parts[6],
			user_id=(int(parts[7]) if parts[7] != '' else None),
			user_name=parts[8],
			date=(parser.parse(parts[9]) if parts[9] != '' else None),
			hashtags=parts[10],
			indication=parts[11],
			cleaned_text=parts[12],
			vector=[float(x) for x in parts[13][1:-1].split(', ')]
		)
		db.session.add(t)

		if i % 1000 == 0:
			db.session.commit()
	f.close()
	db.session.commit()
