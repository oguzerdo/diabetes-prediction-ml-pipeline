# GITHUB

commit:
	git commit -am "commit from make file"

push:
	git push origin main

pull:
	git pull origin main

fetch:
	git fetch origin main

reset:
	rm -f .git/index
	git reset

req:
	pip freeze > requirements.txt

compush: commit push

run:
	python main.py --no-debug

debug:
	python main.py

test:
	python main.py --no-tuning

req:
	pip list --format=freeze > requirements.txt

install:
	pip install -r requirements.txt


scores:
	python scripts/model_history.py

