.PHONY: install run test docker
install:
	pip install -r requirements.txt
run:
	python main.py
test:
	pytest -q
docker:
	docker build -t nba-game-predictor . && docker run --rm nba-game-predictor
