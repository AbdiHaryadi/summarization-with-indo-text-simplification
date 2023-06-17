config := default
valconfig := validation
out := result.txt 

train: src/train.py
	python3 src/train.py $(config) > log/$(config)_train.txt

test:src/validate.py
	python3 src/validate.py $(config) test

predict:src/predict.py
	python3 src/predict.py $(config)

run-test:
	python3 -m unittest discover src/test
	
fetch-data: src/data/fetch_data.sh
	chmod +x ./src/data/fetch_data.sh
	./src/data/fetch_data.sh
