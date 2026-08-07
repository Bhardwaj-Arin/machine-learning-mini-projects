train-customer:
	cd customer_churn && python src/train.py

train-spam:
	cd spam_sms_detection && python src/train.py

train-movie:
	cd movie_genre_classification && python src/train.py

app:
	streamlit run app/streamlit_app.py

clean:
	python -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').glob('**/__pycache__') if p.is_dir()]"
