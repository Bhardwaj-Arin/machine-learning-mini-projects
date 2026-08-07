train-customer:
	cd customer_churn && python src/train.py

train-spam:
	cd spam_sms_detection && python src/train.py

train-movie:
	cd movie_genre_classification && python src/train.py

<<<<<<< HEAD
app:
	streamlit run app/streamlit_app.py
=======
app-customer:
	cd customer_churn && streamlit run app/streamlit_app.py

app-spam:
	cd spam_sms_detection && streamlit run app/streamlit_app.py

app-movie:
	cd movie_genre_classification && streamlit run app/streamlit_app.py
>>>>>>> d4efc887168de391e4c3fa141bfad0bf2d2cbdc6

clean:
	python -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').glob('**/__pycache__') if p.is_dir()]"
