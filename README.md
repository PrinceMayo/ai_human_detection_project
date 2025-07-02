#Project Documentation

This project is the creation of a Streamlit app that can process text blocks and determine if the text is either written as AI or Human based on a given dataset. The process of review is done by 6 different machine learning methods. Those being, SVM, Decision Tree, Ada Boost, CNN, LSTM and RNN. The app takes the given text and chosen machine learning method and responds with whether it believes it to be AI or Human written. Alongside this response is a brief description of the models and their estimated accuracy.

Streamlit Browser App- https://vigilant-space-capybara-w4x54wwpq9g2v9vw-8502.app.github.dev/ 


git clone https://github.com/PrinceMayo/ai_human_detection_project.git
cd ai_human_detection_project

Pip Dependencies
 streamlit, scikit-learn, seaborn, pandas, numpy, nltk, joblib, tensorflow, openpyxl

nltk.download - stopwords

Running the Program:
streamlit run app.py

If you wish to train using your own dataset:
place your dataset file in the training_data folder and change the file name and if necessary the read command for notebook code box 3 in Project Code

Model files will be saved with current dataset until overwritten via the Project Code code box 3

ProjectCode.ipynb needs to be run in a python 3.11.13 environment in order to work with tensorflow

