Titanic.py trains and tunes an XGBoost model on the Titanic dataset, then prints accuracy, a confusion matrix, a detailed classification report, and pops up a bar-chart of the 20 most important features. Put train.csv, test.csv, and gender_submission.csv in the same folder as the script (or edit the three path constants at the top if they’re elsewhere). Install the dependencies with
pip install pandas scikit-learn xgboost matplotlib
and run the script with
python Titanic.py.
You’ll see the randomized hyper-parameter search progress, the best cross-validated score and parameters, followed by the test-set metrics and the feature-importance plot. If you want a Kaggle-ready predictions file, just uncomment the submission.to_csv(...) line near the end; the script will then write submission.csv in the same folder.
