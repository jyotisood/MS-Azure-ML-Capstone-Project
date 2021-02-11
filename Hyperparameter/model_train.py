from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import argparse
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
import joblib
from sklearn.model_selection import train_test_split
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

run = Run.get_context()

def main():
    
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00451/dataR2.csv'
    data = TabularDatasetFactory.from_delimited_files(url)
    x = data.to_pandas_dataframe()
    y = x.pop("Classification")    
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.33, random_state=200)

    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
    parser.add_argument('--solver', type=str, default='lbfgs', help="chose the algorithm to train the model")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    run.log("Algorithm: ", args.solver)

    model = LogisticRegression(solver=args.solver, C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    pred_prob= model.predict_proba(x_test)
    AUC =roc_auc_score(y_test, pred_prob[:,1])
    run.log("AUC", np.float(AUC))

    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model,'outputs/model.joblib')

if __name__ == '__main__':
    main()

