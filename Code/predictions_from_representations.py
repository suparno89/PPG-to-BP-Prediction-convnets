import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
from sklearn.feature_selection import RFE
from sklearn.dummy import DummyRegressor
import random
import lightgbm as lgb
import os



INPUT_FILE = '../intermediate_data/representations/cardioveg_biking_spectogram_resnet.pkl'
LEAVE_N_OUT_VALIDATIONS = 2
OUTPUT_FILE = '../results/representations/results_cardioveg_leave2out_final.csv'
IMAGE_REPRESENTATION = 'scalograms'
MODEL_FOR_REPRESENTATION = 'resnet'
DATASET = 'cardioveg-fullday'


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def format_df(df):
    #df['patientid'] = pd.to_numeric(df['patientid'])
    df['sbp'] = pd.to_numeric(df['sbp'])
    df['dbp'] = pd.to_numeric(df['dbp'])
    df.drop(df.loc[(df['sbp'] == 0)|(df['dbp'] == 0)].index, inplace = True)
    return df

def model_params():

    estimator_base = SVR(kernel="linear")
    selector = RFE(estimator_base, 40, step=10)

    estimators_lr = []
    estimators_lr.append(('standardize', StandardScaler()))
    estimators_lr.append(('lr',  ElasticNet(alpha=0.1, l1_ratio=0.5, random_state = 42)))
    pipeline_lr = Pipeline(estimators_lr)


    estimators_lr_rfe = []
    estimators_lr_rfe.append(('standardize', StandardScaler()))
    estimators_lr_rfe.append(('selector', selector))
    estimators_lr_rfe.append(('lr',  ElasticNet(alpha=0.1, l1_ratio=0.5, random_state = 42)))
    pipeline_lr_rfe = Pipeline(estimators_lr_rfe)

    estimators_gbm_rfe = []
    estimators_gbm_rfe.append(('standardize', StandardScaler()))
    estimators_gbm_rfe.append(('selector', selector))
    estimators_gbm_rfe.append(('gbm',  GradientBoostingRegressor(learning_rate=0.01, n_estimators=50, random_state = 42)))
    pipeline_gbm_rfe = Pipeline(estimators_gbm_rfe)

    estimators_rf_rfe = []
    estimators_rf_rfe.append(('standardize', StandardScaler()))
    estimators_rf_rfe.append(('selector', selector))
    estimators_rf_rfe.append(('rf',  RandomForestRegressor(n_estimators=20, max_depth = 10, random_state = 42, n_jobs= -1)))
    pipeline_rf_rfe = Pipeline(estimators_rf_rfe)

    estimators_rf = []
    estimators_rf.append(('standardize', StandardScaler()))
    estimators_rf.append(('rf',  RandomForestRegressor(n_estimators=20, max_depth = 10, random_state = 42, n_jobs= -1)))
    pipeline_rf = Pipeline(estimators_rf)

    estimators_lgbm_rfe = []
    estimators_lgbm_rfe.append(('standardize', StandardScaler()))
    estimators_lgbm_rfe.append(('selector', selector))
    estimators_lgbm_rfe.append(('lgbm',  lgb.LGBMRegressor(learning_rate=0.05, n_estimators=20, n_jobs=-1,random_state=42)))
    pipeline_lgbm_rfe = Pipeline(estimators_lgbm_rfe)   

    estimators_lgbm= []
    estimators_lgbm.append(('standardize', StandardScaler()))
    estimators_lgbm.append(('lgbm',  lgb.LGBMRegressor(learning_rate=0.05, n_estimators=20, n_jobs=-1,random_state=42)))
    pipeline_lgbm = Pipeline(estimators_lgbm)   



    dummy_mean = DummyRegressor(strategy='mean')

    dict_models = { "lr" : pipeline_lr, 
                   # "gbm": pipeline_gbm,
                   "lr_rfe": pipeline_lr_rfe,
                  # "rf_rfe": pipeline_rf_rfe,
                   "gbm_rfe": pipeline_gbm_rfe,
                   "lgbm_rfe" : pipeline_lgbm_rfe, 
                   "dummy_mean": dummy_mean,
                   "lgbm": pipeline_lgbm,
                   "rf": pipeline_rf}
    return dict_models


def run_models(dict_models, df):

    results = []

    for predicted_variable in ['sbp', 'dbp']:
        print("running models for: " + str(predicted_variable))
        for key, value in dict_models.items():
            print("running model: "+ str(key))
            i = 0
            patient_ids = np.unique(df['patientid'])
            rmse = []
            r2 = []
            mape = []
            mae = []

            while len(patient_ids) > 1:
                i= i + 1
                random.seed(42)
                patient_test_ids = random.choices(patient_ids, k = LEAVE_N_OUT_VALIDATIONS)
                patient_ids = [e for e in patient_ids if e not in patient_test_ids]
                df_test = df.loc[df['patientid'].isin(patient_test_ids)].dropna()
                df_train = df[~df['patientid'].isin(patient_test_ids)].dropna()
                print("running fold" + str(i) + "with train/test datasize" + str(df_train.shape) + '/' + str(df_test.shape))
                
                value.fit(X = np.stack(df_train["representation"]), y = df_train[predicted_variable].values)
                predicted_labels = value.predict(np.stack(df_test["representation"]))
                rmse.append(np.sqrt(mean_squared_error(df_test[predicted_variable], predicted_labels)))  
                r2.append(r2_score(df_test[predicted_variable], predicted_labels))
                mape.append(mean_absolute_percentage_error(df_test[predicted_variable], predicted_labels))
                mae.append(mean_absolute_error(df_test[predicted_variable], predicted_labels))
            
            dict_results = {}
            dict_results['dataset'] = DATASET
            dict_results['image_type'] = IMAGE_REPRESENTATION
            dict_results['representation_net'] = MODEL_FOR_REPRESENTATION
            dict_results['predicted_variable'] = predicted_variable
            dict_results['name'] = key 
            dict_results['rmse_mean'] = np.mean(np.array(rmse))
            dict_results['rmse_sd'] = np.std(np.array(rmse))
            dict_results['r2_mean'] = np.mean(np.array(r2))
            dict_results['r2_sd'] = np.std(np.array(r2))
            dict_results['mape_mean'] = np.mean(np.array(mape))
            dict_results['mape_sd'] = np.std(np.array(mape))
            dict_results['mae_mean'] = np.mean(np.array(mae))
            dict_results['mae_sd'] = np.std(np.array(mae))

            results.append(dict_results)
            
    return results

def write_file(df_results, filepath):
    if not os.path.isfile(filepath):
        print("output file doesn't exist, creating a new one...")
        df_results.to_csv(filepath, header=True, index = False)
    else: 
        print("output file exists, appending to the existing file...")
        df_results.to_csv(filepath, mode='a', header=False, index = False)


if __name__ == "__main__":
    df = pd.read_pickle(INPUT_FILE)
    df = format_df(df)
    dict_models = model_params()
    results = run_models(dict_models, df)
    df_results = pd.DataFrame(results)
    print(df_results.head()) 
    #write_file(df_results, OUTPUT_FILE)
    #df_results.to_excel('../results/representations/results_cardioveg.xlsx', index = False) 
    #df_results.to_csv(OUTPUT_FILE, index = False, header=True)