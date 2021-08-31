import numpy as np
from fa import *
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

class Tunning_FA():
    def __init__(self, df, target,parameters, n_folds = 3, n_population = 20, n_interactions = 100, alpha = 0.1, betamin = 0.2, gamma = 1):
        self.df = df.reset_index(drop=True)
        self.target = target
        self.n_folds = n_folds
        self.n_dimension = len(parameters)
        self.n_population = n_population
        self.n_interactions = n_interactions
        self.alpha = alpha
        self.betamin = betamin
        self.lower_boundary = []
        self.upper_boundary = []
        self.gamma = gamma
        self.time = 0
        self.fitness_function = self.load_initial_fitness_fun
        self.get_lower_boundary(parameters)
        self.get_upper_boundary(parameters)
    
    def get_lower_boundary(self, parameters):
        for _, value in parameters.items():
            self.lower_boundary.append(value[0])

    def get_upper_boundary(self, parameters):
        for _, value in parameters.items():
            self.upper_boundary.append(value[1])
    
    def load_initial_fitness_fun(self, D, fireFly):
        estimator = xgb.XGBRegressor(
         learning_rate = fireFly[0],
         max_depth= int(fireFly[1]),
         min_child_weight= fireFly[2],
         gamma= fireFly[3],
         colsample_bytree= fireFly[4],
         subsample = fireFly[5],
         reg_alpha = fireFly[6],
         reg_lambda = fireFly[7],
         n_estimators = int(fireFly[8]),
         seed= 248)
        n_fold = self.n_folds
        tscv = TimeSeriesSplit(n_splits=n_fold)
        mape = 0
        for train_index, test_index in tscv.split(self.df):
            new_x_train = self.df.loc[train_index].drop(columns=[self.target])
            new_y_train = self.df.loc[train_index][self.target]

            new_x_test = self.df.loc[test_index].drop(columns=[self.target])
            new_y_test = self.df.loc[test_index][self.target]

            estimator.fit(new_x_train, new_y_train, eval_metric='rmse')
            
            y_predict = estimator.predict(new_x_test)
            MAPE_test = np.mean(np.abs((new_y_test.values - y_predict)/new_y_test.values))
            mape += MAPE_test
        return mape/n_fold
        
    
    def Update_Parameters(self, new_param):
        for key, _ in new_param.items():
            self.parameters[key] = new_param[key]
    
    def Reset_Parameters(self):
        self.parameters = {}

    def Run(self):
        start = time.time()
        alg = NewFireflyAlgorithm(self.n_dimension, self.n_population, self.n_interactions, self.alpha , self.betamin , self.gamma , self.lower_boundary, self.upper_boundary, self.fitness_function)
        _, best_fireflies, best_index = alg.Run()
        end = time.time()
        self.time = end - start
        best_firefly = best_fireflies[best_index[0]]
        self.best_firefly = {
            "learning_rate" : best_firefly[0],
            "max_depth" : int(best_firefly[1]),
            "min_child_weight" : best_firefly[2],
            "gamma": best_firefly[3],
            "colsample_bytree": best_firefly[4],
            "subsample" : best_firefly[5],
            "reg_alpha" : best_firefly[6],
            "reg_lambda" : best_firefly[7],
            "n_estimators" : int(best_firefly[8])
        }
        return self.best_firefly
        