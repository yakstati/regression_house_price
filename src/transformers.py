from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd
import json
import optuna
from sklearn.model_selection import KFold
from pathlib import Path


class DataCleaner(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X:pd.DataFrame, y:pd.DataFrame=None):
        self.square_q1_ = X['Square'].quantile(0.01)
        self.square_q99_ = X['Square'].quantile(0.99)
        self.rooms_q1_ = X['Rooms'].quantile(0.01)
        self.rooms_q99_ = X['Rooms'].quantile(0.99)
        self.median_square_per_room_ = (X.loc[(X['Rooms'] > self.rooms_q1_) & (X['Rooms'] < self.rooms_q99_) & (X['Square'] > self.square_q1_) & (X['Square'] < self.square_q99_), 'Square'] / X.loc[(X['Rooms'] > self.rooms_q1_) & (X['Rooms'] < self.rooms_q99_) & (X['Square'] > self.square_q1_) & (X['Square'] < self.square_q99_), 'Rooms']).median()
        self.kitchensquare_median_ = X['KitchenSquare'].median()
        self.kitchensquare_q99_ = X['KitchenSquare'].quantile(0.99)

        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X.drop('Id', axis=1, inplace=True)
        square_less_q1_mask = X['Square'] < self.square_q1_
        square_q99_greater_mask = X['Square'] > self.square_q99_
        rooms_0_mask = (X['Rooms'] == 0)
        rooms_q1_greater_mask = X['Rooms'] > self.rooms_q1_
        rooms_q1_q99_outside_mask = (X['Rooms'] < self.rooms_q1_) | (X['Rooms'] > self.rooms_q99_) 
        square_ls_q1_rooms_0_mask = square_less_q1_mask & rooms_0_mask
        X.loc[square_ls_q1_rooms_0_mask, ['KitchenSquare', 'LifeSquare']] = [0, 0]

        rooms_gt_q1_mask = X['Rooms'] > self.rooms_q1_
        rooms_ls_q99_mask = X['Rooms'] < self.rooms_q99_
        rooms_q1_q99_inside_mask = rooms_gt_q1_mask & rooms_ls_q99_mask
        X.loc[(square_less_q1_mask) & (rooms_q1_q99_inside_mask), 'Square'] = X.loc[(square_less_q1_mask) & (rooms_q1_q99_inside_mask), 'Rooms'] * self.median_square_per_room_

        #indices_to_drop = X[(X['Rooms'] == 0) & (X['Square'] == 0) & (X['LifeSquare'] == 0) & (X['KitchenSquare'] == 0)].index
        #X.drop(index=indices_to_drop, inplace=True)

        X['Square_per_room'] = X['Square'] / X['Rooms']
        square_per_room_greater_median_mask = X['Square_per_room'] > (5 * self.median_square_per_room_)
        square_per_room_less_median_mask    = X['Square_per_room'] < (self.median_square_per_room_ / 5)
        upper_cap = X['Rooms'] * self.median_square_per_room_ * 5
        lower_cap = X['Rooms'] * self.median_square_per_room_ / 5

        mask_hi = square_q99_greater_mask & square_per_room_greater_median_mask & rooms_q1_greater_mask
        mask_lo = square_q99_greater_mask & square_per_room_less_median_mask    & rooms_q1_greater_mask

        X.loc[mask_hi, 'Square'] = np.minimum(X.loc[mask_hi, 'Square'], upper_cap.loc[mask_hi])
        X.loc[mask_lo, 'Square'] = np.maximum(X.loc[mask_lo, 'Square'], lower_cap.loc[mask_lo])
        X.drop('Square_per_room', axis=1, inplace=True)

        X.loc[rooms_q1_q99_outside_mask, 'Rooms'] = X.loc[rooms_q1_q99_outside_mask, 'Rooms'].clip(lower=1, upper=7)

        lifesquare_greater_square_mask = ((X['LifeSquare'] - X['Square']) < (X['Square'])) & ((X['LifeSquare'] - X['Square']) > 0)
        X.loc[lifesquare_greater_square_mask, ['LifeSquare', 'Square']] = X.loc[lifesquare_greater_square_mask, ['Square', 'LifeSquare']].values

        kitchensquare_0_rooms_gt_1_mask = (X['KitchenSquare'] == 0) & (X['Rooms'] > 1)
        X.loc[kitchensquare_0_rooms_gt_1_mask, 'KitchenSquare'] = self.kitchensquare_median_


        kitchensquare_greater_q99_mask = X['KitchenSquare'] > self.kitchensquare_q99_
        X.loc[kitchensquare_greater_q99_mask, 'KitchenSquare'] = self.kitchensquare_q99_

        housefloor_0_mask = (X['HouseFloor'] == 0)
        housefloor_gt_60_mask = (X['HouseFloor'] > 60)
        floor_gt_housefloor_mask = (X['Floor'] > X['HouseFloor'])
        housefloor_0_or_gt_60_mask = housefloor_0_mask | housefloor_gt_60_mask | floor_gt_housefloor_mask
        X.loc[housefloor_0_or_gt_60_mask, 'HouseFloor'] = X.loc[housefloor_0_or_gt_60_mask, 'Floor']

        houseyear_gt_2026 = X['HouseYear'] > 2026
        X.loc[houseyear_gt_2026, 'HouseYear'] = 2026

        X['Ecology_2_is_A'], X['Ecology_2_is_B'] = (X['Ecology_2'] == 'A') * 1, (X['Ecology_2'] == 'B') * 1
        X = X.drop('Ecology_2', axis=1)
            
        X['Ecology_3_is_A'], X['Ecology_3_is_B'] = (X['Ecology_3'] == 'A') * 1, (X['Ecology_3'] == 'B') * 1
        X = X.drop('Ecology_3', axis=1)
            
        X['Shops_2_is_A'], X['Shops_2_is_B'] = (X['Shops_2'] == 'A') * 1, (X['Shops_2'] == 'B') * 1
        X = X.drop('Shops_2', axis=1)

        return X


class LifesquareKNNImputer(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        X = X.copy()
        X = X[(X['LifeSquare'] < 200) & (X['LifeSquare'] > 10)]

        fit_mask = X['LifeSquare'].notna()
        self.cols_for_knn_ = [col for col in X.columns if col not in ['LifeSquare', 'Healthcare_1']]

        X_to_scale = X.loc[fit_mask, self.cols_for_knn_]
        target = X.loc[fit_mask, 'LifeSquare']

        self.scaler_ = RobustScaler()
        X_scaled = self.scaler_.fit_transform(X_to_scale)

        imputer = KNeighborsRegressor()

        project_root = next(
            parent for parent in Path(__file__).resolve().parents
            if (parent / 'configs').is_dir()
        )

        config_path = project_root / 'configs' / 'model_config.json'

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as config:
                configs = json.load(config)
        else:
            configs = []

        model_info = next((p for p in configs if p["model_name"] == "LifeSquareImputer"), None)
        self.best_params_ = model_info["params"] if model_info else None

        if self.best_params_ is None:
            param_grid = {
                'n_neighbors': optuna.distributions.IntDistribution(1, 30)
            }

            cv = KFold(n_splits=3, random_state=42, shuffle=True)

            optuna_search = optuna.integration.OptunaSearchCV(
                imputer,
                param_grid,
                cv=cv,
                n_trials=30,
                scoring='neg_root_mean_squared_error',
                random_state=42
            )

            optuna_search.fit(X_scaled, target)
            self.best_params_ = optuna_search.best_params_

            model_info = {
                "model_name": "LifeSquareImputer",
                "params": self.best_params_
            }

            configs.append(model_info)

            with open(config_path, "w", encoding="utf-8") as config:
                json.dump(configs, config, indent=4, ensure_ascii=False)

        self.lifesquare_knn_imputer_ = KNeighborsRegressor(**self.best_params_)
        self.lifesquare_knn_imputer_.fit(X_scaled, target)

        return self

    def transform(self, X):
        X = X.copy()

        miss_mask = X['LifeSquare'].isna() | (X['LifeSquare'] > 500)
        X_scaled = self.scaler_.transform(X.loc[miss_mask, self.cols_for_knn_])
        X.loc[miss_mask, 'LifeSquare'] = self.lifesquare_knn_imputer_.predict(X_scaled)

        return X

class HealthcareKNNImputer(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        fit_mask = X['Healthcare_1'].notna()
        self.cols_for_knn_ = list(set(X.columns.to_list()) - set(['Healthcare_1']))
        X_to_scale = X.loc[fit_mask, self.cols_for_knn_]
        target = X.loc[fit_mask, 'Healthcare_1']
        self.scaler_ = RobustScaler()
        X_scaled = self.scaler_.fit_transform(X_to_scale)
        imputer = KNeighborsRegressor()

        project_root = next(
                    parent for parent in Path(__file__).resolve().parents
                    if (parent / 'configs').is_dir()
                )
        config_path = project_root / 'configs' / 'model_config.json'

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as config:
                configs = json.load(config)
        else:
            configs = []

        model_info = next((p for p in configs if p["model_name"] == "HealthcareImputer"), None)
        self.best_params_ = model_info["params"] if model_info else None

        if self.best_params_ is None:
            param_grid = {
                'n_neighbors': optuna.distributions.IntDistribution(1, 30)
            }

            cv = KFold(n_splits=3, random_state=42, shuffle=True)

            optuna_search = optuna.integration.OptunaSearchCV(
                imputer,
                param_grid,
                cv=cv,
                n_trials=30,
                scoring='neg_root_mean_squared_error',
                random_state=42
            )

            optuna_search.fit(X_scaled, target)
            self.best_params_ = optuna_search.best_params_

            model_info = {
                "model_name": "HealthcareImputer",
                "params": self.best_params_
            }

            configs.append(model_info)

            with open(config_path, "w", encoding="utf-8") as config:
                json.dump(configs, config, indent=4, ensure_ascii=False)        
        self.healthcare_knn_imputer_ = KNeighborsRegressor(**self.best_params_)
        self.healthcare_knn_imputer_.fit(X_scaled, target)

        return self

    def transform(self, X):
        X = X.copy()

        miss_mask = X['Healthcare_1'].isna()
        X_scaled = self.scaler_.transform(X.loc[miss_mask,self.cols_for_knn_])
        X.loc[miss_mask, 'Healthcare_1'] = self.healthcare_knn_imputer_.predict(X_scaled)
        
        return X

class FeaturesCreator(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        temp = X[['DistrictId', 'Square']].copy()
        temp['Price'] = y.values   

        self.median_price_df_ = temp.groupby('DistrictId').apply(lambda g: g['Price'].median() / g['Square'].median()).rename('median_price_per_square').reset_index()

        self.median_all_district_ = temp['Price'].median() / temp['Square'].median()


        temp = X[['Rooms']].copy()
        temp['Price'] = y.values

        self.median_price_df2_ = temp.groupby('Rooms')['Price'].median().rename('median_price_per_room').reset_index()
        
        self.median_per_one_room_ = temp.loc[temp['Rooms'] == 1, 'Price'].median()



        return self

    def transform(self, X):
        X = X.copy()
        
        X['Is_First_Floor'] = (X['Floor'] == 1) * 1

        X['Is_Last_Floor'] = (X['Floor'] == X['HouseFloor']) * 1

        X['Lifesquare_Ratio'] = (X['LifeSquare'] / X['Square']).fillna(0)

        X = X.merge(self.median_price_df_, on='DistrictId', how='left')
        X['median_price_per_square'] = X['median_price_per_square'].fillna(self.median_all_district_)

        X = X.merge(self.median_price_df2_, on='Rooms', how='left')
        X['median_price_per_room'] = X['median_price_per_room'].fillna(X['Rooms'] * self.median_per_one_room_)

        X['Floor_HouseFloor_Ratio'] = X['Floor'] / X['HouseFloor']

        is_studio_mask = (X['Rooms'] == 1) & (X['LifeSquare'] > 0) & (X['KitchenSquare'] == 0)
        X['Is_Studio'] = is_studio_mask * 1

        return X


class SimpleDataCleaner(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.median_lifesquare_ = X['LifeSquare'].median()
        self.median_healthcare = X['Healthcare_1'].median()

        return self

    def transform(self, X):
        X.copy()
        X['LifeSquare'].fillna(self.median_lifesquare_, inplace=True)
        X['Healthcare_1'].fillna(self.median_healthcare, inplace=True)
        X.drop('Id', axis=1, inplace=True)
        X = pd.get_dummies(X, columns=['Ecology_2', 'Ecology_3', 'Shops_2'], drop_first=True)
        return X