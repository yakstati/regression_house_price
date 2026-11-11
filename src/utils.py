import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score as r2, mean_squared_error as mse
import seaborn as sns
import pandas as pd

def evaluate_preds(true_values, pred_values, save: bool=False):
    true_values = np.ravel(true_values)
    pred_values = np.ravel(pred_values)

    fig, ax = plt.subplots()
    textstr = ("R2:  " + str(round(r2(true_values, pred_values), 3)) + "\n" +
          "RMSE:  " + str(round(np.sqrt(mse(true_values, pred_values)), 3)) + "\n" +
          "MSE:  " + str(round(mse(true_values, pred_values), 3)))
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)    
    sns.scatterplot(x=pred_values, y=true_values)
    ax.plot([min(pred_values), max(pred_values)], [min(true_values), max(true_values)], linestyle='--', color='black')
    plt.xlabel('Predicted values')
    plt.ylabel('True values')
    plt.title('True vs Predicted values')

    plt.show()

    return round(r2(true_values, pred_values), 3), round(np.sqrt(mse(true_values, pred_values)), 3)

def show_weights(model):
    importances = model.coef_
    feature_names = model.feature_names_in_

    for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"{name}: {imp:.4f}")

def save_res(model_name: str, r2_val: float, rmse_val: float, res_table: pd.DataFrame):
    res_table.loc[model_name, 'r2'] = r2_val
    res_table.loc[model_name, 'rmse'] = rmse_val

def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
#                 if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
#                     df[col] = df[col].astype(np.float16)
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
