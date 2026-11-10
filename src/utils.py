import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score as r2, mean_squared_error as mse
import seaborn as sns

def evaluate_preds(true_values, pred_values, save=False):
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
    ax.plot([min(pred_values), max(pred_values)], [min(true_values), max(true_values)], linestyle='--', color='black')  # ���������, ��� true_values = pred_values
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
