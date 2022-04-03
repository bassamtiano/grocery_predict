from locale import normalize
import sys

import math

from model.trainer import GroceryDataModule, GroceryTrainer
import torch
import numpy as np
import pandas as pd

from utils.preprocessor import Preprocessor
from utils.preprocessor_online import PreprocessorOnline

from sklearn.metrics import mean_squared_error

# from utils.trainer import ManualTrainer
# from utils.grocery_trainer import GroceryTrainer, GroceryDataModule

import pytorch_lightning as pl

if __name__ == "__main__":
    
    ppro = PreprocessorOnline(batch_size = 2)
    x_train, y_train, x_test, y_test, scaler, train_set_scaled, test_set_scaled, data_sales = ppro.preprocessor()

    device = torch.device("cpu")

    model = GroceryTrainer()
    dm = GroceryDataModule(
        batch_size = 50
    ) 

    trainer = pl.Trainer(gpus = 1, precision = 16, max_epochs = 1)
    trainer.fit(model, datamodule = dm)

    y_pred = trainer.predict(model = model, dataloaders = dm)[0]
    y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])
    y_pred = y_pred.cpu()

    pred_test_set = []

    for index in range(0,len(y_pred)):
        pred_test_set.append(np.concatenate([y_pred[index],x_test[index]],axis=1))
    
    
    pred_test_set = np.array(pred_test_set)
    pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])

    pred_test_set_inverted = scaler.inverse_transform(pred_test_set)

    result_list = []
    sales_dates = list(data_sales[-21:].date)
    act_sales = list(data_sales[-21:].sales)
    for index in range(0,len(pred_test_set_inverted)):
        result_dict = {}
        result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + act_sales[index])
        result_dict['date'] = sales_dates[index+1]
        result_list.append(result_dict)
    data_result = pd.DataFrame(result_list)

    data_sales_pred = pd.merge(data_sales,data_result,on='date',how='left')

    data_accuracy = data_sales_pred.dropna()
    
    y_pred = data_accuracy["pred_value"].tolist()
    y_true = data_accuracy["sales"].tolist()

    # accuracy_score = mean_squared_error(y_true, y_pred, squared=False)
    # print(accuracy_score)


    # print(y_pred)
    # print(len(y_pred))
    
    # print(y_true)
    # print(len(y_true))

    data_accuracy["difference"] = abs(data_accuracy["sales"]) - abs(data_accuracy["pred_value"]) if data_accuracy["sales"] > data_accuracy["pred_value"] else abs(data_accuracy["pred_value"]) - abs(data_accuracy["sales"])
    # data_accuracy["loss"] = (abs(abs(data_accuracy["sales"]) - abs(data_accuracy["pred_value"])) if abs(data_accuracy["sales"]) > abs(data_accuracy["pred_value"]) else abs(data_accuracy["pred_value"]) - abs(data_accuracy["sales"])) / data_accuracy["sales"]

    print(data_accuracy)

    
