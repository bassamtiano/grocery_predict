import os, sys

import pickle
import pandas as pd

import statsmodels.formula.api as smf
from sklearn.preprocessing import MinMaxScaler

import torch

class PreprocessorOnline ():
    def __init__(self,
                batch_size,
                dataset_dir = "datasets/OnlineRetail.csv") -> None:
        if os.path.exists(dataset_dir) :
            self.data = pd.read_csv(dataset_dir, encoding= 'unicode_escape')
        
    def preprocessor(self):
        g_data = self.data.dropna(how = "all")

        g_data = g_data[g_data["InvoiceDate"].str[0:2] != 'Or']

        g_data["Quantity"] = pd.to_numeric(g_data["Quantity"])
        g_data["UnitEach"] = pd.to_numeric(g_data["UnitPrice"])


        g_data["Date"] = pd.to_datetime(
                            g_data["InvoiceDate"].apply(lambda x: x.split(" ")[0])
                        )
        g_data["Month"] = pd.to_datetime(
                            g_data["InvoiceDate"].apply(lambda x: x.split(" ")[0])
                        ).dt.month

        s_data = g_data.groupby("Date")["UnitPrice"].sum().reset_index()

        s_data = s_data.rename(columns = {"Date": "date", "UnitPrice": "sales"})



        diff_data = s_data.copy()
        diff_data["prev_sales"] = diff_data["sales"].shift(1)
        diff_data = diff_data.dropna()

        diff_data["diff"] = (diff_data["sales"] - diff_data["prev_sales"])

        sup_data = diff_data.drop(["prev_sales"], axis = 1)


        for inc in range(1, 13):
            field_name = "lag_" + str(inc)
            sup_data[field_name] = sup_data["diff"].shift(inc)

        sup_data = sup_data.dropna().reset_index(drop = True)
        
        model = smf.ols(formula = 'diff ~ lag_1 + lag_2 + lag_3 + lag_4 + lag_5 + lag_6 + lag_7 + lag_8 + lag_9 + lag_10 + lag_11 + lag_12', data = sup_data)
        model_fit = model.fit()

        regression_adj_rsq = model_fit.rsquared_adj

        data_model = sup_data.drop(["sales", "date"], axis = 1)

        train_set = data_model[0: -20].values
        test_set = data_model[-20:].values

        scaler = MinMaxScaler(feature_range = (-1, 1))

        scaler = scaler.fit(train_set)

        train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
        train_set_scaled = scaler.transform(train_set)

        test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
        test_set_scaled = scaler.transform(test_set)

        x_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]
        x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])

        x_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

        x_train, y_train = torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()
        x_test, y_test = torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float()

        return x_train, y_train, x_test, y_test, scaler, train_set_scaled, test_set_scaled, s_data


    