import os, sys
import pickle

from typing import Dict


from datetime import datetime , date

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader

class Preprocessor ():

    def __init__(self,
                batch_size,
                dataset_dir: str = "datasets/grocery_data.csv",
                preprocessed_dir: str = "datasets/grocery_preprocessed.pkl",
                city_dir: str = "datasets/city.pkl",
                product_dir: str = "datasets/product.pkl") -> None:

        if os.path.exists(preprocessed_dir):
            print("Load Data")
            self.dataset = pd.read_pickle(preprocessed_dir)
        else:
            print("Creating Preprocessed Data")
            self.dataset = self.load_data(dir = dataset_dir, save_dir = preprocessed_dir)

        if os.path.exists(city_dir):
            print("Load Data")
            with open(city_dir, "rb") as rc_handler:
                self.city_id: Dict = pickle.load(rc_handler)
        else:
            print("Creating Preprocessed Data")
            self.city_id = self.extract_index(self.dataset, index_name = "City", index_key="Sales")

        if os.path.exists(product_dir):
            print("Load Data")
            with open(product_dir, "rb") as rc_handler:
                self.product_id: Dict = pickle.load(rc_handler)
        else:
            print("Creating Preprocessed Data")
            self.product_id = self.extract_index(self.dataset, index_name = "Product", index_key="Quantity Ordered")
        
    def extract_index(self, dataset: pd.DataFrame, index_name: str, index_key: str) -> Dict:
        sum_for_data = dataset.groupby([index_name]).sum()
        sum_for_data = sum_for_data[["Quantity Ordered", "Sales"]].reset_index()

        sorted_data = sum_for_data.sort_values(by=index_key, ascending=False)

        data_dict = {c : i for i, c in enumerate(sorted_data.reset_index()[index_name].tolist())}

        with open("datasets/" + index_name.lower() + ".pkl", "wb") as w_handler:
            pickle.dump(data_dict, w_handler, protocol=pickle.HIGHEST_PROTOCOL)

        return data_dict

    def get_city(self, address):
        return address.split(",")[1].strip(" ")

    def get_state(self, address):
        return address.split(",")[2].split(" ")[1]

    def load_data(self, dir: str, save_dir: str):
        # Load CSV File
        d_grocery = pd.read_csv(dir)
        # Grab only NAN row
        d_nan = d_grocery[d_grocery.isna().any(axis=1)]

        # Remove NAN from datasets
        d_grocery = d_grocery.dropna(how="all")

        # In column Order Date remove Cell that have string instead of datetime
        d_grocery = d_grocery[d_grocery['Order Date'].str[0:2] != 'Or']
        
        # Convert and Corecting All Item in Column Quantity Ordered and Price Each to Integer
        d_grocery["Quantity Ordered"] = pd.to_numeric(d_grocery["Quantity Ordered"])
        d_grocery["Price Each"] = pd.to_numeric(d_grocery["Price Each"])

        # Get Date Item (Month Only) In cell : 2 Ways

        # 1 Way, Slice String in Cell Order Date in order to get only month
        d_grocery["Month"] = d_grocery["Order Date"].str[0:2]
        # Change Data Type from String to Integer
        d_grocery["Month"] = d_grocery["Month"].astype('int32')


        # 2 Way, With DateTime Function
        d_grocery["Month"] = pd.to_datetime(d_grocery["Order Date"]).dt.month

        # Apply Function to Column in Dataframe and Add to New Column
        d_grocery["City"] = d_grocery["Purchase Address"].apply(lambda x: f"{self.get_city(x)} {self.get_state(x)}")

        # Calculate Sales which quantity * price each
        d_grocery["Sales"] = d_grocery["Quantity Ordered"].astype("int") * d_grocery["Price Each"].astype("float")
        
        d_grocery.to_pickle(save_dir)

        return d_grocery

    def preprocessor(self):
        data = self.dataset[["Order Date", "Product", "City", "Quantity Ordered", "Sales"]]
  
        data["Date"] = pd.to_datetime(data["Order Date"].apply(lambda x: x.split(" ")[0]))
        data = data[["Date", "Product", "City", "Quantity Ordered", "Sales"]]
        
        data_sales = data.groupby("Date")["Sales"].sum().reset_index()
        data_sales = data_sales.rename(columns = {"Date": "date", "Sales": "sales"})

        data_diff = data_sales.copy()
        data_diff["prev_sales"] = data_diff["sales"].shift(1)
        data_diff = data_diff.dropna()
        data_diff["diff"] = (data_diff["sales"] - data_diff["prev_sales"])

        data_supervised = data_diff.drop(["prev_sales"], axis = 1)

        for inc in range(1, 13):
            field_name = "lag_" + str(inc)
            data_supervised[field_name] = data_supervised["diff"].shift(inc)
        
        data_supervised = data_supervised.dropna().reset_index(drop = True)
        
        model = smf.ols(formula = 'diff ~ lag_1 + lag_2 + lag_3 + lag_4 + lag_5 + lag_6 + lag_7 + lag_8 + lag_9 + lag_10 + lag_11 + lag_12', data = data_supervised)
        model_fit = model.fit()
        
        regression_adj_rsq = model_fit.rsquared_adj

        data_model = data_supervised.drop(["sales", "date"], axis = 1)

        train_set = data_model[0: -6].values
        test_set = data_model[-6:].values

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

        return x_train, y_train, x_test, y_test, scaler, train_set_scaled, test_set_scaled, data_sales