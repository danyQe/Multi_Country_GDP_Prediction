import numpy as np
import pandas as pd
from utils.metrics import metric
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import random
import itertools
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tqdm import tqdm  # 进度条显示
import time

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4"


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果你使用多个GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 计算最小值和最大值, 并进行归一化
def norm_lstm_tensor(data, labels, freq='quarter'):
    if freq == 'quarter':
        drop_index = -3
    else:
        drop_index = -2
    
    # 将最后一维的数据分开
    data_to_norm = data[:, :, :drop_index]  # 除了最后一个维度
    labels_to_norm = labels[:, :drop_index]  # 除了最后一个维度

    # 计算data的最小值和最大值
    max_vals_data, _ = torch.max(data_to_norm, dim=0)  # 对第一个维度求最大值
    max_vals_data, _ = torch.max(max_vals_data, dim=0)  # 再对第二个维度求最大值
    
    min_vals_data, _ = torch.min(data_to_norm, dim=0)  # 对第一个维度求最小值
    min_vals_data, _ = torch.min(min_vals_data, dim=0)  # 再对第二个维度求最小值

    # 计算labels的最小值和最大值
    min_vals_label = labels_to_norm.min(dim=0, keepdim=True).values[-1]
    max_vals_label = labels_to_norm.max(dim=0, keepdim=True).values[-1]

    min_value_all = torch.min(min_vals_data, min_vals_label)
    max_value_all = torch.max(max_vals_data, max_vals_label)

    min_value = min_value_all[-1]
    max_value = max_value_all[-1]

    # 计算 Min-Max 归一化
    # 对 data (去除最后一个维度的部分) 进行 Min-Max 归一化
    normalized_data_to_norm = (data_to_norm - min_value_all) / (max_value_all - min_value_all)

    # 对 label (去除最后一个维度的部分) 进行 Min-Max 归一化
    normalized_labels_to_norm = (labels_to_norm - min_value_all) / (max_value_all - min_value_all)

    # 重新拼接保留的最后一个维度
    normalized_data = torch.cat([normalized_data_to_norm, data[:, :, drop_index:]], dim=2)
    normalized_label = torch.cat([normalized_labels_to_norm, labels[:, drop_index:]], dim=1)
    
    return normalized_data, normalized_label, min_value, max_value


def split_lstm_dataset_by_year(data, labels, year, freq='quarter'):
    if freq == 'quarter':
        dim_index = -2
    else:
        dim_index = -1
        
    train_index_list = []
    test_index_list = []
    for i in range(len(labels)):
        if labels[i, dim_index] >= year:
            test_index_list.append(i)
        else:
            train_index_list.append(i)


    train_data = data[train_index_list, :, :dim_index-1]
    train_targets = labels[train_index_list, :dim_index-1]
    
    test_data = data[test_index_list, :, :dim_index-1]
    test_targets = labels[test_index_list, :dim_index-1]
    return train_data, test_data, train_targets, test_targets


def reverse_norm(row):
    # row = row.cpu()
    if len(row.shape) == 2:
        gap = max_value.item() - min_value.item()
        return row[:, -1] * gap + min_value.item()
    else:
        gap = max_value.item() - min_value.item()
        return row * gap + min_value.item()


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def get_linear_res(train_data, test_data, train_targets, test_targets):
    X_train = train_data
    y_train = train_targets

    X_test = test_data
    y_test = test_targets
    
    # 创建线性回归模型
    model = LinearRegression()
    
    # 训练模型
    model.fit(X_train, y_train)


    # 进行预测
    y_pred_train = model.predict(X_train)[:, -1]
    y_pred_test = model.predict(X_test)[:, -1]

    y_train = y_train[:, -1]
    y_test = y_test[:, -1]

    y_pred_train = reverse_norm(y_pred_train)
    y_pred_test = reverse_norm(y_pred_test)
    y_train = reverse_norm(y_train)
    y_test = reverse_norm(y_test)

    # 评估模型
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    
    mae, mse, rmse, mape, mspe, rse, corr = metric(torch.Tensor(y_train),
                                                   torch.Tensor(y_pred_train))

    train_res = [x.item() for x in [mae, mse, rmse, mape, mspe, rse, corr]]
    
    mae, mse, rmse, mape, mspe, rse, corr = metric(torch.Tensor(y_test), 
                                                   torch.Tensor(y_pred_test))

    test_res = [x.item() for x in [mae, mse, rmse, mape, mspe, rse, corr]]

    return train_res, test_res



res_list = []
for file_item in os.listdir('/content/Multi_Country_GDP_Prediction/dataset/'):
    if ('LSTM_data_' in file_item):
        print(file_item)
    else:
        continue
# for file_item in ['MLP_data_q_95-19.pt']:
    temp_dict = {}
    start_time = time.time()
    data_path = '/content/Multi_Country_GDP_Prediction/dataset/' + file_item
    label_path = '/content/Multi_Country_GDP_Prediction/dataset/' + file_item.replace('LSTM_data', 'LSTM_label')
    
    set_seed(1)
    data = torch.load(data_path)
    labels = torch.load(label_path)
    
    data, labels, min_value, max_value = norm_lstm_tensor(data, labels, 'quarter')

    # 13-19 use 2019 as test dataset, other use 2018-2019 as test dataset
    if '13-19' in file_item:
        year = 2019
    else:
        year = 2018
        
    train_data, test_data, train_targets, test_targets = split_lstm_dataset_by_year(data, labels, year, freq='quarter')

    # flatten data
    train_data = train_data.view(train_data.size()[0], -1)
    test_data = test_data.view(test_data.size()[0], -1)
    
    # print(train_data.size())
    # print(train_targets.size())
    # break
    train_res, test_res = get_linear_res(train_data, test_data, train_targets, test_targets)

    temp_dict['data'] = file_item
    temp_dict['train_mae'] = train_res[0]
    temp_dict['train_mse'] = train_res[1]
    temp_dict['train_rmse'] = train_res[2]
    temp_dict['train_mape'] = train_res[3]
    temp_dict['train_mspe'] = train_res[4]
    temp_dict['train_rse'] = train_res[5]
    temp_dict['train_corr'] = train_res[6]


    temp_dict['test_mae'] = test_res[0]
    temp_dict['test_mse'] = test_res[1]
    temp_dict['test_rmse'] = test_res[2]
    temp_dict['test_mape'] = test_res[3]
    temp_dict['test_mspe'] = test_res[4]
    temp_dict['test_rse'] = test_res[5]
    temp_dict['test_corr'] = test_res[6]

    res_list.append(temp_dict)
    print('cost time: ', time.time() - start_time)

df_res = pd.DataFrame(res_list)
print(df_res)
df_res.to_csv('linear_lstm_q_res.csv')


