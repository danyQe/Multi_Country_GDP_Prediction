import torch

data = torch.load('/content/Multi_Country_GDP_Prediction/dataset/LSTM_data_gdp_light_mean_q_t8_13-19.pt', map_location=torch.device('cpu'))
print(data)
