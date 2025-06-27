import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pyts.image import GramianAngularField
import joblib
from skimage.transform import resize

df = pd.read_csv('dataset/input.csv')
empty_cells = np.arange(0, df['DK_load_actual_entsoe_transparency'].size, 1.0)
empty_cells = empty_cells.reshape(df['DK_load_actual_entsoe_transparency'].size // 24, 24)

power_load = pd.DataFrame(data=empty_cells.copy(),columns=[x for x in range(24)])
solar_generation = pd.DataFrame(data=empty_cells.copy(), columns=[x for x in range(24)])
wind_generation = pd.DataFrame(data=empty_cells.copy(), columns=[x for x in range(24)])

# Populate the dataframes with 24 hour records
for x in range(df['DK_load_actual_entsoe_transparency'].size):
    power_load.loc[x//24, x%24] = df['DK_load_actual_entsoe_transparency'][x]
    solar_generation.loc[x//24, x%24] = df['DK_solar_generation_actual'][x]
    wind_generation.loc[x//24, x%24] = df['DK_wind_generation_actual'][x]

data = pd.concat([wind_generation, power_load, solar_generation], axis=1)

scaler = joblib.load('models/scaler.save')
data = scaler.transform(data)
season_names = {0:'Winter', 1:'Spring', 2:'Summer', 3:'Autumn'}

class MLP(nn.Module):
    # Input size is 72 because we have 3 days with 24 hours in each
    # Output size is 4 because we have 4 seaons to predict
    # Hidden size is 128 because it experimentally gave the best accuracy results
    def __init__(self, input_size=24*3, output_size=4, hidden_size=72):
        super(MLP, self).__init__()

        self.conv = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 3*hidden_size),
            nn.ReLU(),
            nn.Linear(3*hidden_size, output_size)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class CNN_1D(nn.Module):
    def __init__(self):
        super(CNN_1D, self).__init__()
        self.conv1 = nn.Sequential(
            # (batch_size, 3, 24) -> (batch_size, 24, 22) 
            nn.Conv1d(in_channels=3, out_channels=24, kernel_size=3),
            # (batch_size, 24, 22) -> (batch_size, 24, 22) 
            nn.LeakyReLU(),
            # (batch_size, 24, 22) -> (batch_size, 24, 11) 
            nn.MaxPool1d(kernel_size=2, stride=2),
            # (batch_size, 24, 22) -> (batch_size, 24, 11)
            nn.BatchNorm1d(24)
        )
        self.conv2 = nn.Sequential(
            # (batch_size, 24, 11) -> (batch_size, 96, 9)
            nn.Conv1d(in_channels=24, out_channels=96, kernel_size=3),
            # (batch_size, 96, 9) -> (batch_size, 96, 9)
            nn.ReLU(),
            # (batch_size, 96, 9) -> (batch_size, 96, 4)
            nn.MaxPool1d(kernel_size=2, stride=2),
            # (batch_size, 96, 9) -> (batch_size, 96, 4)
            nn.Dropout1d(0.15)
        )
        # (batch_size, 96, 9) -> (batch_size, 1, 384)
        self.flatten = nn.Flatten()
        # (batch_size, 1, 384) -> (batch_size, 1, 4)
        self.fully_connected = nn.Linear(96*4, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fully_connected(x)
        return x

class CNN_2D(nn.Module):
    # hidden size was chosen experimentally
    def __init__(self, hidden_size=96):
        super(CNN_2D, self).__init__()

        self.conv1 = nn.Sequential(
            # (batch_size, 3, 4, 6) -> (batch_size, 48, 4, 6)
            nn.Conv2d(in_channels=3, out_channels=hidden_size, kernel_size=3, padding=1),
            # (batch_size, 3, 4, 6) -> (batch_size, 48, 4, 6)
            nn.BatchNorm2d(hidden_size),
            # (batch_size, 3, 4, 6) -> (batch_size, 48, 4, 6)
            nn.LeakyReLU(),
            # (batch_size, 3, 4, 6) -> (batch_size, 48, 2, 3)
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            # (batch_size, 48, 2, 3) -> (batch_size, 96, 2, 3)
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size * 2, kernel_size=3, padding=1),
            # (batch_size, 96, 2, 3) -> (batch_size, 96, 2, 3)
            nn.BatchNorm2d(hidden_size * 2),
            # (batch_size, 96, 2, 3) -> (batch_size, 96, 2, 3)
            nn.LeakyReLU(),
            # (batch_size, 96, 2, 3) -> (batch_size, 96, 1, 1)
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # (batch_size, 96, 1, 1) -> (batch_size, 96, 1, 1)
        self.dropout = nn.Dropout(0.3) # dropout rate chosen experimentally
        # (batch_size, 96, 1, 1) -> (batch_size, 96, 1, 1)
        self.flatten = nn.Flatten()
        # (batch_size, 96, 1, 1) -> (batch_size, 48, 1, 1)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        # (batch_size, 48, 1, 1) -> (batch_size, 4, 1, 1)
        self.fc2 = nn.Linear(hidden_size, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def predict(x_reshaped, model):
    model.eval()

    with torch.no_grad():
        test_outputs = model(x_reshaped)
        test_preds = torch.argmax(test_outputs, dim=1)
        
    for i in test_preds:
            print(season_names[i.item()])

def transformToImage(data):
    # GramianAngularField with difference method gave the best accuracy results
    gaf = GramianAngularField(image_size=24, method='difference', sample_range = (-1, 1))

    gaf_transformed = []
    for feature_idx in range(data.shape[1]):     
        feature_gaf = gaf.transform(data[:, feature_idx, :])
        # image size will be 4x6
        feature_gaf = np.stack([resize(img, (4, 6)) for img in feature_gaf])
        gaf_transformed.append(np.expand_dims(feature_gaf, axis=1))

    data_gaf = np.concatenate(gaf_transformed, axis=1)
    # print("Final shape:", data_gaf.shape)
    return data_gaf


model1 = MLP()
model2 = CNN_1D()
model3 = CNN_2D()
model1.load_state_dict(torch.load("models/MLP.pth"))
model2.load_state_dict(torch.load("models/1D_CNN.pth"))
model3.load_state_dict(torch.load("models/2D_CNN.pth"))


x = torch.cat([torch.tensor(data, dtype=torch.float32)], dim=0)
print('MLP:')
predict(x, model1)
x = np.array(data)
x_reshaped = x.reshape(-1, 3, 24)


print('\n1D CNN:')
predict(torch.tensor(x_reshaped, dtype=torch.float32), model2)

x_image = transformToImage(x_reshaped)

print('\n2D CNN:')
predict(torch.tensor(x_image, dtype=torch.float32), model3)

# print(model3.state_dict())
