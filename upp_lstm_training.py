import pandas as pd
from datetime import datetime
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from matplotlib import pyplot as plt

"""
1. Prepare Feature Set
"""

# parser for date columns [date, start_time, end_time]
def parser(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


# read the data
df = pd.read_csv('data/dataset_jongno_refined.csv.csv', parse_dates=[0], date_parser=parser)

df.index = df.datetime

# set up the transformer (one hot encoder, feature scaler)
preprocess = make_column_transformer(
    (OneHotEncoder(), [1, 17]),
    (RobustScaler(), [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
)

train = preprocess.fit_transform(df)        # scaling and encoding

# put X and y all together to ndarray
y_train = df['pm10'].values
y_train = y_train.reshape(len(y_train), 1)
train = np.append(train, y_train, axis=1)

# calculate the size of input feature vector
input_size = train.shape[1]-1       # excludes the target variable

"""
    range(size - seq_len + 1): for predicting current time steps
    range(size - seq_len):  for predicting day-ahead  
"""

# transformation (ndarray -> torch)
def transform_data(input_data, seq_len):
    x_lst, y_lst = [], []
    size = len(input_data)
    for i in range(size - seq_len + 1):
        # input sequence
        seq = input_data[i:i+seq_len, :input_size]
        # target values of current time steps
        target = input_data[i+seq_len-1, -1]
        x_lst.append(seq)
        y_lst.append(target)
    x_arr = np.array(x_lst)
    y_arr = np.array(y_lst)
    print("[INFO]x_arr.shape = " + str(x_arr.shape))
    print("[INFO]y_arr.shape = " + str(y_arr.shape))
    return x_arr, y_arr


# specify a device (gpu|cpu)
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.float

seq_len = 7
batch_size = 100

x_train, y_train = transform_data(train, seq_len)

# calculate a number of batches
num_batches = int(x_train.shape[0] / batch_size)

if x_train.shape[0] % batch_size != 0:
    num_batches += 1


"""
2. Model Definition
"""

# hyperparameters
hidden_size = 150        # default: 150
output_dim = 1
num_layers = 3          # default: 3
learning_rate = 1e-3    # default: 1e-3
num_epochs = 300       # default: 300


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.seq_len = 50
        self.num_layers = num_layers
        # define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        # define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)
    def init_hidden(self):
        # initialize hidden states
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).type(dtype),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).type(dtype))
    def forward(self, input):
        # forward pass through LSTM layer
        lstm_out, self.hidden = self.lstm(input) # [1, batch_size, 24]
        # only take the output from the final time step
        y_pred = self.linear(lstm_out[:, -1])
        return y_pred.view(-1)

model = LSTM(input_size, hidden_size, batch_size=1, output_dim=output_dim, num_layers=num_layers)
model.seq_len = seq_len
model.cuda()    # for cuda
loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


"""
3. Train the Model
"""

hist = np.zeros(num_epochs)     # loss history
for t in range(num_epochs):     # for each epoch
#for t in range(1):             # [TEST]
    y_pred = np.empty(0)
    for i in range(num_batches):  # for each batch
        print("Training the model: %d/%dth epoch, %d/%dth batch..."
              % (t+1, num_epochs, i+1, num_batches), end='\r')
        # last batch
        if i == num_batches-1:
            x_batch_arr = x_train[i*batch_size:]
            y_batch_arr = y_train[i*batch_size:]
        # other batches
        else:
            x_batch_arr = x_train[i*batch_size:i*batch_size+batch_size]
            y_batch_arr = y_train[i*batch_size:i*batch_size+batch_size]
        # transformation (ndarray -> torch)
        x_batch = Variable(torch.from_numpy(x_batch_arr).float()).type(dtype)
        y_batch = Variable(torch.from_numpy(y_batch_arr).float()).type(dtype)
        model.batch_size = x_batch.shape[0]
        model.hidden = model.init_hidden()
        # get predictions for the batch
        pred_i = model(x_batch)
        # forward pass
        loss_train = loss_fn(pred_i, y_batch)
        # zero out gradient, else they will accumulate between epochs
        optimizer.zero_grad()
        # backward pass
        loss_train.backward()
        # update parameters
        optimizer.step()
        # store the predictions
        y_pred = np.append(y_pred, pred_i.detach().cpu().numpy(), axis=0)
    if t == 0:
        loss_prev = float('inf')
    else:
        loss_prev = hist[t-1]
    # measure a loss in the current epohch
    loss_train = loss_fn(torch.from_numpy(y_pred), torch.from_numpy(y_train)).item()
    print("[INFO] Epoch ", t, ", Loss: ", loss_train, ", Difference: ", (loss_train - loss_prev))
    hist[t] = loss_train


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



"""
4. Visualization
"""

# default visualization setup
plt.figure(dpi=100)     # set the resolution of plot
# set the default parameters of visualization
color_main = '#2c4b9d'
color_sub = '#00a650'
color_ssub = '#ef9c00'
color_sssub = '#e6551e'
font_family = 'Calibri'
plt.rcParams.update({'font.family': font_family, 'font.size': 23, 'lines.linewidth': 1,
                    "patch.force_edgecolor": True, 'legend.fontsize': 18})

# line plot
plt.plot(y_train, label="Actual Data", color=color_main)
plt.plot(y_pred, label="Predictions", color=color_sub)
plt.show()

# scatter plot
fig, ax = plt.subplots()
ax.scatter(y_train, y_pred, 10, color=color_main)   # 10: marker size
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], '--', lw=1, color=color_ssub)
ax.set_xlabel('Actual Data')
ax.set_ylabel('Predictions')
plt.show()