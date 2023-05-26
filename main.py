import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Load the data
df1 = pd.read_csv('C:/Users/Shantanu/Downloads/LSTM/google_stock_2018-07-08.csv')
df2 = pd.read_csv('C:/Users/Shantanu/Downloads/LSTM/nike_stock_2018-05-26.csv')

# Concatenate the dataframes and drop the date column
df = pd.concat([df1['Close'], df2['Close']], axis=1)
df.columns = ['Company1', 'Company2']
df = df.dropna()

# Scale the data
scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(df)

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train = scaled_df[:train_size]
test = scaled_df[train_size:]

# Create a function to split the data into X and y
def create_dataset(data, n_steps):
    X, y = [], []
    for i in range(len(data)-n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

# Define the number of time steps
n_steps = 30

# Split the data into X and y
X_train, y_train = create_dataset(train, n_steps)
X_test, y_test = create_dataset(test, n_steps)

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, 2)))
model.add(Dense(2))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16)

# Make predictions on the testing
y_pred = model.predict(X_test)
mse1 = np.mean((y_test[:,0] - y_pred[:,0])**2)
mse2 = np.mean((y_test[:,1] - y_pred[:,1])**2)
plt.plot(y_pred[:,0], label='Predicted GOOGLE')
plt.plot(y_test[:,1], label='NIKE')
plt.plot(y_pred[:,1], label='Predicted NIKE')

# Plot the actual and predicted stock prices
plt.plot(y_test[:,0], label='GOOGLE')
plt.legend()
plt.show()

if mse1 < mse2:
    best_company = 'GOOGLE'
else:
    best_company = 'NIKE'

# Print the results
print('Mean Squared Error for GOOGLE:', mse1)
print('Mean Squared Error for NIKE:', mse2)
print('The best stock is:', best_company)

