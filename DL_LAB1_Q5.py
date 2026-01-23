import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

file_path = "/home/deeraj/Downloads/house_price_full+(2) - house_price_full+(2) - house_price_full+(2) - house_price_full+(2).csv"
df = pd.read_csv(file_path)

df.head()

feature_cols = ['bedrooms', 'sqft_living']
target_col = 'price'

X = df[feature_cols].values
y = df[target_col].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(2, 2)   # W', b'
        self.output = nn.Linear(2, 1)   # W'', b''

    def forward(self, x):
        z1 = self.hidden(x)
        h1 = torch.relu(z1)
        z2 = self.output(h1)
        return z2                     
    
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 2000

for epoch in range(epochs):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

with torch.no_grad():
    test_pred = model(X_test)
    test_loss = criterion(test_pred, y_test)

print("Test MSE:", test_loss.item())

