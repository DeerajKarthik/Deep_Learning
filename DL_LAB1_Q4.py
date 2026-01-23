import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
])


y = torch.tensor([
    [0.],
    [1.],
    [1.],
    [0.]
])


class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.hidden = nn.Linear(2, 2)  
        self.output = nn.Linear(2, 1)   

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x


model = XORNet()

criterion = nn.BCELoss()       
optimizer = optim.SGD(model.parameters(), lr=0.1)

epochs = 50

for epoch in range(epochs):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


print("\nXOR Predictions:")
with torch.no_grad():
    predictions = model(X)
    print(torch.round(predictions))


