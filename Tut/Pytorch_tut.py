#%%
import torch

x = torch.rand(2, 2)
y = torch.rand(2, 2)
print(x)
print(y)
z = x + y
z = torch.add(x,y)
print(z)

y.add_(x)    #every function that has _ in it will modify the variable it refers
print(y)

z = torch.sub(x,y)
print(z)

z = torch.mul(x,y)
y.mul_(x)

z = torch.sub(x,y)

#%%
x = torch.rand(5, 3)
print(x)
print(x[1,:])
print(x[1,0].item())   # give the actual value of the tensor


## Reshape a tensor
x = torch.rand(4,4)
print(x)
y = x.view(-1,8)       #if you put a -1 python feagure the second dimension that hasn't been specified
print(y)
print(y.size)

# Convert numpy to torch and viceversa
import numpy as np

a = torch.ones(5)
print(a)
b =a.numpy()
print(b)
print(type(b))          #if we modify a or b the other element change too, they point to the same memory state

a.add_(1)
print(a)
print(b)

a = np.ones(5)
print(a)
b= torch.from_numpy(a)
print(b)

a +=1
print(a)
print(b)

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5,device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    #z.numpy()                          #numpy only on CPU
    z = z.to("cpu")

x = torch.ones(5, requires_grad=True)
print(x)

#%% Gradient

x = torch.rand(3, requires_grad=True)
print(x)

y = x+2
print(y)

z = y*y*2
#z = z.mean()
print(z)

v = torch.tensor([0.1,1.0,0.001], dtype = torch.float32)
z.backward(v)
print(x.grad)

#x.requires_grad_(False)
#x.detach()
#with torch.no_grad():

x.requires_grad_(False)
print(x)

y = x.detach()
print(y)

with torch.no_grad():
    y = x +2
    print(y)

# %%

weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad)

    weights.grad.zero_()

# %% Backpropagation

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

# forward pass and compute the loss
y_hat = w * x
loss = (y_hat - y)**2

print(loss)

#backward pass
loss.backward()
print(w.grad)

### Update weights
### next forward and backwards

# %% Automatic gradient computation 

# first manually 
import numpy as np

# f = w * x

# f = 2 * x
X = np.array([1,2,3,4], dtype = np.float32)
Y = np.array([2,3,6,8], dtype = np.float32)

w = 0.0

#model prediction 
def forward(x):
    return w *x
#loss = MSE
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()
#gradient
#MSE = 1/N * (w*x - y)**2
#dJ/dw = 1/N 2x (w*x - y)
def gradient(x,y,y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

#Training 
learning_rate = 0.01
n_inters = 15

for epoch in range(n_inters):
    # prediction = forward pass
    y_pred = forward(X)

    #loss 
    l = loss(Y,y_pred)

    # gradients 
    dw = gradient(X,Y,y_pred)

    # update weights
    w -= learning_rate * dw

    if epoch % 2 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(10):.3f}')

# %% Automatic gradient computation 

# second manually and pytorch
import numpy as np
import torch

# f = w * x

# f = 2 * x
X = torch.tensor([1,2,3,4], dtype = torch.float32)
Y = torch.tensor([2,4,6,8], dtype = torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

#model prediction 
def forward(x):
    return w *x
#loss = MSE
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()


print(f'Prediction before training: f(5) = {forward(5):.3f}')

#Training 
learning_rate = 0.01
n_inters = 100

for epoch in range(n_inters):
    # prediction = forward pass
    y_pred = forward(X)

    #loss 
    l = loss(Y,y_pred)

    # gradients = backward pass
    l.backward() #dl/sw

    # update weights
    with torch.no_grad():
        w -= learning_rate * w.grad
    # zero gradients
    w.grad.zero_()
    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')

# %% 
# Automatic gradient computation 
# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#    -forward pass: compute prediction
#    -backward pass: gradients
#    -update weights
# Third manually prediction and pytorch
import torch
import torch.nn as nn

# f = w * x

# f = 2 * x
X = torch.tensor([1,2,3,4], dtype = torch.float32)
Y = torch.tensor([2,4,6,8], dtype = torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

#model prediction 
def forward(x):
    return w *x


print(f'Prediction before training: f(5) = {forward(5):.3f}')

#Training 
learning_rate = 0.01
n_inters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr= learning_rate)

for epoch in range(n_inters):
    # prediction = forward pass
    y_pred = forward(X)

    #loss 
    l = loss(Y,y_pred)

    # gradients = backward pass
    l.backward() #dl/sw

    # update weights
    optimizer.step()
    # zero gradients
    optimizer.zero_grad()
    
    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')

# %% 
# Automatic gradient computation 
# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#    -forward pass: compute prediction
#    -backward pass: gradients
#    -update weights
# Fourth pytorch
import torch
import torch.nn as nn

# f = w * x

# f = 2 * x
X = torch.tensor([[1],[2],[3],[4]], dtype = torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype = torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape

input_size = n_features
output_size = n_features

model = nn.Linear(input_size,output_size)

#custom model
class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression,self).__init__()
        #define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self,x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)
print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

#Training 
learning_rate = 0.01
n_inters = 150

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)

for epoch in range(n_inters):
    # prediction = forward pass
    y_pred = model(X)

    #loss 
    l = loss(Y,y_pred)

    # gradients = backward pass
    l.backward() #dl/sw

    # update weights
    optimizer.step()
    # zero gradients
    optimizer.zero_grad()
    
    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')


# %% LINEAR REGRESSION
# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#    -forward pass: compute prediction
#    -backward pass: gradients
#    -update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets 
import matplotlib.pyplot as plt
# 0) Prepare data 
X_numpy, Y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))
y = Y.view(Y.shape[0], 1)

n_samples, n_features = X.shape

# 1) Model
input_size = n_features
output_size = 1
model = nn.Linear(input_size,output_size)

# 2) Construct loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), learning_rate)

# 3) Training loop
num_epochs = 100
for epoch in range(num_epochs):
    #forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted,y)

    #backward pass
    loss.backward()
    #update
    optimizer.step()

    optimizer.zero_grad()

    if(epoch+1) % 10 ==0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy,Y_numpy, 'ro')
plt.plot(X_numpy,predicted, 'b')
plt.show()

# %% LOGISTIC REGRESSION 
# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#    -forward pass: compute prediction
#    -backward pass: gradients
#    -update weights
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) Prepare the data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

#scale 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)

# 1) model
# f = wx + b, sigmoid at the end
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    
    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_features)

# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)
# 3) Training loop
num_epochs = 100
for epoch in range(num_epochs):
    #forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted,y_train)

    #backward pass
    loss.backward()

    #updates
    optimizer.step()

    optimizer.zero_grad()

    if(epoch+1) % 10 ==0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')
