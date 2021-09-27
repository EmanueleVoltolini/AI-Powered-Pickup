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

z = torch.diff(x,y)

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

#%%

x = torch.rand(3, requires_grad=True)
print(x)

y = x+2

