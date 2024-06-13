import torch
from torch.nn import functional as F

def print_matrices(a,b):
    c = a @ b
    print("a = ")
    print(a)
    print("--")
    print("b = ")
    print(b)
    print("--")
    print("c = ")
    print(c)
    print("--")

torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)
print(x.shape)



xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1] # (t, C)
        xbow[b,t] = torch.mean(xprev, 0) #each element is the mean of it and the previous elements for each channel


#matmul trick
torch.manual_seed(42)
a = torch.ones(3,3)
b = torch.randint(0, 10, (3,2)).float()

print_matrices(a, b)
    
lower_triangular = torch.tril(torch.ones(3,3))
a = lower_triangular/torch.sum(lower_triangular, 1, keepdim = True)

print_matrices(a,b)

wei = torch.tril(torch.ones(T,T))
wei = wei / wei.sum(1, keepdim= True)
xbow2 = wei @ x # (B, T, T) @ (B, T, C) -----> (B, T, C)



tril = torch.tril(torch.ones(T,T))
wei = torch.zeros((T, T)) #start as 0 but are going to be data-dependent
wei = wei.masked_fill(tril ==0, float("-inf")) #cannot look into the future
wei = F.softmax(wei, dim = -1)
xbow3 = wei @ x # (B, T, T) @ (B, T, C) -----> (B, T, C)