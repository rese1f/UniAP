import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce


class ClassificationHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(1536, 1536)
        self.linear_2 = nn.Linear(1536, 1536)
        self.linear_3 = nn.Linear(1536, 1536)
        self.linear_4 = nn.Linear(1536, 1536)
        self.head_1 = nn.Linear(1536, 256)
        self.head_2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, W_Qs, W_Ss):
        B, T, N_Q, _, _, _ = W_Qs[0].shape 
        B, T, N_S, _, _, _ = W_Ss[0].shape 
        Ws = []
        for i in range(4):
            W_Q, W_S = W_Qs[i], W_Ss[i]
            if W_Q.size(2) != W_S.size(2):
                W_Q = repeat( W_Q, 'B T 1 C H W -> B T N C H W', N = W_S.size(2))
            W = torch.concat((W_Q, W_S), dim=3)
            W = rearrange(W, 'B T N C H W -> (B T N) (H W) C')
            W = torch.mean(W, dim=1)
            Ws.append(W)

        W1, W2, W3, W4 = Ws
        
        x = self.linear_1(W1)
        x = x + W1
        x = self.relu(x)

        x = self.linear_2(W2)
        x = x + W2
        x = self.relu(x)

        x = self.linear_3(W3)
        x = x + W3
        x = self.relu(x)

        x = self.linear_4(W4)
        x = x + W4
        x = self.relu(x)

        x = self.head_1(x)
        x = self.relu(x)
        x = self.head_2(x)
        x = self.sigmoid(x)
    
        x = rearrange(x, '(B T N) F -> B T N F', B=B, T=T, N=N_S)
        x = reduce(x, 'B T (N_Q n) F -> B T N_Q F', 'mean', N_Q=N_Q)

        return x