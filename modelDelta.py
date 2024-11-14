import torch
import torch.nn.functional as F
import numpy as np
from modules.utils import timer

def softThresh(V, alpha, device):
    zero_t = torch.from_numpy(np.zeros(V.shape)).type(torch.float32).to(device)
    h = torch.sign(V) * torch.maximum(torch.abs(V)-alpha, zero_t)
    return h

class LFISTA(torch.nn.Module):
    def __init__(self, hyp, idctM=None):
        super(LFISTA, self).__init__()

        self.obsDim = hyp["obsDim"]
        self.sparseDim = hyp["sparseDim"]
        self.fistaIter = hyp["fistaIter"]
        self.device = hyp["device"]
        self.learning_rate = hyp["learning_rate"]
        
        if hyp["MODE"].upper() == "TRAIN":
            self.isTrain = True
        else:
            self.isTrain = False

        # Used for initalization
        self.idctM = idctM

        # L = torch.Tensor([[1.0]])
        # self.L = torch.nn.Parameter(L, requires_grad=True)
        self.L = None

        # Softshrink threshold value
        alpha = torch.Tensor([[hyp["alpha_initialize"]]])
        self.alpha = torch.nn.Parameter(alpha, requires_grad=True)
        # self.alpha = hyp["alpha_initialize"]
        # self.alpha = torch.zeros((2*self.sparseDim,1), dtype=torch.float32, requires_grad=False)
        # self.alpha[:self.sparseDim, :] = 0.01
        # self.alpha[self.sparseDim:, :] = 0.1
        # self.alpha.to(self.device)

        # self.alpha = 1e-4

        # Define and Initialize the dictionary matrix
        W = torch.randn((self.sparseDim, self.sparseDim))
        W = F.normalize(W, p=2, dim=0)

        # self.register_parameter("W", torch.nn.Parameter(W, requires_grad=True))
        self.W = torch.nn.Parameter(W, requires_grad=True)
        self.W.to(self.device)

        # self.softThresh = torch.nn.Softshrink(lambd=self.learning_rate/self.L * 0.5)
        # self.softThresh = torch.nn.ReLU()

    def get_param(self, name):
        return self.state_dict(keep_vars=True)[name]

    def normalize(self):
        self.W.data = F.normalize(self.W.data, p=2, dim=0)

    def PhiW(self, x, src):
        theta_m = x[:, :self.obsDim, :]
        delta = x[:, self.obsDim:, :]

        result = torch.mul(src, torch.matmul(self.W, theta_m)) + delta
        return result

    def PhiWT(self, y, src):
    
        Theta_m = torch.mul(src, y)
        delta = torch.clone(y)

        result = torch.cat((torch.matmul(torch.t(self.W), Theta_m) , delta), 1)
        return result

    def projection(self, x):
        theta_m = x[:,:self.obsDim,:]
        # delta = x[:,2*self.obsDim:3*self.obsDim,:]

        # theta_m >= 0
        theta_m[theta_m < 0] = 0

        # delta between -1 and 1
        # delta[delta > 1] = 1
        # delta[delta < -1] = -1

        x[:,:self.obsDim,:] = theta_m
        # x[:,2*self.obsDim:3*self.obsDim,:] = delta

        return x

    def forward(self, src, Y):

        self.device = self.W.device

        num_batches = Y.shape[0]

        # Initialize all the variables
        x_old = torch.zeros((num_batches, 2*self.sparseDim, 1), device=self.device, requires_grad=False)
        x_new = torch.zeros((num_batches, 2*self.sparseDim, 1), device=self.device, requires_grad=False)
        
        if self.idctM != None:
            y_k = self.PhiWT(Y, src, self.idctM)
        else:
            y_k = torch.zeros((num_batches, 2*self.sparseDim, 1), device=self.device, requires_grad=False)

        t_old = torch.tensor(1, device=self.device, requires_grad=False).float()

        if self.isTrain or self.L == None: 
            # Calculate Libshitz constant
            T = torch.matmul(self.W.T, self.W)
            # eg, _ = torch.linalg.eig(T)
            eg = torch.linalg.eigvals(T)
            eg = torch.abs(eg)
            self.L = torch.max(eg)

        # print("L:", self.L)

        for itr in range(self.fistaIter):
            PhiWy_k = self.PhiW(y_k, src)
            res = Y - PhiWy_k
            PhiTres = self.PhiWT(res, src)

            x_new = softThresh(y_k + (PhiTres) / self.L, self.alpha/self.L * 0.5, device=self.device)
            # x_forward = y_k + (PhiTres) / self.L
            # x_new = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - (self.alpha/self.L * 0.5)))
            
            t_new = (1 + torch.sqrt(1 + 4 * t_old * t_old)) / 2
            
            # x_new = self.projection(x_new)

            y_k = x_new + ((t_old - 1) / t_new) * (x_new - x_old)

            x_old = x_new
            t_old = t_new

        # y_pred = torch.matmul(phiW, x_new)

        return x_new