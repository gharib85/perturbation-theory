import cupy as cp
import torch

def jacobi(M):
    M_cp = cp.asarray(M)

    D, V = cp.cusolver.syevj(M_cp, with_eigen_vector = True)
    D, V = torch.as_tensor(D, device = 'cuda'), torch.as_tensor(V, device = 'cuda')
    return(torch.diag(D), V)
