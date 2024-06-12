import torch
import numpy as np

'''

class Adasub(torch.optim.Optimizer):
    """
    Implements AdaSub algorithm.
    """

    def __init__(self, parameters, lr=1e-3, n_directions = 2, device = "cpu"):
        defaults = {"lr": lr, "device":device}
        super().__init__(parameters, defaults)
        
        self.state['step'] = 0
        self.n_directions = n_directions
        self.ro = 0.02
        self.device = device
        
        for p in self.get_params():
            p.update_value = 0

    def get_params(self):
        return (p for group in self.param_groups for p in group['params'] if p.requires_grad)

    @torch.no_grad()
    def update_subspace(self,old_subSpace,new_gradient):
        if self.state['step'] < self.n_directions:
            new_subSpace = torch.cat([old_subSpace,new_gradient],1)
        else:
            new_subSpace = torch.cat([old_subSpace[:,1:],new_gradient],1)
        return new_subSpace

    def Hessian_v(self,grad,vec,p):
        Hv = torch.autograd.grad(
            grad,
            p,
            grad_outputs=vec,
            only_inputs=True,
            retain_graph=True,
        )
        Hv_flaten = []
        for i in range(len(Hv)):
            Hv_flaten.append(Hv[i].reshape(-1))
        return torch.cat(Hv_flaten,0).view(-1,1)

    def Hessian_M(self,grad,matrix,p):
        H_M = []
        for i in range(matrix.shape[1]):
            H_M.append(
                self.Hessian_v(grad,matrix[:, i].view(-1, 1), p)
            )
        return torch.cat(H_M,1)  

    @torch.no_grad()
    def correction_Hessian(self, H):
        # Debug: Check if H contains inf or nan
        if torch.isnan(H).any() or torch.isinf(H).any():
            raise ValueError('Hessian matrix contains NaN or Inf')
        eig, U = torch.linalg.eigh(H)
        # Debug: Check if eig contains inf or nan
        if torch.isnan(eig).any() or torch.isinf(eig).any():
            raise ValueError(f'eig contains NaN or Inf')
        alfa = 0
        if eig.min() < self.ro:
            alfa = self.ro - eig.min()
        return U, eig, alfa

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if self.state['step'] == 0:
                    d_p = p.grad
                    with torch.no_grad():
                        p.add_(d_p, alpha=-group['lr'])
                    p.subSpace = p.grad.data.view(-1,1)

                else:
                    flat_grad = p.grad.view(-1,1)
                    # Debug: Check if grad contains inf or nan
                    if torch.isnan(flat_grad).any() or torch.isinf(flat_grad).any():
                        # count = torch.sum(torch.isnan(flat_grad) | torch.isinf(flat_grad)).item()
                        # raise ValueError(f'Gradient contains {count} NaN or Inf (out of {flat_grad.numel()} elements)')
                        # clip inf or nan values
                        flat_grad[torch.isnan(flat_grad)] = 0
                        flat_grad[torch.isposinf(flat_grad)] = 1
                        flat_grad[torch.isneginf(flat_grad)] = -1
                    p.subSpace = self.update_subspace(p.subSpace, flat_grad.data)
                    Q, _ = torch.linalg.qr(p.subSpace.data)
                    # Debug: Check if Q contains inf or nan
                    if torch.isnan(Q).any() or torch.isinf(Q).any():
                        # count = torch.sum(torch.isnan(Q) | torch.isinf(Q)).item()
                        # raise ValueError(f'Q contains {count} NaN or Inf (out of {Q.numel()} elements)')
                        # clip inf or nan values and normalize
                        Q[torch.isnan(Q)] = 0
                        Q[torch.isposinf(Q)] = 1
                        Q[torch.isneginf(Q)] = -1
                        Q /= torch.norm(Q, dim=0)
                    HQ = self.Hessian_M(flat_grad, Q, p)
                    # Debug: Check if HQ contains inf or nan
                    if torch.isnan(HQ).any() or torch.isinf(HQ).any():
                        # count = torch.sum(torch.isnan(HQ) | torch.isinf(HQ)).item()
                        # raise ValueError(f'Hessian matrix contains {count} NaN or Inf (out of {HQ.numel()} elements)')
                        # clip inf or nan values
                        HQ[torch.isnan(HQ)] = 0
                        HQ[torch.isposinf(HQ)] = 1
                        HQ[torch.isneginf(HQ)] = -1
                    U, eig, alfa = self.correction_Hessian(Q.T @ HQ)
                    y = U @ torch.diag(1 / (eig + alfa)) @ U.T @ (Q.T @ flat_grad)
                    d_p = Q @ y
                    p.update_value = d_p.view_as(p.data)


        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                with torch.no_grad():
                    p.add_(p.update_value, alpha=-group['lr'])


        self.state['step'] += 1

        return loss

'''

class Adasub(torch.optim.Optimizer):
    """
    Implements AdaSub algorithm.
    """

    def __init__(self, parameters, lr=1e-3, n_directions = 2, device = "cpu"):
        defaults = {"lr": lr, "device":device}
        super().__init__(parameters, defaults)
        self.state['step'] = 0
        self.n_directions = n_directions
        self.ro = 0.02
        self.device = device
        for p in self.get_params():
            p.update_value = 0


    def get_params(self):
        return (p for group in self.param_groups for p in group['params'] if p.requires_grad)


    @torch.no_grad()
    def update_subspace(self,old_subSpace,new_gradient):
        if self.state['step'] < self.n_directions:
            new_subSpace = torch.cat([old_subSpace,new_gradient],1)
        else:
            new_subSpace = torch.cat([old_subSpace[:,1:],new_gradient],1)
        return new_subSpace # size: (batch size, 2)


    @torch.no_grad()
    def correction_Hessian(self,H):
        # Debug: Check if H contains inf or nan
        if torch.isnan(H).any() or torch.isinf(H).any():
            raise ValueError('Hessian matrix contains NaN or Inf')
        eig, U = torch.linalg.eigh(H) # a bit slow
        # Debug: Check if eig contains inf or nan
        if torch.isnan(eig).any() or torch.isinf(eig).any():
            raise ValueError(f'eig contains NaN or Inf')
        alfa = 0
        if eig.min() < self.ro:
            alfa = self.ro - eig.min()
        return U, eig, alfa
        # try with eig instead
        # eig, U = torch.linalg.eig(H)
        # eig = eig.real
        # U = U.real
        # alfa = 0
        # if eig.min() < self.ro:
        #     alfa = self.ro - eig.min()
        # return U, eig, alfa
        
        
    @torch.no_grad()
    def set_update(self):
        """
        Computes the update for each trainable parameter.
        Goal: call autograd.grad only one time (NOT ONE TIME PER PARAMETER)
        """

        params = []
        grads = []
        flat_grads = []
        matrixs = [] # each matrix (Q) has the same size as the subspace
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if self.state['step'] == 0:
                    d_p = p.grad
                    with torch.no_grad():
                        p.add_(d_p, alpha=-group['lr'])
                    p.subSpace = p.grad.data.view(-1,1)
                else:
                    # p can be 1D or 2D for transformers, p.grad has the same shape as p
                    grad = p.grad
                    # Debug: Check if grad contains inf or nan
                    if torch.isnan(grad).any() or torch.isinf(grad).any():
                        # count = torch.sum(torch.isnan(grad) | torch.isinf(grad)).item()
                        # raise ValueError(f'Gradient contains {count} NaN or Inf (out of {grad.numel()} elements)')
                        # clip inf or nan values
                        grad[torch.isnan(grad)] = 0
                        grad[torch.isposinf(grad)] = 1
                        grad[torch.isneginf(grad)] = -1
                    # correct inf or nan values
                    # grad[torch.isnan(grad)] = 0
                    # grad[torch.isinf(grad)] = 0
                    flat_grad = grad.view(-1, 1)
                    p.subSpace = self.update_subspace(p.subSpace, flat_grad.data)
                    Q, _ = torch.linalg.qr(p.subSpace.data) # fast enough, acts on flattened data
                    # Debug: Check if Q contains inf or nan
                    if torch.isnan(Q).any() or torch.isinf(Q).any():
                        # count = torch.sum(torch.isnan(Q) | torch.isinf(Q)).item()
                        # raise ValueError(f'Q contains {count} NaN or Inf (out of {Q.numel()} elements)')
                        # clip inf or nan values and normalize
                        Q[torch.isnan(Q)] = 0
                        Q[torch.isposinf(Q)] = 1
                        Q[torch.isneginf(Q)] = -1
                        Q /= torch.norm(Q, dim=0)
                    Q = Q.view((*p.shape, self.n_directions)) # restore original shape
                    params.append(p)
                    grads.append(grad)
                    flat_grads.append(flat_grad)
                    matrixs.append(Q)
                    
        if len(params) == 0:
            return
        
        Hvs_basis = []
        HQs = []
            
        for i in range(self.n_directions): # process each basis vector separately
            # compute the Hessian applied to the subspace basis
            Hvs = torch.autograd.grad(grads,
                                      params,
                                      grad_outputs=[matrix[:,:,i] if matrix.dim()==3 else matrix[:,i] for matrix in matrixs],
                                      only_inputs=True, # only_inputs argument is deprecated and is ignored now (defaults to True)
                                      retain_graph=True
                                     )
            # Debug: Check if Hvs contains inf or nan
            for H in Hvs:
                if torch.isnan(H).any() or torch.isinf(H).any():
                    # count = torch.sum(torch.isnan(H) | torch.isinf(H)).item()
                    # raise ValueError(f'Hessian-vector product contains {count} NaN or Inf (out of {H.numel()} elements)')
                    # clip inf or nan values
                    H[torch.isnan(H)] = 0
                    H[torch.isposinf(H)] = 1
                    H[torch.isneginf(H)] = -1
            # Hvs is a list of Hessian-vector products for the i-th direction
            Hvs_basis.append(Hvs)
        
        for idx in range(len(params)):
            HQs.append(torch.cat([Hvs[idx].view(-1, 1) for Hvs in Hvs_basis], 1)) # flattened
            
        idx = 0
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                U, eig, alfa = self.correction_Hessian(matrixs[idx].view(-1, self.n_directions).T @ HQs[idx])
                y = U @ torch.diag(1 / (eig + alfa)) @ U.T @ (matrixs[idx].view(-1, self.n_directions).T @ flat_grads[idx])
                d_p = matrixs[idx].view(-1, self.n_directions) @ y
                p.update_value = d_p.view_as(p.data)
                # print(p.update_value)
                
                idx += 1

        
    def step(self, closure=None):
        
        # print('Running AdaSub step...')
        # print(torch.cuda.memory_allocated())
        
        loss = None
        if closure is not None:
            loss = closure()
        
        
        self.set_update()


        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                with torch.no_grad():
                    p.add_(p.update_value, alpha=-group['lr'])
                    

        self.state['step'] += 1

        return loss

