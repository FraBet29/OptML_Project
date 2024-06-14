import torch
import numpy as np
import time


class Adasub(torch.optim.Optimizer):
    """
    Implements the AdaSub algorithm.
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
        eig, U = torch.linalg.eigh(H) # a bit slow
        alfa = 0
        if eig.min() < self.ro:
            alfa = self.ro - eig.min()
        return U, eig, alfa
        
        
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
                    flat_grad = grad.view(-1, 1)
                    p.subSpace = self.update_subspace(p.subSpace, flat_grad.data)
                    Q, _ = torch.linalg.qr(p.subSpace.data) # fast enough, acts on flattened data
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
                
                idx += 1

        
    def step(self, closure=None):
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