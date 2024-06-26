# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 11:27:34 2017
@author: Ashish Katiyar
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 09 14:38:12 2017
@author: Ashish Katiyar
"""

import math

import torch

from torch.optim import Optimizer





class Nadam(Optimizer):

    """Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    """



    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,

                 weight_decay=0):

        defaults = dict(lr=lr, betas=betas, eps=eps,

                        weight_decay=weight_decay)

        super(Nadam, self).__init__(params, defaults)



    def step(self, closure=None):

        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None

        if closure is not None:

            loss = closure()



        for group in self.param_groups:

            for p in group['params']:

                if p.grad is None:

                    continue

                grad = p.grad.data

                if grad.is_sparse:

                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')



                state = self.state[p]



                # State initialization

                if len(state) == 0:

                    state['step'] = 0

                    # Exponential moving average of gradient values

                    state['exp_avg'] = torch.zeros_like(p.data)

                    # Exponential moving average of squared gradient values

                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['prod_mu_t'] = 1.



                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']



                state['step'] += 1



                if group['weight_decay'] != 0:

                    grad = grad.add(group['weight_decay'], p.data)



                # Decay the first and second moment running average coefficient

                
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                
                
                # prod_mu_t = 1
                mu_t = beta1*(1 - 0.5*0.96**(state['step']/250))
                mu_t_1 = beta1*(1 - 0.5*0.96**((state['step']+1)/250))
                prod_mu_t = state['prod_mu_t'] * mu_t
                prod_mu_t_1 = prod_mu_t * mu_t_1

                state['prod_mu_t'] = prod_mu_t
                # for i in range(state['step']):
                #     mu_t = beta1*(1 - 0.5*0.96**(i/250))
                #     mu_t_1 = beta1*(1 - 0.5*0.96**((i+1)/250))
                #     prod_mu_t = prod_mu_t * mu_t
                #     prod_mu_t_1 = prod_mu_t * mu_t_1
                    
                g_hat = grad/(1-prod_mu_t) 
                m_hat = exp_avg / (1-prod_mu_t_1)
                
                m_bar = (1-mu_t)*g_hat + mu_t_1*m_hat
                    
                exp_avg_sq_hat = exp_avg_sq/(1 - beta2 ** state['step'])
                
                denom = exp_avg_sq_hat.sqrt().add_(group['eps'])



                bias_correction1 = 1 - beta1 ** state['step']

                bias_correction2 = 1 - beta2 ** state['step']

                #step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                step_size = group['lr']

                p.data.addcdiv_(-step_size, m_bar, denom)
                #p.data.addcdiv_(-step_size, exp_avg, denom)



        return loss
