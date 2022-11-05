# -*- coding: utf-8 -*-

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling
    '''

    def __init__(self,
                 optimizer,
                 init_lr,#0.5
                 d_model,#768
                 n_warmup_steps,#2000
                 gradient_overlay=1):

        self._optimizer = optimizer
        self.init_lr = init_lr
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0
        self.lr_factor = float(1) / gradient_overlay

    def step(self):
        """Step with the inner optimizer"""
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        """Zero out the gradients with the inner optimizer"""
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        """Learning rate scheduling per step"""
        self.n_steps += 1
        lr = self.init_lr * self._get_lr_scale() * self.lr_factor

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
