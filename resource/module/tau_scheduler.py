# -*- coding: utf-8 -*-


class TauScheduler:
    def __init__(self, origin_tau, mini_tau, n_step, s_step=0):
        self.origin_tau = origin_tau
        self.mini_tau = mini_tau
        self.n_step = n_step
        self.s_step = s_step
        self.step_interval = (self.mini_tau - self.origin_tau) / self.n_step

    def step_on(self, advance=False):
        if advance:
            self.s_step += 1
            return

        if self.s_step >= self.n_step:
            return self.mini_tau
        self.s_step += 1
        return self.s_step * self.step_interval + self.origin_tau

    def dump(self):
        self_dict = {
            "origin_tau": self.origin_tau,
            "mini_tau": self.mini_tau,
            "n_step": self.n_step,
            "s_step": self.s_step,
        }

        return self_dict

    @staticmethod
    def load(self_dict):
        return TauScheduler(self_dict["origin_tau"],
                            self_dict["mini_tau"],
                            self_dict["n_step"],
                            self_dict["s_step"])

    def self_load(self, self_dict):
        self.origin_tau = self_dict["origin_tau"]
        self.mini_tau = self_dict["mini_tau"]
        self.n_step = self_dict["n_step"]
        self.s_step = self_dict["s_step"]
        self.step_interval = (self.mini_tau - self.origin_tau) / self.n_step
