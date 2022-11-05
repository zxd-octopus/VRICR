import sys
import torch
import torch.nn as nn

import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self,
                 device):
        super(BaseModel, self).__init__()
        self.device = device

    def get_name(self):
        return self.__class__.__name__

    def load(self, filename,strict = True):
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location=self.device)
        try:
            self.load_state_dict(state_dict, strict=strict)
        except (Exception,) as e:
            print(e)
            confirm = ""
            while confirm not in ["Y", "N"]:
                confirm = input("load in strict mode, error occur, force load [y or Y for YES, n or N for NO] >>>")
                confirm = confirm.strip().upper()

            if confirm.upper() != "Y":
                sys.exit()

            self.load_state_dict(state_dict, strict=False)

    def dump(self, filename):
        torch.save(self.state_dict(), filename)

    def loss(self, *inputs):
        outputs, targets = inputs[0], inputs[1]
        outputs = outputs.to(self.device)
        targets = targets.to(self.device)
        return F.nll_loss(torch.log(outputs+1e-20), targets)


if __name__ == '__main__':
    base_model = BaseModel(None)
    print(base_model.get_name())