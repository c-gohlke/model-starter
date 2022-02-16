import torch


class BaseModel(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.linear0 = torch.nn.Linear(self.params["in0"], self.params["out0"])
        self.linear1 = torch.nn.Linear(self.params["in1"], self.params["out1"])
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, X):
        out = self.linear0(X)
        out = self.relu(out)
        out = self.linear1(out)
        out = self.sigmoid(out)
        return out