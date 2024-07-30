from torch import nn

class FiLMCondition(nn.Module):
    def __init__(self, cond_channels, out_channels):
        super(FiLMCondition, self).__init__()
        self.proj_add = nn.Linear(cond_channels, out_channels)
        self.proj_mult = nn.Linear(cond_channels, out_channels)
        nn.init.zeros_(self.proj_add.weight)
        nn.init.zeros_(self.proj_add.bias)
        nn.init.zeros_(self.proj_mult.weight)
        nn.init.zeros_(self.proj_mult.bias)

    def forward(self, x, cond):
        # x.shape: (batch size, channel, height, width)
        shift = self.proj_add(cond)
        scale = self.proj_mult(cond)
        B, C = shift.shape
        shift = shift.view(B, C, 1, 1) # (b c) -> (b c 1 1)
        B, C = scale.shape
        scale = scale.view(B, C, 1, 1) # (b c) -> (b c 1 1)
        x = (1 + scale)*x + shift
        return x



