from matcha.models.components.flow_matching import CFM
import torch

class SVCCFM(torch.nn.Module):
    def __init__(self,model_config,attr_names):
        self.cfm=CFM(in_channels=100,out_channel=100)

    def forward(batch):
        self.batch.mels=None


