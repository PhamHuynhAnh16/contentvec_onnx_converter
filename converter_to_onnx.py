import onnx
import torch
import onnxsim

from torch import nn
from transformers import HubertModel


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

class Contentvec(nn.Module):
    def __init__(self, embedders):
        super(Contentvec, self).__init__()
        hubert_model = HubertModelWithFinalProj.from_pretrained(embedders)
        hubert_model.to("cpu")

        hubert_model = hubert_model.float()
        self.hubert_model = hubert_model.eval()

    def forward(self, feats):
        feats = feats.view(-1)
        feats = feats.mean(-1) if feats.dim() == 2 else feats

        assert feats.dim() == 1, feats.dim()
        feats = self.hubert_model(feats.view(1, -1).to("cpu"))["last_hidden_state"]

        return self.hubert_model.final_proj(feats[0]).unsqueeze(0), feats

input_model = ["contentvec_base", "chinese_hubert_base", "japanese_hubert_base", "hubert_base", "korean_hubert_base", "portuguese_hubert_base"]

for m in input_model:
    output_model = m + ".onnx"

    torch.onnx.export(Contentvec(m), (torch.randn(1, 16384, dtype=torch.float32, device="cpu").clip(min=-1., max=1.).to("cpu")), output_model, do_constant_folding=False, opset_version=17, verbose=False, input_names=["feats"], output_names=["feats_9", "feats_12"], dynamic_axes={"feats": [1]})
    model, _ = onnxsim.simplify(output_model)

    onnx.save(model, output_model)