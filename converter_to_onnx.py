import onnx
import torch
import onnxslim

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
        self.hubert_model = hubert_model.float().eval()

    def forward(self, feats, output_layer):
        feats_out = torch.index_select(torch.stack(self.hubert_model.forward(feats, output_hidden_states=True, return_dict=True).hidden_states, dim=0), dim=0, index=output_layer.unsqueeze(0)).squeeze(0)
        feats_proj = self.hubert_model.final_proj(feats_out)
        return feats_out, feats_proj

input_model = ["contentvec_base", "chinese_hubert_base", "japanese_hubert_base", "hubert_base", "korean_hubert_base", "portuguese_hubert_base", "vietnamese_hubert_base", "spin-v1", "spin-v2"]
device = "cuda:0" if torch.cuda.is_available() else "cpu"

for m in input_model:
    output_model = m + ".onnx"
    print(f"Exporting: {m}...")

    feats = torch.randn(1, 16384, dtype=torch.float32, device=device).clip(min=-1., max=1.)
    layer = torch.tensor(12, device=device, dtype=torch.int64)

    torch.onnx.export(
        Contentvec(m).to(device), 
        (feats, layer), 
        output_model, 
        do_constant_folding=True,
        opset_version=17, 
        verbose=False, 
        input_names=["feats", "output_layer"], 
        output_names=["feats", "feats_proj"], 
        dynamic_axes={
            "feats": {1: "sequence_length"},
            "feats_out": {1: "sequence_length"},
            "feats_proj": {1: "sequence_length"},
        }
    )

    model = onnxslim.slim(output_model)
    onnx.save(model, output_model)
    print(f"Succeed: {output_model} saved and simplified!")