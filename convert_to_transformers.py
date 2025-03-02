import torch
import logging

from fairseq import checkpoint_utils
from transformers import HubertConfig, HubertModel

logging.getLogger("fairseq").setLevel(logging.WARNING)
logging.getLogger("torch.distributed.nn.jit.instantiator").setLevel(logging.WARNING)

model_path = ["contentvec_base.pt", "chinese_hubert_base.pt", "japanese_hubert_base.pt", "hubert_base.pt", "korean_hubert_base.pt", "portuguese_hubert_base.pt"]

class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = torch.nn.Linear(config.hidden_size, config.classifier_proj_size)

for m in model_path:
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task([m], suffix="")
    model = models[0]
    model.eval()
    model.eval()

    hubert = HubertModelWithFinalProj(HubertConfig())
    mapping = {"masked_spec_embed": "mask_emb", "encoder.layer_norm.bias": "encoder.layer_norm.bias", "encoder.layer_norm.weight": "encoder.layer_norm.weight", "encoder.pos_conv_embed.conv.bias": "encoder.pos_conv.0.bias", "encoder.pos_conv_embed.conv.weight_g": "encoder.pos_conv.0.weight_g", "encoder.pos_conv_embed.conv.weight_v": "encoder.pos_conv.0.weight_v", "feature_projection.layer_norm.bias": "layer_norm.bias", "feature_projection.layer_norm.weight": "layer_norm.weight", "feature_projection.projection.bias": "post_extract_proj.bias", "feature_projection.projection.weight": "post_extract_proj.weight", "final_proj.bias": "final_proj.bias", "final_proj.weight": "final_proj.weight"}

    for layer in range(12):
        for j in ["q", "k", "v"]:
            mapping[f"encoder.layers.{layer}.attention.{j}_proj.weight"] = f"encoder.layers.{layer}.self_attn.{j}_proj.weight"
            mapping[f"encoder.layers.{layer}.attention.{j}_proj.bias"] = f"encoder.layers.{layer}.self_attn.{j}_proj.bias"

        mapping[f"encoder.layers.{layer}.final_layer_norm.bias"] = f"encoder.layers.{layer}.final_layer_norm.bias"
        mapping[f"encoder.layers.{layer}.final_layer_norm.weight"] = f"encoder.layers.{layer}.final_layer_norm.weight"
        mapping[f"encoder.layers.{layer}.layer_norm.bias"] = f"encoder.layers.{layer}.self_attn_layer_norm.bias"
        mapping[f"encoder.layers.{layer}.layer_norm.weight"] = f"encoder.layers.{layer}.self_attn_layer_norm.weight"
        mapping[f"encoder.layers.{layer}.attention.out_proj.bias"] = f"encoder.layers.{layer}.self_attn.out_proj.bias"
        
        mapping[f"encoder.layers.{layer}.attention.out_proj.weight"] = f"encoder.layers.{layer}.self_attn.out_proj.weight"
        mapping[f"encoder.layers.{layer}.feed_forward.intermediate_dense.bias"] = f"encoder.layers.{layer}.fc1.bias"
        mapping[f"encoder.layers.{layer}.feed_forward.intermediate_dense.weight"] = f"encoder.layers.{layer}.fc1.weight"
        mapping[f"encoder.layers.{layer}.feed_forward.output_dense.bias"] = f"encoder.layers.{layer}.fc2.bias"
        mapping[f"encoder.layers.{layer}.feed_forward.output_dense.weight"] = f"encoder.layers.{layer}.fc2.weight"

    for layer in range(7):
        mapping[f"feature_extractor.conv_layers.{layer}.conv.weight"] = f"feature_extractor.conv_layers.{layer}.0.weight"

        if layer != 0: continue

        mapping[f"feature_extractor.conv_layers.{layer}.layer_norm.weight"] = f"feature_extractor.conv_layers.{layer}.2.weight"
        mapping[f"feature_extractor.conv_layers.{layer}.layer_norm.bias"] = f"feature_extractor.conv_layers.{layer}.2.bias"

    hf_keys = set(hubert.state_dict().keys())
    fair_keys = set(model.state_dict().keys())

    hf_keys -= set(mapping.keys())
    fair_keys -= set(mapping.values())

    for i, j in zip(sorted(hf_keys), sorted(fair_keys)):
        print(i, j)

    print(hf_keys, fair_keys)
    print(len(hf_keys), len(fair_keys))

    new_state_dict = {}
    for k, v in mapping.items():
        new_state_dict[k] = model.state_dict()[v]

    hubert.load_state_dict(new_state_dict, strict=False)
    hubert.eval()

    with torch.no_grad():
        new_input = torch.randn(1, 16384)

        assert torch.allclose(hubert(new_input, output_hidden_states=True)["hidden_states"][12], model.extract_features(**{"source": new_input, "padding_mask": torch.zeros(1, 16384, dtype=torch.bool), "output_layer": 12})[0], atol=1e-3)

    print("Đã vượt qua kiểm tra. Bắt đầu xuất mô hình")

    hubert.save_pretrained(m.replace('.pt', ''))
    torch.save(hubert.state_dict(), f"{m.replace('.pt', '')}/pytorch_model.bin")
    
    print("Đã lưu mô hình!")