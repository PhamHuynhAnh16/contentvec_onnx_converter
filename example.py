import torch
import librosa
import onnxruntime

import numpy as np
import soundfile as sf

input_model = "contentvec_base.onnx"
input_audio = "test.wav"
output_layer = 12
final_proj = True

def get_providers():
    ort_providers = onnxruntime.get_available_providers()

    if "CUDAExecutionProvider" in ort_providers: providers = ["CUDAExecutionProvider"]
    elif "CoreMLExecutionProvider" in ort_providers: providers = ["CoreMLExecutionProvider"]
    else: providers = ["CPUExecutionProvider"]

    return providers

class HubertModelONNX:
    def __init__(
        self, 
        embedder_model_path, 
        providers
    ):
        sess_options = onnxruntime.SessionOptions()
        sess_options.log_severity_level = 3

        self.model = onnxruntime.InferenceSession(
            embedder_model_path, 
            sess_options=sess_options, 
            providers=providers
        )

        self._finalproj = False
        self.final_proj = self._final_proj

    def _final_proj(self, source):
        return source
    
    def extract_features(self, source, output_layer = None):
        device = source.device

        logits = self.model.run(
            [self.model.get_outputs()[0].name, self.model.get_outputs()[1].name], 
            {
                self.model.get_inputs()[0].name: source.float().detach().cpu().numpy(),
                self.model.get_inputs()[1].name: np.array(output_layer, dtype=np.int64),
            }
        )

        return [
            torch.as_tensor(
                logits[int(self._finalproj)], 
                dtype=torch.float32, 
                device=device
            )
        ]

hubert_model = HubertModelONNX(
    input_model, 
    get_providers()
)

def load_audio(file):
    audio, sr = sf.read(file)
    if len(audio.shape) > 1: audio = librosa.to_mono(audio.T)
    if sr != 16000: audio = librosa.resample(audio, orig_sr=sr, target_sr=16000, res_type="soxr_vhq")
    return audio.flatten()

feats = torch.from_numpy(load_audio(input_audio)).float()

if feats.dim() == 2: feats = feats.mean(-1)
assert feats.dim() == 1, feats.dim()
feats = feats.view(1, -1)

hubert_model._finalproj = final_proj
feats = hubert_model.extract_features(feats, output_layer)
feats = feats[0]

print("Shape:", feats.shape)
print(feats)