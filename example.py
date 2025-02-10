import torch
import librosa
import onnxruntime
import soundfile as sf

input_model = "contentvec_base.onnx"
input_audio = "test.wav"
output_layer = 12

def get_providers():
    ort_providers = onnxruntime.get_available_providers()

    if "CUDAExecutionProvider" in ort_providers: providers = ["CUDAExecutionProvider"]
    elif "CoreMLExecutionProvider" in ort_providers: providers = ["CoreMLExecutionProvider"]
    else: providers = ["CPUExecutionProvider"]

    return providers

def extract_features(model, feats, layer):
    return torch.as_tensor(model.run(["feats_9", "feats_12"], {"feats": feats.detach().cpu().numpy()})[0 if layer == 9 else 1], dtype=torch.float32, device=feats.device)

def load_audio(file):
    audio, sr = sf.read(file)
    if len(audio.shape) > 1: audio = librosa.to_mono(audio.T)
    if sr != 16000: audio = librosa.resample(audio, orig_sr=sr, target_sr=16000, res_type="soxr_vhq")
    return audio.flatten()

sess_options = onnxruntime.SessionOptions()
sess_options.log_severity_level = 3
models = onnxruntime.InferenceSession(input_model, sess_options=sess_options, providers=get_providers())
feats = torch.from_numpy(load_audio(input_audio)).float()

if feats.dim() == 2: feats = feats.mean(-1)
assert feats.dim() == 1, feats.dim()

feats = feats.view(1, -1)
feats = extract_features(models, feats, output_layer)

print("Shape:", feats.shape)
print(feats)