import numpy as np
import soundfile as sf
import librosa
import torchaudio

TARGET_SR = 16000

def _read_audio_any(x):

    if isinstance(x, dict) and "array" in x and "sampling_rate" in x:
        return x["array"], x["sampling_rate"]


    if isinstance(x, dict):
        path = x.get("path") or x.get("src") or x.get("file")
    else:
        path = x
    path = str(path)

    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        try:
            wav_t, sr = torchaudio.load(path)
            wav = wav_t.squeeze(0).numpy()
        except Exception as e:
            raise RuntimeError(f"오디오 로드 실패: {path}\n{e}")

    if isinstance(wav, np.ndarray) and wav.ndim > 1:
        wav = wav.mean(axis=1)

    if sr != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    return wav, sr

def prepare_dataset(batch, feature_extractor, tokenizer):
    wav, sr = _read_audio_any(batch["audio"])

    batch["input_features"] = feature_extractor(wav, sampling_rate=sr).input_features[0]


    batch["labels"] = tokenizer(batch["label"]).input_ids
    return batch
