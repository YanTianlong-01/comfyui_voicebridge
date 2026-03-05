# model_utils.py
import os
import gc
import shutil
import numpy as np
import torch
import folder_paths
import comfy.model_management as mm

# ------------------------------------------------- Global Model Cache --------------------------------------------------

_ASR_MODEL_CACHE = {}   # {cache_key: model}
_TTS_MODEL_CACHE = {}   # {cache_key: model}


def _soft_empty_cache():
    """释放 GPU 显存和 Python 垃圾回收"""
    mm.soft_empty_cache()
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def unload_asr_model(cache_key=None):
    """卸载 ASR 缓存模型并释放显存"""
    global _ASR_MODEL_CACHE
    if cache_key and cache_key in _ASR_MODEL_CACHE:
        print(f"[VoiceBridge] Unloading ASR model: {cache_key}")
        del _ASR_MODEL_CACHE[cache_key]
    elif _ASR_MODEL_CACHE:
        print(f"[VoiceBridge] Unloading {len(_ASR_MODEL_CACHE)} cached ASR model(s)")
        _ASR_MODEL_CACHE.clear()
    _soft_empty_cache()


def unload_tts_model(cache_key=None):
    """卸载 TTS 缓存模型并释放显存"""
    global _TTS_MODEL_CACHE
    if cache_key and cache_key in _TTS_MODEL_CACHE:
        print(f"[VoiceBridge] Unloading TTS model: {cache_key}")
        del _TTS_MODEL_CACHE[cache_key]
    elif _TTS_MODEL_CACHE:
        print(f"[VoiceBridge] Unloading {len(_TTS_MODEL_CACHE)} cached TTS model(s)")
        _TTS_MODEL_CACHE.clear()
    _soft_empty_cache()

class ModelKey(str):
    """
    模型标识符类，继承自str，用于在节点间传递模型标识。
    主要目的是让ComfyUI显示为自定义类型而不是普通字符串。
    """
    pass


# ----------------------------------------------------- Qwen3-Models --------------------------------------------

# Register Qwen3-ASR models folder with ComfyUI
QWEN3_ASR_MODELS_DIR = os.path.join(folder_paths.models_dir, "Qwen3-ASR")
os.makedirs(QWEN3_ASR_MODELS_DIR, exist_ok=True)
folder_paths.add_model_folder_path("Qwen3-ASR", QWEN3_ASR_MODELS_DIR)

# Model repo mappings
QWEN3_ASR_MODELS = {
    "Qwen/Qwen3-ASR-1.7B": "Qwen3-ASR-1.7B",
    "Qwen/Qwen3-ASR-0.6B": "Qwen3-ASR-0.6B",
}

QWEN3_FORCED_ALIGNERS = {
    "None": None,
    "Qwen/Qwen3-ForcedAligner-0.6B": "Qwen3-ForcedAligner-0.6B",
}

# Supported languages
SUPPORTED_LANGUAGES = [
    "auto",
    "Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish",
    "Portuguese", "Indonesian", "Italian", "Korean", "Russian", "Thai",
    "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay", "Dutch", "Swedish",
    "Danish", "Finnish", "Polish", "Czech", "Filipino", "Persian", "Greek",
    "Hungarian", "Macedonian", "Romanian"
]


# Register Qwen3-TTS models folder with ComfyUI
QWEN3_TTS_MODELS_DIR = os.path.join(folder_paths.models_dir, "Qwen3-TTS")
os.makedirs(QWEN3_TTS_MODELS_DIR, exist_ok=True)
folder_paths.add_model_folder_path("Qwen3-TTS", QWEN3_TTS_MODELS_DIR)

# Register Qwen3-TTS prompts folder for voice embeddings
QWEN3_TTS_PROMPTS_DIR = os.path.join(folder_paths.models_dir, "Qwen3-TTS", "prompts")
os.makedirs(QWEN3_TTS_PROMPTS_DIR, exist_ok=True)
folder_paths.add_model_folder_path("Qwen3-TTS-Prompts", QWEN3_TTS_PROMPTS_DIR)

# Model repo mappings
QWEN3_TTS_MODELS = {
    # "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    # "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": "Qwen3-TTS-12Hz-1.7B-Base",
    # "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": "Qwen3-TTS-12Hz-0.6B-Base",
}

# Tokenizer repo mapping
QWEN3_TTS_TOKENIZERS = {
    "Qwen/Qwen3-TTS-Tokenizer-12Hz": "Qwen3-TTS-Tokenizer-12Hz",
}

# Language mapping dictionary to engine codes
LANGUAGE_MAP = {
    "Auto": "auto",
    "Chinese": "chinese",
    "English": "english",
    "Japanese": "japanese",
    "Korean": "korean",
    "French": "french",
    "German": "german",
    "Spanish": "spanish",
    "Portuguese": "portuguese",
    "Russian": "russian",
    "Italian": "italian",
}

# ----------------------------------------------------- Model Loader --------------------------------------------

def get_local_model_path(repo_id: str, type: str = "Qwen3-ASR") -> str:
    if type == "Qwen3-ASR":
        folder_name = QWEN3_ASR_MODELS.get(repo_id) or QWEN3_FORCED_ALIGNERS.get(repo_id) or repo_id.replace("/", "_")
        return os.path.join(QWEN3_ASR_MODELS_DIR, folder_name)
    elif type == "Qwen3-TTS":
        folder_name = QWEN3_TTS_MODELS.get(repo_id) or QWEN3_TTS_TOKENIZERS.get(repo_id) or repo_id.replace("/", "_")
        return os.path.join(QWEN3_TTS_MODELS_DIR, folder_name)


def migrate_cached_model(repo_id: str, target_path: str) -> bool:
    if os.path.exists(target_path) and os.listdir(target_path):
        return True
    
    hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    hf_model_dir = os.path.join(hf_cache, f"models--{repo_id.replace('/', '--')}")
    if os.path.exists(hf_model_dir):
        snapshots_dir = os.path.join(hf_model_dir, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            if snapshots:
                source = os.path.join(snapshots_dir, snapshots[0])
                print(f"[VoiceBridge] Migrating model from HuggingFace cache: {source} -> {target_path}")
                shutil.copytree(source, target_path, dirs_exist_ok=True)
                return True
    
    ms_cache = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub")
    ms_model_dir = os.path.join(ms_cache, repo_id.replace("/", os.sep))
    if os.path.exists(ms_model_dir):
        print(f"[VoiceBridge] Migrating model from ModelScope cache: {ms_model_dir} -> {target_path}")
        shutil.copytree(ms_model_dir, target_path, dirs_exist_ok=True)
        return True
    
    return False


def download_model_to_comfyui(repo_id: str, source: str, type: str = "Qwen3-ASR") -> str:
    target_path = get_local_model_path(repo_id, type)
    
    if migrate_cached_model(repo_id, target_path):
        print(f"[VoiceBridge] Model available at: {target_path}")
        return target_path
    
    os.makedirs(target_path, exist_ok=True)
    
    if source == "ModelScope":
        from modelscope import snapshot_download
        print(f"[VoiceBridge] Downloading {repo_id} from ModelScope to {target_path}...")
        snapshot_download(repo_id, local_dir=target_path)
    else:
        from huggingface_hub import snapshot_download
        print(f"[VoiceBridge] Downloading {repo_id} from HuggingFace to {target_path}...")
        snapshot_download(repo_id, local_dir=target_path)
    
    return target_path


def load_audio_input(audio_input):
    if audio_input is None:
        return None
        
    waveform = audio_input["waveform"]
    sr = audio_input["sample_rate"]
    
    wav = waveform[0]
    
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0)
    else:
        wav = wav.squeeze(0)
        
    return (wav.numpy().astype(np.float32), sr)