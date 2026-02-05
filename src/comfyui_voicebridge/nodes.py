# voice bridge nodes
import re
import os
import shutil
import torch
import numpy as np
import folder_paths
import comfy.model_management as mm
from qwen_asr import Qwen3ASRModel
from qwen_tts import Qwen3TTSModel
from openai import OpenAI

# for srt2audio
import soundfile as sf
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple
import subprocess
from pydub import AudioSegment

# ----------------------------------------------------- SRT to Audio Process --------------------------------------------

@dataclass
class SubtitleEntry:
    """字幕条目"""
    index: int
    start_time_ms: int
    end_time_ms: int
    text: str
    audio_path: Optional[str] = None
    audio_duration_ms: Optional[int] = None


def time_to_ms(time_str: str) -> int:
    time_str = time_str.strip().replace(',', '.')
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return int((hours * 3600 + minutes * 60 + seconds) * 1000)


def ms_to_time(ms: int) -> str:
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    seconds = ms // 1000
    milliseconds = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def parse_srt_string(srt_content: str) -> List[SubtitleEntry]:
    pattern = r'(\d+)\n([\d:,]+)\s*-->\s*([\d:,]+)\n(.+?)(?=\n\n|\Z)'
    matches = re.findall(pattern, srt_content, re.DOTALL)
    
    entries = []
    for match in matches:
        index = int(match[0])
        start_time = time_to_ms(match[1])
        end_time = time_to_ms(match[2])
        text = match[3].strip().replace('\n', ' ')
        entries.append(SubtitleEntry(
            index=index,
            start_time_ms=start_time,
            end_time_ms=end_time,
            text=text
        ))
    
    return entries


def save_srt_string(entries: List[SubtitleEntry]) -> str:
    srt_lines = []
    for entry in entries:
        srt_lines.append(f"{entry.index}")
        srt_lines.append(f"{ms_to_time(entry.start_time_ms)} --> {ms_to_time(entry.end_time_ms)}")
        srt_lines.append(f"{entry.text}")
        srt_lines.append("")
    
    return '\n'.join(srt_lines)


def get_audio_duration_ms(audio_path: str) -> int:
    audio = AudioSegment.from_file(audio_path)
    return len(audio)


def speed_up_audio(input_path: str, output_path: str, speed_factor: float):
    if speed_factor > 2.0:
        atempo_filters = []
        remaining = speed_factor
        while remaining > 2.0:
            atempo_filters.append("atempo=2.0")
            remaining /= 2.0
        atempo_filters.append(f"atempo={remaining:.4f}")
        filter_str = ",".join(atempo_filters)
    else:
        filter_str = f"atempo={speed_factor:.4f}"
    
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-filter:a', filter_str,
        output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def merge_audio_files(entries: List[SubtitleEntry], total_duration_ms: int) -> Tuple[np.ndarray, int]:
    base_audio = AudioSegment.silent(duration=total_duration_ms)
    
    for entry in entries:
        audio = AudioSegment.from_file(entry.audio_path)
        position = entry.start_time_ms
        base_audio = base_audio.overlay(audio, position=position)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    
    base_audio.export(tmp_path, format="wav")
    
    wav_data, sample_rate = sf.read(tmp_path)
    
    try:
        os.unlink(tmp_path)
    except:
        pass

    if wav_data.ndim == 1:
        wav_data = wav_data.reshape(1,1,-1)
    elif wav_data.ndim == 2:
        wav_data = wav_data.reshape(1,1,-1)
    wav_tensor = torch.from_numpy(wav_data).float()
    
    return wav_tensor, sample_rate

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
                print(f"Migrating model from HuggingFace cache: {source} -> {target_path}")
                shutil.copytree(source, target_path, dirs_exist_ok=True)
                return True
    
    ms_cache = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub")
    ms_model_dir = os.path.join(ms_cache, repo_id.replace("/", os.sep))
    if os.path.exists(ms_model_dir):
        print(f"Migrating model from ModelScope cache: {ms_model_dir} -> {target_path}")
        shutil.copytree(ms_model_dir, target_path, dirs_exist_ok=True)
        return True
    
    return False


def download_model_to_comfyui(repo_id: str, source: str, type: str = "Qwen3-ASR") -> str:
    target_path = get_local_model_path(repo_id, type)
    
    if migrate_cached_model(repo_id, target_path):
        print(f"Model available at: {target_path}")
        return target_path
    
    os.makedirs(target_path, exist_ok=True)
    
    if source == "ModelScope":
        from modelscope import snapshot_download
        print(f"Downloading {repo_id} from ModelScope to {target_path}...")
        snapshot_download(repo_id, local_dir=target_path)
    else:
        from huggingface_hub import snapshot_download
        print(f"Downloading {repo_id} from HuggingFace to {target_path}...")
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

# ----------------------------------------------------------------------- Qwen3 ASR -----------------------------------------------------------------


class Qwen3ASRLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "repo_id": (list(QWEN3_ASR_MODELS.keys()), {"default": "Qwen/Qwen3-ASR-1.7B"}),
                "source": (["HuggingFace", "ModelScope"], {"default": "HuggingFace"}),
                "precision": (["fp16", "bf16", "fp32"], {"default": "bf16"}),
                "attention": (["auto", "flash_attention_2", "sdpa", "eager"], {"default": "auto"}),
            },
            "optional": {
                "max_new_tokens": ("INT", {"default": 256, "min": 1, "max": 4096, "tooltip": "The maximum number of tokens to generate in the transcription."}),
                "forced_aligner": (list(QWEN3_FORCED_ALIGNERS.keys()), {"default": "None"}),
                "local_model_path_asr": ("STRING", {"default": "", "multiline": False, "tooltip": "The local path to the ASR model. If provided, the model will be loaded from this path instead of downloading it from HuggingFace or ModelScope."}),
                "local_model_path_fa": ("STRING", {"default": "", "multiline": False, "tooltip": "The local path to the forced aligner model. If provided, the model will be loaded from this path instead of downloading it from HuggingFace or ModelScope."}),
            }
        }

    RETURN_TYPES = ("QWEN3_ASR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "VoiceBridge"

    def load_model(self, repo_id, source, precision, attention, max_new_tokens=256, forced_aligner="None", local_model_path_asr="", local_model_path_fa=""):
        device = mm.get_torch_device()
        
        dtype = torch.float32
        if precision == "bf16":
            if device.type == "mps":
                dtype = torch.float16
                print("Note: Using fp16 on MPS (bf16 has limited support)")
            else:
                dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16
            
        if local_model_path_asr and local_model_path_asr.strip() != "":
            model_path = local_model_path_asr.strip()
            model_path = os.path.join(folder_paths.models_dir, model_path)
            print(f"Loading from local path: {model_path}")
        else:
            local_path = get_local_model_path(repo_id, "Qwen3-ASR")
            if os.path.exists(local_path) and os.listdir(local_path):
                model_path = local_path
                print(f"Loading from ComfyUI models folder: {model_path}")
            else:
                model_path = download_model_to_comfyui(repo_id, source, "Qwen3-ASR")
        
        model_kwargs = dict(
            dtype=dtype,
            device_map=str(device),
            max_inference_batch_size=32,
            max_new_tokens=max_new_tokens,
        )
        if attention != "auto":
            model_kwargs["attn_implementation"] = attention
            
        if forced_aligner and forced_aligner != "None":
            aligner_local = get_local_model_path(forced_aligner, "Qwen3-ASR")
            if local_model_path_fa and local_model_path_fa.strip() != "":
                aligner_local = local_model_path_fa.strip()
                aligner_local = os.path.join(folder_paths.models_dir, aligner_local)
                print(f"Loading from local path: {aligner_local}")
            elif not (os.path.exists(aligner_local) and os.listdir(aligner_local)):
                aligner_local = download_model_to_comfyui(forced_aligner, source, "Qwen3-ASR")
            print(f"Loading Force Aligner from local path: {aligner_local}")

            model_kwargs["forced_aligner"] = aligner_local
            model_kwargs["forced_aligner_kwargs"] = dict(
                dtype=dtype,
                device_map=str(device),
            )
            if attention != "auto":
                model_kwargs["forced_aligner_kwargs"]["attn_implementation"] = attention
        
        print(f"Loading Qwen3-ASR model from {model_path}...")
        model = Qwen3ASRModel.from_pretrained(model_path, **model_kwargs)
        
        return (model,)


class Qwen3ASRTranscribe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN3_ASR_MODEL",),
                "audio": ("AUDIO",),
            },
            "optional": {
                "language": (SUPPORTED_LANGUAGES, {"default": "auto"}),
                "context": ("STRING", {"default": "", "multiline": True}),
                "return_timestamps": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LIST", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("forced_aligns", "text", "language", "timestamps",)
    FUNCTION = "transcribe"
    CATEGORY = "VoiceBridge"

    def transcribe(self, model, audio, language="auto", context="", return_timestamps=False):
        audio_data = load_audio_input(audio)
        if audio_data is None:
            return ("", "", "")
        
        lang = None if language == "auto" else language
        ctx = context if context.strip() else ""
        
        results = model.transcribe(
            audio=audio_data,
            language=lang,
            context=ctx if ctx else None,
            return_time_stamps=return_timestamps,
        )
        
        result = results[0]
        text = result.text
        detected_lang = result.language or ""
        
        timestamps_str = ""
        if return_timestamps and result.time_stamps:
            ts_lines = []
            for ts in result.time_stamps:
                ts_lines.append(f"{ts.start_time:.2f}-{ts.end_time:.2f}: {ts.text}")
            timestamps_str = "\n".join(ts_lines)
        
        return (result.time_stamps, text, detected_lang, timestamps_str)


# ----------------------------------------------------------------------- Qwen3 TTS -----------------------------------------------------------------
class Qwen3TTSLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "repo_id": (list(QWEN3_TTS_MODELS.keys()), {"default": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"}),
                "source": (["HuggingFace", "ModelScope"], {"default": "HuggingFace"}),
                "precision": (["fp16", "bf16", "fp32"], {"default": "bf16"}),
                "attention": (["auto", "flash_attention_2", "sdpa", "eager"], {"default": "auto"}),
            },
            "optional": {
                "local_model_path": ("STRING", {"default": "", "multiline": False}),
            }
        }
    RETURN_TYPES = ("QWEN3_TTS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "VoiceBridge"

    def load_model(self, repo_id, source, precision, attention, local_model_path=""):
        device = mm.get_torch_device()
        
        dtype = torch.float32
        if precision == "bf16":
            if device.type == "mps":
                dtype = torch.float16
                print("Note: Using fp16 on MPS (bf16 has limited support)")
            else:
                dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16
            
        if local_model_path and local_model_path.strip() != "":
            model_path = local_model_path.strip()
            model_path = os.path.join(folder_paths.models_dir, model_path)
            print(f"Loading from local path: {model_path}")
        else:
            local_path = get_local_model_path(repo_id, "Qwen3-TTS")
            if os.path.exists(local_path) and os.listdir(local_path):
                model_path = local_path
                print(f"Loading from ComfyUI models folder: {model_path}")
            else:
                model_path = download_model_to_comfyui(repo_id, source, "Qwen3-TTS")
        
        model_kwargs = dict(
            dtype=dtype,
            device_map=str(device),
        )
        if attention != "auto":
            model_kwargs["attn_implementation"] = attention
        
        print(f"Loading Qwen3-TTS model from {model_path}...")
        model = Qwen3TTSModel.from_pretrained(model_path, **model_kwargs)
        
        return (model,)
    
class VoiceClonePrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN3_TTS_MODEL",),
                "ref_audio": ("AUDIO", {"tooltip": "Reference audio (ComfyUI Audio)"}),
                "ref_text": ("STRING", {"multiline": True, "default": "", "placeholder": "Reference audio text (highly recommended for better quality)"}),
            }
        }
    
    RETURN_TYPES = ("VOICE_CLONE_PROMPT",)
    RETURN_NAMES = ("voice_clone_prompt",)
    FUNCTION = "create_prompt"
    CATEGORY = "VoiceBridge"

    def create_prompt(self, model, ref_audio, ref_text):
        audio_data = load_audio_input(ref_audio)
        if audio_data is None:
            return ("",)
        
        result = model.create_voice_clone_prompt(
            ref_audio=audio_data,
            ref_text=ref_text,
            x_vector_only_mode=False,
        )
        print("Voice Clone Prompt created successfully!")
        return (result,)
    
class SRTToAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN3_TTS_MODEL",),
                "srt_string": ("STRING", {"multiline": True, "default": "", "placeholder": "SRT text"}),
                "voice_clone_prompt": ("VOICE_CLONE_PROMPT",),
            },
            "optional": {
                "language": (SUPPORTED_LANGUAGES, {"default": "auto"}),
                "tempo_limit": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 2.0, "step": 0.1, "tooltip": "Maximum speed-up factor for audio that exceeds subtitle duration"}),
                "batch_size": ("INT", {"default": 10, "min": 1, "max": 50, "tooltip": "Number of subtitles to process in each batch"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "adjusted_srt")
    FUNCTION = "convert_srt_to_audio"
    CATEGORY = "VoiceBridge"

    def convert_srt_to_audio(
        self, 
        model, 
        srt_string, 
        voice_clone_prompt, 
        language="auto",
        tempo_limit: float = 1.5,
        batch_size: int = 10
    ):
        """
        Convert SRT subtitles to audio
        
        Args:
            model: Qwen3-TTS model
            srt_string: subtitle string in SRT format
            voice_clone_prompt: voice cloning prompt
            language: language (default auto)
            tempo_limit: maximum acceleration multiple limit
            batch_size: batch processing size
        
        Returns:
            audio: audio in ComfyUI format (waveform, sample_rate)
            adjusted_srt: adjusted SRT string
        """
        if not srt_string or not srt_string.strip():
            print("Error: Empty SRT string provided")
            return ({"waveform": np.array([[0.0]]), "sample_rate": 16000}, "")
        
        print(f"Parsing SRT content ({len(srt_string)} chars)...")
        entries = parse_srt_string(srt_string)
        print(f"Found {len(entries)} subtitle entries")
        
        if len(entries) == 0:
            print("Error: No valid subtitle entries found in SRT")
            return ({"waveform": np.array([[0.0]]), "sample_rate": 16000}, "")
        
        temp_dir = tempfile.mkdtemp(prefix="srt_audio_")
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Using temp directory: {temp_dir}")
        
        lang = LANGUAGE_MAP.get(language, "auto")
        
        try:
            print("Starting audio generation...")
            for i in range(0, len(entries), batch_size):
                batch = entries[i:i+batch_size]
                texts = [e.text for e in batch]
                paths = [os.path.join(temp_dir, f"audio_{e.index:04d}.wav") for e in batch]
                
                print(f"  Generating batch {i//batch_size + 1}: entries {i+1}-{min(i+batch_size, len(entries))}")
                
                wavs, sr = model.generate_voice_clone(
                    text=texts,
                    language=[lang] * len(texts) if lang else [lang] * len(texts),
                    voice_clone_prompt=voice_clone_prompt,
                )
                
                for wav, path in zip(wavs, paths):
                    sf.write(path, wav, sr)
                
                for entry, path in zip(batch, paths):
                    entry.audio_path = path
                    entry.audio_duration_ms = get_audio_duration_ms(path)
                
                torch.cuda.empty_cache()
            
            print("Processing duration mismatches...")
            self._process_duration(entries, temp_dir, tempo_limit)
            
            last_entry = entries[-1]
            total_duration = last_entry.start_time_ms + last_entry.audio_duration_ms + 1000
            
            # Synthesize the final audio
            print("Merging audio files...")
            wav_tensor, sample_rate = merge_audio_files(entries, total_duration)

            # Prepare audio output in ComfyUI format
            audio_output = {
                "waveform": wav_tensor,
                "sample_rate": sample_rate
            }

            
            # Generate the adjusted SRT string
            adjusted_srt = save_srt_string(entries)
            print(f"Completed! Output audio: {wav_tensor.shape[-1]} samples at {sample_rate}Hz")
            
            return (audio_output, adjusted_srt)
            
        except Exception as e:
            print(f"Error during audio conversion: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Clean up the temporary directory
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
            
            # Return a silent audio as fallback
            silent_audio = {"waveform": torch.zeros(1, 16000), "sample_rate": 16000}
            return (silent_audio, "")
    
    def _process_duration(self, entries: List[SubtitleEntry], temp_dir: str, tempo_limit: float):
        """
        Handling the issue where the audio duration does not match the subtitle duration
        
        Args:
            entries: List of subtitle entries
            temp_dir: Temporary file directory
            tempo_limit: Maximum acceleration multiple
        """
        for i, entry in enumerate(entries):
            subtitle_duration = entry.end_time_ms - entry.start_time_ms
            audio_duration = entry.audio_duration_ms
            
            # Calculate the available time (the time until the next subtitle starts)
            if i < len(entries) - 1:
                available_time = entries[i+1].start_time_ms - entry.start_time_ms
            else:
                available_time = subtitle_duration + 5000  # 5 more seconds for safety
            
            print(f"  [{entry.index}] Subtitle: {subtitle_duration}ms, Audio: {audio_duration}ms, Available: {available_time}ms")
            
            if audio_duration <= subtitle_duration:
                # If the audio is shorter than the subtitle -> adjust the end time of the subtitle
                new_end_time = entry.start_time_ms + audio_duration
                print(f"       -> Audio shorter, adjusting end time: {entry.end_time_ms}ms -> {new_end_time}ms")
                entry.end_time_ms = new_end_time
                
            elif audio_duration > available_time:
                # If the audio is longer than the available time -> speed up the audio
                speed_factor = audio_duration / available_time
                
                if speed_factor > tempo_limit:
                    print(f"       -> Warning: Required speed-up {speed_factor:.2f}x exceeds limit {tempo_limit}x, using limit")
                    speed_factor = tempo_limit
                
                print(f"       -> Audio too long, speeding up by {speed_factor:.2f}x")
                
                # Speed up audio
                sped_up_path = os.path.join(temp_dir, f"audio_{entry.index:04d}_sped.wav")
                speed_up_audio(entry.audio_path, sped_up_path, speed_factor)
                entry.audio_path = sped_up_path
                entry.audio_duration_ms = get_audio_duration_ms(sped_up_path)
                
                # Update the end time of the subtitle
                entry.end_time_ms = entry.start_time_ms + entry.audio_duration_ms
                
            else:
                # The audio is within the available range but exceeds the original subtitle duration -> only adjust the subtitle end time
                print(f"       -> Audio slightly longer than subtitle but within available time, adjusting end time")
                entry.end_time_ms = entry.start_time_ms + audio_duration

# ---------------------------------------------------------------- Voice Bridge Linker -----------------------------------------------------------

DELIMITERS = ['，', '。', '！', '？', '；', '：', 
              ',',  '.',  '!',  '?',  ';',  ':', 
              '\n', '\r', '\t'
            ]



def split_string_regex(text, delimiters):
    pattern = '|'.join(re.escape(d) for d in delimiters)
    segments = re.split(f'({pattern})', text)
    result = []
    current = ""
    for segment in segments:
        if segment in delimiters:
            result.append((current + segment).strip())
            current = ""
        else:
            current += segment
    if current:
        result.append(current.strip())
    return [seg[:-1] for seg in result if seg]

def is_english_char(char):
    return char.isascii() and char.isalpha()

def get_seg_timestamps(segments, forced_aligns):
    srt_time_stamps = []
    word_index = 0

    for i, segment in enumerate(segments):
        start_char = segment[0]
        end_char = segment[-1]
        # ------ check if it's english word ------ #
        if is_english_char(start_char):
            start_char = segment.split()[0]
        if is_english_char(end_char):
            end_char = segment.split()[-1]
        if start_char == end_char == segment:
            srt_time_stamps.append((forced_aligns[word_index].start_time, forced_aligns[word_index].end_time))
            word_index += 1
            continue

        start_time = forced_aligns[word_index].start_time
        end_char_count = segment.count(end_char)
        if end_char_count == 1:
            while(forced_aligns[word_index].text != end_char):
                word_index += 1
        else:
            count_char = 1
            while (forced_aligns[word_index].text != end_char) and (count_char < end_char_count):
                word_index += 1
                count_char += 1
        end_time = forced_aligns[word_index].end_time
        srt_time_stamps.append((start_time, end_time))
        word_index += 1
    return srt_time_stamps


def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    milliseconds = int((secs - int(secs)) * 1000)
    secs = int(secs)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

def generate_srt_string(result_segments, srt_time_stamp):
    srt_string =''''''

    for i, (segment, (start_time, end_time)) in enumerate(zip(result_segments, srt_time_stamp)):
        srt_string += f"{i+1}\n"
        start_str = format_timestamp(start_time)
        end_str = format_timestamp(end_time)
        srt_string += f"{start_str} --> {end_str}\n"
        srt_string += f"{segment}\n"
        srt_string += "\n"

    return srt_string

def save_srt_file(srt_string, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(srt_string)


def get_unique_filepath(base_dir, base_name, extension):
    if not extension.startswith('.'):
        extension = '.' + extension

    index = 0
    while True:
        if index == 0:
            filename = base_name + extension
        else:
            filename = f"{base_name}_{index}{extension}"
        filepath = os.path.join(base_dir, filename)
        if not os.path.exists(filepath):
            return filepath
        index += 1


class GenerateSRT:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING",),
                "forced_aligns" : ("LIST",),
            },
            "optional": {
                "save_srt": ("BOOLEAN", {"default": True}),
                "file_name" : ("STRING", {"default": "VoiceBridge\subtitle"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("srt_string",)
    FUNCTION = "generate_srt"
    CATEGORY = "VoiceBridge"

    def generate_srt(self, text, forced_aligns, save_srt=True, file_name="VoiceBridge\subtitle"):

        output_dir = folder_paths.get_output_directory()
        save_path = get_unique_filepath(output_dir, file_name, ".srt")


        result_segments = split_string_regex(text, DELIMITERS)

        srt_time_stamps = get_seg_timestamps(result_segments, forced_aligns)

        srt_string = generate_srt_string(result_segments, srt_time_stamps)
        if save_srt:
            save_srt_file(srt_string, save_path)
            print("srt file save to path: ", save_path)

        return (srt_string,)
    
class SaveSRTFromString:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "srt_string": ("STRING",),
                "file_name" : ("STRING", {"default": "VoiceBridge\subtitle"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("save_path",)
    FUNCTION = "save_srt"
    CATEGORY = "VoiceBridge"

    def save_srt(self, srt_string, file_name="VoiceBridge\subtitle"):
        output_dir = folder_paths.get_output_directory()
        save_path = get_unique_filepath(output_dir, file_name, ".srt")

        save_srt_file(srt_string, save_path)
        print("srt file save to path: ", save_path)

        return (save_path,)

class OpenAIAPI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("STRING",),
                "base_url": ("STRING",),
                "api_key": ("STRING",),
                "system_prompt": ("STRING",{"default": "You are a helpful assistant.", "multiline": True}),
                "prompt": ("STRING",{"default": "Hello", "multiline": True}),
            },
            "optional": {
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 100_0000}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0, "max": 1}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "call_api"
    CATEGORY = "VoiceBridge"

    def call_api(self, model, base_url, api_key, system_prompt, prompt, max_tokens=4096, temperature=0.7, top_p=0.95):
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=False
        )
        
        result = response.choices[0].message.content
        return (result, )





# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "GenerateSRT": GenerateSRT,
    "Qwen3ASRLoaderVB": Qwen3ASRLoader,
    "Qwen3ASRTranscribeVB": Qwen3ASRTranscribe,
    "OenAIAPIVB": OpenAIAPI,
    "SaveSRTFromString": SaveSRTFromString,
    "Qwen3TTSLoaderVB": Qwen3TTSLoader,
    "VoiceClonePrompt": VoiceClonePrompt,
    "SRTToAudio": SRTToAudio,

}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "GenerateSRT": "Generate SRT",
    "Qwen3ASRLoaderVB": "Qwen3-ASR Loader VoiceBridge",
    "Qwen3ASRTranscribeVB": "Qwen3-ASR Transcribe VoiceBridge",
    "OenAIAPIVB": "OpenAI API VoiceBridge",
    "SaveSRTFromString": "Save SRT From String",
    "Qwen3TTSLoaderVB": "Qwen3-TTS Loader VoiceBridge",
    "VoiceClonePrompt": "Voice Clone Prompt",
    "SRTToAudio": "SRT To Audio",
}