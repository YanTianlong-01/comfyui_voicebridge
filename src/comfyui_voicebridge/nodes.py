# nodes.py
import os
import torch
import shutil
import folder_paths
import soundfile as sf
import numpy as np
import comfy.model_management as mm
from typing import List, Optional, Tuple

from .model_utils import (
    _ASR_MODEL_CACHE, _TTS_MODEL_CACHE,
    increment_unload_counter, get_unload_counter,
    unload_asr_model, unload_tts_model,
    ModelKey,
    QWEN3_ASR_MODELS, QWEN3_FORCED_ALIGNERS, SUPPORTED_LANGUAGES,
    QWEN3_TTS_MODELS, LANGUAGE_MAP,
    get_local_model_path, download_model_to_comfyui, load_audio_input
)
from .srt_utils import (
    SubtitleEntry,
    parse_srt_string, save_srt_string,
    get_audio_duration_ms, speed_up_audio, merge_audio_files,
    split_string_regex, get_seg_timestamps_2, adjust_srt_timestamps,
    generate_srt_string, save_srt_file, get_unique_filepath,
    CN_DELIMITERS, EN_DELIMITERS, format_timestamp
)


# voice bridge nodes
# import re
# import os
# import gc
# import shutil
# import torch
# import numpy as np
# import folder_paths
# import comfy.model_management as mm

# # for srt2audio
# import soundfile as sf
# import tempfile
# from dataclasses import dataclass
# from typing import List, Optional, Tuple
# import subprocess
# from pydub import AudioSegment





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

    RETURN_TYPES = ("MODEL_KEY",)
    RETURN_NAMES = ("model_key",)
    FUNCTION = "load_model"
    CATEGORY = "VoiceBridge"

    def IS_CHANGED(cls, repo_id, source, precision, attention, max_new_tokens=256,
                   forced_aligner="None", local_model_path_asr="", local_model_path_fa=""):
        # 将输入参数和卸载计数器打包，任何变化都会触发重新执行
        params = (repo_id, source, precision, attention, max_new_tokens,
                  forced_aligner, local_model_path_asr, local_model_path_fa)
        return (params, get_unload_counter('asr'))

    def load_model(self, repo_id, source, precision, attention, max_new_tokens=256, forced_aligner="None", local_model_path_asr="", local_model_path_fa=""):
        # 延迟导入以缩短 ComfyUI 初始加载时间
        from qwen_asr import Qwen3ASRModel

        _ASR_MODEL_CACHE
        device = mm.get_torch_device()
        
        dtype = torch.float32
        if precision == "bf16":
            if device.type == "mps":
                dtype = torch.float16
                print(f"[VoiceBridge] Note: Using fp16 on MPS (bf16 has limited support)")
            else:
                dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16
            
        if local_model_path_asr and local_model_path_asr.strip() != "":
            model_path = local_model_path_asr.strip()
            model_path = os.path.join(folder_paths.models_dir, model_path)
            print(f"[VoiceBridge] Loading ASR from local path: {model_path}")
        else:
            local_path = get_local_model_path(repo_id, "Qwen3-ASR")
            if os.path.exists(local_path) and os.listdir(local_path):
                model_path = local_path
                print(f"[VoiceBridge] Loading ASR from ComfyUI models folder: {model_path}")
            else:
                model_path = download_model_to_comfyui(repo_id, source, "Qwen3-ASR")
        
        # 缓存键：模型路径 + 设备 + 精度 + 强制对齐器
        cache_key = repr((model_path, str(device), str(dtype), forced_aligner))
        if cache_key in _ASR_MODEL_CACHE:
            print(f"[VoiceBridge] Using cached ASR model: {repo_id}")
            return (ModelKey(cache_key),)
        
        # 加载新模型前清理旧缓存
        if _ASR_MODEL_CACHE:
            print(f"[VoiceBridge] Clearing existing ASR cache for new model...")
            unload_asr_model()
        
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
                print(f"[VoiceBridge] Loading Force Aligner from local path: {aligner_local}")
            elif not (os.path.exists(aligner_local) and os.listdir(aligner_local)):
                aligner_local = download_model_to_comfyui(forced_aligner, source, "Qwen3-ASR")
            print(f"[VoiceBridge] Loading Force Aligner: {aligner_local}")

            model_kwargs["forced_aligner"] = aligner_local
            model_kwargs["forced_aligner_kwargs"] = dict(
                dtype=dtype,
                device_map=str(device),
            )
            if attention != "auto":
                model_kwargs["forced_aligner_kwargs"]["attn_implementation"] = attention
        
        print(f"[VoiceBridge] Loading Qwen3-ASR model from {model_path}...")
        model = Qwen3ASRModel.from_pretrained(model_path, **model_kwargs)
        
        # 缓存模型
        _ASR_MODEL_CACHE[cache_key] = model
        print(f"[VoiceBridge] ASR model loaded and cached: {repo_id}")
        
        return (ModelKey(cache_key),)


class Qwen3ASRTranscribe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_key": ("MODEL_KEY",),
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

    def transcribe(self, model_key, audio, language="auto", context="", return_timestamps=False):
        audio_data = load_audio_input(audio)
        if audio_data is None:
            return ("", "", "")
        
        lang = None if language == "auto" else language
        ctx = context if context.strip() else ""

        _ASR_MODEL_CACHE
        model = _ASR_MODEL_CACHE.get(str(model_key))
        if model is None:
            raise RuntimeError(f"ASR model with key '{model_key}' not found in cache. Please run the loader first.")
        
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
    RETURN_TYPES = ("MODEL_KEY",)
    RETURN_NAMES = ("model_key",)
    FUNCTION = "load_model"
    CATEGORY = "VoiceBridge"

    def IS_CHANGED(cls, repo_id, source, precision, attention, local_model_path="",):
        # 将输入参数和卸载计数器打包，任何变化都会触发重新执行
        params = (repo_id, source, precision, attention, local_model_path)
        return (params, get_unload_counter('tts'))

    def load_model(self, repo_id, source, precision, attention, local_model_path=""):
        # 延迟导入以缩短 ComfyUI 初始加载时间
        from qwen_tts import Qwen3TTSModel

        _TTS_MODEL_CACHE
        device = mm.get_torch_device()
        
        dtype = torch.float32
        if precision == "bf16":
            if device.type == "mps":
                dtype = torch.float16
                print(f"[VoiceBridge] Note: Using fp16 on MPS (bf16 has limited support)")
            else:
                dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16
            
        if local_model_path and local_model_path.strip() != "":
            model_path = local_model_path.strip()
            model_path = os.path.join(folder_paths.models_dir, model_path)
            print(f"[VoiceBridge] Loading TTS from local path: {model_path}")
        else:
            local_path = get_local_model_path(repo_id, "Qwen3-TTS")
            if os.path.exists(local_path) and os.listdir(local_path):
                model_path = local_path
                print(f"[VoiceBridge] Loading TTS from ComfyUI models folder: {model_path}")
            else:
                model_path = download_model_to_comfyui(repo_id, source, "Qwen3-TTS")
        
        # 缓存键：模型路径 + 设备 + 精度
        cache_key = repr((model_path, str(device), str(dtype)))
        if cache_key in _TTS_MODEL_CACHE:
            print(f"[VoiceBridge] Using cached TTS model: {repo_id}")
            return (ModelKey(cache_key),)
        
        # 加载新模型前清理旧缓存
        if _TTS_MODEL_CACHE:
            print(f"[VoiceBridge] Clearing existing TTS cache for new model...")
            unload_tts_model()
        
        model_kwargs = dict(
            dtype=dtype,
            device_map=str(device),
        )
        if attention != "auto":
            model_kwargs["attn_implementation"] = attention
        
        print(f"[VoiceBridge] Loading Qwen3-TTS model from {model_path}...")
        model = Qwen3TTSModel.from_pretrained(model_path, **model_kwargs)
        
        # 缓存模型
        _TTS_MODEL_CACHE[cache_key] = model
        print(f"[VoiceBridge] TTS model loaded and cached: {repo_id}")
        
        return (ModelKey(cache_key),)
    
class VoiceClonePrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_key": ("MODEL_KEY",),
                "ref_audio": ("AUDIO", {"tooltip": "Reference audio (ComfyUI Audio)"}),
                "ref_text": ("STRING", {"multiline": True, "default": "", "placeholder": "Reference audio text (highly recommended for better quality)"}),
            }
        }
    
    RETURN_TYPES = ("VOICE_CLONE_PROMPT",)
    RETURN_NAMES = ("voice_clone_prompt",)
    FUNCTION = "create_prompt"
    CATEGORY = "VoiceBridge"

    def create_prompt(self, model_key, ref_audio, ref_text):
        audio_data = load_audio_input(ref_audio)
        if audio_data is None:
            return ("",)
        
        _TTS_MODEL_CACHE
        model = _TTS_MODEL_CACHE.get(str(model_key))
        if model is None:
            raise RuntimeError(f"TTS model with key '{model_key}' not found in cache. Please run the loader first.")
        
        result = model.create_voice_clone_prompt(
            ref_audio=audio_data,
            ref_text=ref_text,
            x_vector_only_mode=False,
        )
        print(f"[VoiceBridge] Voice Clone Prompt created successfully!")
        return (result,)
    
class SRTToAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_key": ("MODEL_KEY",),
                "srt_string": ("STRING", {"multiline": True, "default": "", "placeholder": "SRT text"}),
                "voice_clone_prompt": ("VOICE_CLONE_PROMPT",),
            },
            "optional": {
                "language": (SUPPORTED_LANGUAGES, {"default": "auto"}),
                "tempo_limit": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 5.0, "step": 0.1, "tooltip": "Maximum speed-up factor for audio that exceeds subtitle duration"}),
                "mini_gap_ms" : ("INT", {"default": 100, "min": 0, "max": 10000, "tooltip": "Minimum gap between subtitles in milliseconds"}),
                "batch_size": ("INT", {"default": 10, "min": 1, "max": 1000, "tooltip": "Number of subtitles to process in each batch"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "adjusted_srt")
    FUNCTION = "convert_srt_to_audio"
    CATEGORY = "VoiceBridge"

    def convert_srt_to_audio(
        self, 
        model_key, 
        srt_string, 
        voice_clone_prompt, 
        language="auto",
        tempo_limit: float = 1.5,
        mini_gap_ms: int = 100,
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
        _TTS_MODEL_CACHE
        model = _TTS_MODEL_CACHE.get(str(model_key))
        if model is None:
            raise RuntimeError(f"TTS model with key '{model_key}' not found in cache. Please run the loader first.")

        if not srt_string or not srt_string.strip():
            print(f"[VoiceBridge] Error: Empty SRT string provided")
            return ({"waveform": np.array([[0.0]]), "sample_rate": 16000}, "")
        
        print(f"[VoiceBridge] Parsing SRT content ({len(srt_string)} chars)...")
        entries = parse_srt_string(srt_string)
        print(f"[VoiceBridge] Found {len(entries)} subtitle entries")
        
        if len(entries) == 0:
            print(f"[VoiceBridge] Error: No valid subtitle entries found in SRT")
            return ({"waveform": np.array([[0.0]]), "sample_rate": 16000}, "")
        
        # 使用 ComfyUI 临时目录
        comfy_temp = folder_paths.get_temp_directory()
        temp_dir = os.path.join(comfy_temp, f"srt_audio_{os.getpid()}_{id(self)}")
        os.makedirs(temp_dir, exist_ok=True)
        print(f"[VoiceBridge] Using temp directory: {temp_dir}")
        
        lang = LANGUAGE_MAP.get(language, "auto")
        
        try:
            print(f"[VoiceBridge] Starting audio generation...")
            for i in range(0, len(entries), batch_size):
                batch = entries[i:i+batch_size]
                texts = [e.text for e in batch]
                paths = [os.path.join(temp_dir, f"audio_{e.index:04d}.wav") for e in batch]
                
                print(f"[VoiceBridge]   Generating batch {i//batch_size + 1}: entries {i+1}-{min(i+batch_size, len(entries))}")
                
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
            
            print(f"[VoiceBridge] Processing duration mismatches...")
            self._process_duration(entries, temp_dir, tempo_limit, mini_gap_ms)
            
            last_entry = entries[-1]
            total_duration = last_entry.start_time_ms + last_entry.audio_duration_ms + 1000
            
            # Synthesize the final audio
            print(f"[VoiceBridge] Merging audio files...")
            wav_tensor, sample_rate = merge_audio_files(entries, total_duration)

            # Prepare audio output in ComfyUI format
            audio_output = {
                "waveform": wav_tensor,
                "sample_rate": sample_rate
            }

            
            # Generate the adjusted SRT string
            adjusted_srt = save_srt_string(entries)
            print(f"[VoiceBridge] Completed! Output audio: {wav_tensor.shape[-1]} samples at {sample_rate}Hz")
            
            return (audio_output, adjusted_srt)
            
        except Exception as e:
            print(f"[VoiceBridge] Error during audio conversion: {str(e)}")
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
    
    def _process_duration(self, entries: List[SubtitleEntry], temp_dir: str, tempo_limit: float, mini_gap_ms: int):
        """
        Handling the issue where the audio duration does not match the subtitle duration
        
        Args:
            entries: List of subtitle entries
            temp_dir: Temporary file directory
            tempo_limit: Maximum acceleration multiple
        """
        gap_ms = mini_gap_ms

        for i, entry in enumerate(entries):
            subtitle_duration = entry.end_time_ms - entry.start_time_ms
            audio_duration = entry.audio_duration_ms
            
            # Calculate the available time (the time until the next subtitle starts)
            if i < len(entries) - 1:
                available_time = entries[i+1].start_time_ms - entry.start_time_ms
            else:
                available_time = subtitle_duration + 5000  # 5 more seconds for safety
            
            print(f"[VoiceBridge]   [{entry.index}] Subtitle: {subtitle_duration}ms, Audio: {audio_duration}ms, Available: {available_time}ms")
            
            if audio_duration <= subtitle_duration:
                # If the audio is shorter than the subtitle -> adjust the end time of the subtitle
                new_end_time = entry.start_time_ms + audio_duration
                print(f"[VoiceBridge]        -> Audio shorter, adjusting end time: {entry.end_time_ms}ms -> {new_end_time}ms")
                entry.end_time_ms = new_end_time
                
            elif audio_duration > available_time:
                # If the audio is longer than the available time -> speed up the audio
                speed_factor = audio_duration / (available_time-gap_ms)
                
                if speed_factor > tempo_limit:
                    print(f"[VoiceBridge]        -> Warning: Required speed-up {speed_factor:.2f}x exceeds limit {tempo_limit}x, using limit")
                    speed_factor = tempo_limit
                
                print(f"[VoiceBridge]        -> Audio too long, speeding up by {speed_factor:.2f}x")
                
                # Speed up audio
                sped_up_path = os.path.join(temp_dir, f"audio_{entry.index:04d}_sped.wav")
                speed_up_audio(entry.audio_path, sped_up_path, speed_factor)
                entry.audio_path = sped_up_path
                entry.audio_duration_ms = get_audio_duration_ms(sped_up_path)
                
                # Update the end time of the subtitle
                entry.end_time_ms = entry.start_time_ms + entry.audio_duration_ms

                if entry.end_time_ms > entries[i+1].start_time_ms and 0 < i < len(entries) - 1:
                    # 如果还是超过了下一个字幕的开始时间，则借用上一个字幕和这个字幕之间的空隙。
                    print(f"[VoiceBridge]        -> Audio still too long, borrowing time from previous subtitle")
                    entry.start_time_ms = entries[i-1].end_time_ms + gap_ms
                    entry.end_time_ms = entry.start_time_ms + entry.audio_duration_ms
                
            else:
                # The audio is within the available range but exceeds the original subtitle duration -> only adjust the subtitle end time
                print(f"[VoiceBridge]        -> Audio slightly longer than subtitle but within available time, adjusting end time")
                entry.end_time_ms = entry.start_time_ms + audio_duration
            
            print(f"[VoiceBridge]        -> New subtitle: {entry.start_time_ms}ms -> {entry.end_time_ms}ms")

        print(f"[VoiceBridge] Cascading shifting") # 级联偏移
        for idx in range(0, len(entries) - 1):
            subtitle_duration = entries[idx].end_time_ms - entries[idx].start_time_ms
            ms_to_next_subtitle = entries[idx+1].start_time_ms - entries[idx].end_time_ms

            print(f"[VoiceBridge]   [{idx+1}] Subtitle: {subtitle_duration}ms, Gap to next: {ms_to_next_subtitle}ms")
            if ms_to_next_subtitle < gap_ms and idx < len(entries) - 1:
                    # 如果还是超过了下一个字幕的开始时间，将后面的字幕依次往后移动，直到不超过下一个字幕的开始时间为止。
                    print(f"[VoiceBridge]        -> Audio still too long, moving subsequent subtitles")
                    borrow_time = entries[idx].end_time_ms - entries[idx+1].start_time_ms
                    for j in range(idx+1, len(entries)):
                        entries[j].start_time_ms += borrow_time + gap_ms
                        entries[j].end_time_ms += borrow_time + gap_ms
                        print(f"[VoiceBridge]          -> Moving subtitle {entries[j].index} backward by {borrow_time + gap_ms}ms")
                        if j == len(entries) - 1:
                            break
                        elif entries[j].end_time_ms < entries[j+1].start_time_ms:
                            break
                        else:
                            borrow_time = entries[j].end_time_ms - entries[j+1].start_time_ms




class GenerateSRT:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "forced_aligns" : ("LIST",),
                "text": ("STRING",),
                "language": ("STRING",)
                
            },
            "optional": {
                "save_srt": ("BOOLEAN", {"default": True}),
                "filename_prefix" : ("STRING", {"default": "VoiceBridge\subtitle"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("srt_string",)
    FUNCTION = "generate_srt"
    CATEGORY = "VoiceBridge"

    def generate_srt(self, forced_aligns, text, language, save_srt=True, filename_prefix="VoiceBridge\subtitle"):

        output_dir = folder_paths.get_output_directory()
        save_path = get_unique_filepath(output_dir, filename_prefix, ".srt")

        if language == "Chinese":
            result_segments = split_string_regex(text, CN_DELIMITERS)
        else:
            result_segments = split_string_regex(text, EN_DELIMITERS)

        srt_time_stamps = get_seg_timestamps_2(result_segments, forced_aligns)

        adjust_segments, adjust_srt_time_stamps = adjust_srt_timestamps(result_segments, srt_time_stamps)

        srt_string = generate_srt_string(adjust_segments, adjust_srt_time_stamps)
        if save_srt:
            save_srt_file(srt_string, save_path)
            print(f"[VoiceBridge] srt file save to path: ", save_path)

        return (srt_string,)
    
class SaveSRTFromString:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "srt_string": ("STRING",),
                "filename_prefix" : ("STRING", {"default": "VoiceBridge\subtitle"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("save_path",)
    FUNCTION = "save_srt"
    CATEGORY = "VoiceBridge"

    def save_srt(self, srt_string, filename_prefix="VoiceBridge\subtitle"):
        output_dir = folder_paths.get_output_directory()
        save_path = get_unique_filepath(output_dir, filename_prefix, ".srt")

        save_srt_file(srt_string, save_path)
        print(f"[VoiceBridge] srt file save to path: ", save_path)

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
        from openai import OpenAI
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

class AnyType(str):
  """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""
  def __eq__(self, __value: object) -> bool:
    return True
  def __ne__(self, __value: object) -> bool:
    return False

any = AnyType("*")

class UnloadModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "anything" : (any, {}),
                "Unload_ASR_Model" : ("BOOLEAN", {"default": True}),
                "Unload_TTS_Model" : ("BOOLEAN", {"default": True}),
            },
        }
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("any",)
    FUNCTION = "unload_model"
    CATEGORY = "VoiceBridge"

    def unload_model(self, anything, Unload_ASR_Model=True, Unload_TTS_Model=True):
        if Unload_ASR_Model:
            unload_asr_model()
            increment_unload_counter("asr")
        if Unload_TTS_Model:
            unload_tts_model()
            increment_unload_counter("tts")
        return (anything, )
 



# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "GenerateSRT": GenerateSRT,
    "VoiceBridgeASRLoader": Qwen3ASRLoader,
    "VoiceBridgeASRTranscribe": Qwen3ASRTranscribe,
    "VoiceBridgeAIAPI": OpenAIAPI,
    "SaveSRTFromString": SaveSRTFromString,
    "VoiceBridgeTTSLoader": Qwen3TTSLoader,
    "VoiceClonePrompt": VoiceClonePrompt,
    "SRTToAudio": SRTToAudio,
    "VoiceBridgeUnloadModel": UnloadModel,

}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "GenerateSRT": "Generate SRT",
    "VoiceBridgeASRLoader": "VoiceBridge ASR Loader",
    "VoiceBridgeASRTranscribe": "VoiceBridge ASR Transcribe",
    "VoiceBridgeAIAPI": "VoiceBridge AI API",
    "SaveSRTFromString": "Save SRT From String",
    "VoiceBridgeTTSLoader": "VoiceBridge TTS Loader",
    "VoiceClonePrompt": "Voice Clone Prompt",
    "SRTToAudio": "SRT To Audio",
    "VoiceBridgeUnloadModel": "VoiceBridge Unload Model",
}