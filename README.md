# ComfyUI-VoiceBridge

<div align="center">
	<a href="https://github.com/YanTianlong-01/comfyui_voicebridge">
		<img src="./assets/Icon.png" width="200" height="200">
	</a>
	<h1>VoiceBridge</h1>
	<p>
		<b>VoiceBridge translates spoken audio from any language to a target language while preserving the original speaker's voice characteristics, and generates bilingual SRT subtitle files.</b>
	</p>
	<br>
</div>

## 🔄 Workflow
This node integrates ASR (Automatic Speech Recognition), LLM (Large Language Model), and TTS (Text-to-Speech) technologies to provide a complete speech translation pipeline.

[Online Workflow](https://www.runninghub.ai/post/2050059053363154946?inviteCode=rh-v1455)
### ASR + LLM + TTS Workflow
![](./assets/workflow.png)

### Universal TTS Workflow
![](./assets/universal-tts.png)

## ✨ Features

- 🌍 **Speech Translation:** Convert speech from one language to any other language while retaining the original speaker's voice timbre.

- 🗣️ **Multi-Language Support:** Speech recognition and translation in dozens of languages covering all major global languages.

- ⏱️ **Automatic Voice Alignment:** The generated translated voice is automatically aligned with the original voice to stay in sync with the video content.

- 📝 **Accurate Subtitle Generation:** Through force-align technology, VoiceBridge produces accurate subtitles synchronized with the voice at the millisecond level.

- 🔌 **Universal TTS Support:** The new decoupled pipeline — `Load SRT` → `SRT Splitter` → `<Any TTS node>` → `Audio List Merger by SRT` — lets you drive **any ComfyUI TTS custom node** (Qwen3-TTS, VoxCPM, Fish Audio S2, LongCat-AudioDiT, CosyVoice, …). As long as the TTS node takes a `STRING` and returns an `AUDIO`.


## 📦 Installation

### Via ComfyUI Manager (Recommended)
Search for "VoiceBridge" in ComfyUI Manager

### Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YanTianlong-01/comfyui_voicebridge.git
cd comfyui_voicebridge
pip install -r requirements.txt
```

### Model Directory Structure
VoiceBridge automatically searches or download models in the following priority:

```text
ComfyUI/
├── models/
│   └── Qwen3-TTS/
│   |   ├── Qwen3-TTS-12Hz-1.7B-Base/
│   |   ├── Qwen3-TTS-12Hz-0.6B-Base/
│   └── Qwen3-ASR/
│       ├── Qwen3-ASR-1.7B/
│       ├── Qwen3-ASR-0.6B/
│       ├── Qwen3-ForcedAligner-0.6B/
```

or you can use the local model path. Setting `local_model_path`=`qwen-tts/Qwen3-TTS-12Hz-1.7B-Base` means use the model in `ComfyUI/models/qwen-tts/Qwen3-TTS-12Hz-1.7B-Base`.

For ASR models, you can set `local_model_path_asr`=`qwen-asr/Qwen3-ASR-1.7B`.
For ForcedAligner models, you can set `local_model_path_fa`=`qwen-asr/Qwen3-ForcedAligner-0.6B`.



## 🧩 Nodes

### 📥 Load SRT

Loads an `.srt` (or `.txt`) subtitle file from `ComfyUI/input/`. The node ships with a dropdown listing existing files, plus a frontend-injected **"choose srt file to upload"** button that mirrors the UX of ComfyUI's built-in *Load Audio* node. Uploaded files are saved to the input directory and appear in the dropdown immediately.

Output is the raw SRT string, ready to be wired into `SRT Splitter.srt_string`.

| Input | Type | Description |
|-------|------|-------------|
| srt | dropdown | Pick an `.srt` / `.txt` file from `ComfyUI/input/`. Uploaded files are auto-added to this list. |
| **Output** | **Type** | **Description** |
| srt_string | STRING | Raw SRT content decoded from the file (UTF-8 with BOM / GBK fallback handled internally). |

---

### ✂️ SRT Splitter

Parses an SRT string and splits it into a list of per-sentence texts plus a metadata object that carries the original timestamps. Because `texts` is emitted as a native ComfyUI `LIST`, any downstream TTS node connected to it will be executed once per sentence automatically — no custom loop nodes required.

| Input | Type | Description |
|-------|------|-------------|
| srt_string | STRING | SRT text from `Load SRT`, `Generate SRT`, or a manual paste. |
| **Output** | **Type** | **Description** |
| texts | LIST[STRING] | List of per-sentence texts, one entry per SRT subtitle. ComfyUI's list-iteration will feed them to the downstream TTS one by one. |
| srt_items | VB_SRT_ITEMS | Metadata object holding `(index, start_ms, end_ms, text)` for every subtitle entry. Feed this into `Audio List Merger by SRT`. |
| count | INT | Number of subtitle entries parsed. |

---

### 🔀 Audio List Merger by SRT

Collects the per-sentence `AUDIO` outputs produced by any external TTS node (iterated over `SRT Splitter.texts`) and merges them back into a single `AUDIO` aligned with the original SRT timestamps. Uses the same duration-matching / tempo-limit / cascading-shift algorithm as the legacy `SRT To Audio` node. The sample rate is read from the first incoming `AUDIO` dict and every other sentence is resampled to match.

| Input | Type | Description |
|-------|------|-------------|
| srt_items | VB_SRT_ITEMS | Metadata from `SRT Splitter` carrying the original timestamps. |
| audios | LIST[AUDIO] | Per-sentence AUDIO outputs from any TTS node — auto-collected by ComfyUI's list-iteration. |
| tempo_limit | FLOAT | (optional) Maximum speed-up factor for audio that overflows a subtitle slot. Default 1.5, range 1.0–5.0. |
| mini_gap_ms | INT | (optional) Minimum gap between consecutive subtitles in milliseconds. Default 100, range 0–10,000. |
| **Output** | **Type** | **Description** |
| audio | AUDIO | Final merged audio in ComfyUI format (`waveform`, `sample_rate`). |
| adjusted_srt | STRING | Adjusted SRT string whose timing reflects the real audio durations after tempo / shift corrections. |

---

### VoiceBridge ASR Loader

Loads the Qwen3-ASR model with auto-download support. This node manages model loading with intelligent caching and supports both HuggingFace and ModelScope model sources.

| Input | Type | Description |
|-------|------|-------------|
| repo_id | dropdown | Model repository selection (Qwen/Qwen3-ASR-1.7B or Qwen/Qwen3-ASR-0.6B) |
| source | dropdown | Model download source: HuggingFace or ModelScope |
| precision | dropdown | Computation precision: fp16, bf16, or fp32 |
| attention | dropdown | Attention implementation: auto, flash_attention_2, sdpa, or eager |
| max_new_tokens | INT | Maximum tokens to generate during transcription (default: 256, range: 1-4096) |
| forced_aligner | dropdown | Optional forced aligner model for word-level timestamps (None or Qwen/Qwen3-ForcedAligner-0.6B) |
| local_model_path_asr | STRING | Optional custom local path for ASR model loading |
| local_model_path_fa | STRING | Optional custom local path for forced aligner model loading |
| **Output** | **Type** | **Description** |
| model | QWEN3_ASR_MODEL | Loaded ASR model instance for transcription operations |

---

### VoiceBridge ASR Transcribe

Transcribes a single audio input to text using the loaded Qwen3-ASR model. Supports automatic language detection and optional timestamp generation for precise word-level timing.

| Input | Type | Description |
|-------|------|-------------|
| model | QWEN3_ASR_MODEL | Pre-loaded ASR model from VoiceBridge ASR Loader |
| audio | AUDIO | Audio input in ComfyUI format (waveform and sample_rate) |
| language | dropdown | Target language for transcription or "auto" for automatic detection |
| context | STRING | Optional context hints to improve transcription accuracy for specialized content |
| return_timestamps | BOOLEAN | Enable word-level timestamp output for precise timing information |
| **Output** | **Type** | **Description** |
| forced_aligns | LIST | List of ForcedAlignItem objects containing word-level timing data (if timestamps enabled) |
| text | STRING | Transcribed text content from the audio input |
| language | STRING | Detected or specified language code |
| timestamps | STRING | Formatted string with word-level timestamps in "start_time-end_time: word" format (if enabled) |



---

### Generate SRT

Generates an SRT subtitle file from transcribed text and forced alignment timestamps. This node combines text segmentation with timestamp data to create properly timed subtitle entries.

| Input | Type | Description |
|-------|------|-------------|
| forced_aligns | LIST | List of ForcedAlignItem objects with word-level timestamps from ASR transcription |
| text | STRING | Input text content to convert to SRT format |
| language | STRING | Language of SRT text |
| save_srt | BOOLEAN | Enable automatic saving of the generated SRT file to output directory (default: True) |
| file_name | STRING | Base file name for the SRT output (default: "VoiceBridge\subtitle") |
| **Output** | **Type** | **Description** |
| srt_string | STRING | Generated SRT formatted string with proper timestamps and segmentation |

---
### VoiceBridge AI API

Provides integration with OpenAI-compatible API endpoints for AI-powered text processing. Supports customizable system prompts and generation parameters for flexible content generation workflows.

| Input | Type | Description |
|-------|------|-------------|
| model | STRING | API model identifier to use for text generation |
| base_url | STRING | Base URL endpoint for the API (must be OpenAI-compatible) |
| api_key | STRING | API authentication key for the service |
| system_prompt | STRING | System instruction defining the AI assistant's behavior and constraints |
| prompt | STRING | User prompt or query to send to the API |
| max_tokens | INT | Maximum number of tokens in the generated response (default: 4096, range: 1-1000000) |
| temperature | FLOAT | Sampling temperature controlling randomness (default: 0.7, range: 0-1) |
| top_p | FLOAT | Nucleus sampling threshold for token selection (default: 0.95, range: 0-1) |
| **Output** | **Type** | **Description** |
| response | STRING | Generated text response from the API |

---

### VoiceBridge TTS Loader

Loads the Qwen3-TTS text-to-speech model with auto-download support. This node handles model caching and supports multiple model variants for different voice synthesis needs.

| Input | Type | Description |
|-------|------|-------------|
| repo_id | dropdown | Model repository selection (Qwen/Qwen3-TTS-12Hz-1.7B-Base or Qwen/Qwen3-TTS-12Hz-0.6B-Base) |
| source | dropdown | Model download source: HuggingFace or ModelScope |
| precision | dropdown | Computation precision: fp16, bf16, or fp32 |
| attention | dropdown | Attention implementation: auto, flash_attention_2, sdpa, or eager |
| local_model_path | STRING | Optional custom local path for TTS model loading |
| **Output** | **Type** | **Description** |
| model | QWEN3_TTS_MODEL | Loaded TTS model instance for voice synthesis operations |

---

### Voice Clone Prompt

Creates a voice clone prompt from reference audio for use with the Qwen3-TTS model. This enables voice cloning by extracting voice characteristics from a sample audio file.

| Input | Type | Description |
|-------|------|-------------|
| model | QWEN3_TTS_MODEL | Pre-loaded TTS model from VoiceBridge TTS Loader |
| ref_audio | AUDIO | Reference audio sample for voice cloning (ComfyUI Audio format) |
| ref_text | STRING | Transcript of the reference audio content (highly recommended for better quality) |
| **Output** | **Type** | **Description** |
| voice_clone_prompt | VOICE_CLONE_PROMPT | Voice clone prompt object containing extracted voice characteristics for TTS generation |

---

### SRT To Audio

Converts SRT subtitle content to audio using the Qwen3-TTS model with voice cloning. Includes intelligent duration matching to align generated audio with subtitle timing, with optional speed adjustment for timing mismatches.

| Input | Type | Description |
|-------|------|-------------|
| model | QWEN3_TTS_MODEL | Pre-loaded TTS model from VoiceBridge TTS Loader |
| srt_string | STRING | SRT formatted subtitle text to convert to audio |
| voice_clone_prompt | VOICE_CLONE_PROMPT | Voice clone prompt from Voice Clone Prompt node |
| language | dropdown | Target language for synthesis or "auto" for automatic detection |
| tempo_limit | FLOAT | Maximum speed-up factor for audio that exceeds subtitle duration (default: 1.5, range: 1.0-5.0) |
| mini_gap_ms | INT | Minimum gap between consecutive subtitles in milliseconds (default: 100, range: 0-10,000) |
| batch_size | INT | Number of subtitle entries to process in each batch (default: 10, range: 1-1,000) |
| **Output** | **Type** | **Description** |
| audio | AUDIO | Generated audio in ComfyUI format (waveform and sample_rate) |
| adjusted_srt | STRING | Adjusted SRT string with corrected timing based on actual audio durations |



---

### Save SRT From String

Saves an SRT formatted string to a file in the ComfyUI output directory. Automatically handles file naming conflicts by appending incrementing indices.

| Input | Type | Description |
|-------|------|-------------|
| srt_string | STRING | SRT formatted string content to save |
| file_name | STRING | Base file name for the SRT output (default: "VoiceBridge\subtitle") |
| **Output** | **Type** | **Description** |
| save_path | STRING | Full path where the SRT file was saved |


---

### 🧹 VoiceBridge Unload Model

Unloads cached ASR and/or TTS models from GPU memory to free VRAM. Useful when chaining different models sequentially or running on limited-memory GPUs. Acts as a pass-through node — wire the `anything` input after the node whose output you want to keep, and the same value will be forwarded on `any` once the unload runs.

| Input | Type | Description |
|-------|------|-------------|
| anything | * | Any value, passed straight through the node so you can insert it anywhere in a chain. |
| Unload_ASR_Model | BOOLEAN | Unload the ASR model cache (default `True`). |
| Unload_TTS_Model | BOOLEAN | Unload the TTS model cache (default `True`). |
| **Output** | **Type** | **Description** |
| any | * | Same value as `anything` (pass-through). |


## 🙏 Acknowledgments

- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) and [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS): Official open-source repository by Alibaba Qwen team.
- [ComfyUI-Qwen3-ASR](https://github.com/DarioFT/ComfyUI-Qwen3-ASR): A nice and clean ComfyUI node by DarioFT.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

