# ComfyUI-VoiceBridge

ComfyUI-VoiceBridge is a powerful ComfyUI custom node that translates spoken audio from any language to a target language while preserving the original speaker's voice characteristics, and generates bilingual SRT subtitle files. 

---
This node integrates ASR (Automatic Speech Recognition), LLM (Large Language Model), and TTS (Text-to-Speech) technologies to provide a complete speech translation pipeline.

![](./assets/workflow.png)

# Features

- **Speech Translation:** Convert speech from one language to any other language while retaining the original speaker's voice timbre


- **Multi-Language Support:** Support for speech recognition and translation in dozens of languages covering major global languages

- **Automatic voice alignment:** The generated translated voice is automatically aligned with the original voice to ensure the synchronization between the translated voice and the video content.

- **Accurate subtitle generation:** Through force align technology, accurate subtitles can be generated, ensuring that the subtitles and voice are synchronized at the millisecond level.


## Installation

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


## Acknowledgments

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS): Official open-source repository by Alibaba Qwen team.
- [ComfyUI-Qwen3-ASR](https://github.com/DarioFT/ComfyUI-Qwen3-ASR): A nice and clean ComfyUI node by DarioFT.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

