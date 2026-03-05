# srt_utils.py
import re
import os
import subprocess
import numpy as np
import torch
import soundfile as sf
from pydub import AudioSegment
from dataclasses import dataclass
from typing import List, Optional, Tuple
import folder_paths


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
    
    # 使用 ComfyUI 临时目录
    comfy_temp = folder_paths.get_temp_directory()
    os.makedirs(comfy_temp, exist_ok=True)
    tmp_path = os.path.join(comfy_temp, f"voicebridge_merge_{os.getpid()}.wav")
    
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


# ---------------------------------------------------------------- Voice Bridge Linker -----------------------------------------------------------

# DELIMITERS = ['，', '。', '！', '？', '；', '：', 
#               ',',  '.',  '!',  '?',  ';',  ':', 
#               '\n', '\r', '\t'
#             ]

CN_DELIMITERS = ['，', '。', '！', '？', '；', '：',]
EN_DELIMITERS = [',', '.', '!', '?', ';', ':',]



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
        tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+|[^\\s\u4e00-\u9fff a-zA-Z0-9]', segment)
        separators = r'[,\.\?!，。！？;；:：、\s\/\\\-—_()（）\[\]【】《》<>"''‘’“”=+*&#@%$^|~`]+'
        word_list = [t for t in tokens if t not in separators]

        start_char = word_list[0][0]
        end_char = word_list[-1][-1]
        print(word_list)

        # ------ check if it's english word ------ #
        if is_english_char(start_char) and forced_aligns[word_index].text == word_list[0]:
            start_char = word_list[0]
        if is_english_char(end_char):
             for test_j in range(word_index, len(forced_aligns)):
                if word_list[-1] == forced_aligns[test_j].text: # 如果在后面的forced aligns被分词了
                    end_char = word_list[-1]
                    break
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
            for word_jdx in range(word_index, len(forced_aligns)):
                if forced_aligns[word_jdx].text == end_char:
                    if count_char < end_char_count:
                        count_char += 1
                        word_index += 1
                    else:
                        break
                else:
                    word_index += 1

        end_time = forced_aligns[word_index].end_time
        srt_time_stamps.append((start_time, end_time))
        word_index += 1
    return srt_time_stamps

def get_seg_timestamps_2(segments, forced_aligns):
    segments_one_word = []
    segments_word_list = []

    for i, segment in enumerate(segments):
        separators = r'[,\.\?!，。！？;；:：、\s\/\\\-—_()（）\[\]【】《》<>"''‘’“”=+*&#@%$^|~`]+'
        word_list = [t for t in re.split(separators, segment) if t]
        final_word_list = []

        for word in word_list:
            tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+|[^\\s\u4e00-\u9fff a-zA-Z0-9]', word)
            final_word_list.extend(tokens)

        start_char = final_word_list[0][0]
        end_char = final_word_list[-1][-1]

        if is_english_char(start_char): # 因为已经使用了token分词，所有如果是英文字母的话，取出来一定是单词或者是单个英文字母，可以和fa中的text对应上。
            start_char = final_word_list[0]
        if is_english_char(end_char):
            end_char = final_word_list[-1]


        segments_word_list.append(final_word_list)
        # print(" ", i, "final_word_list", final_word_list)

        if end_char == start_char == segment:
            segments_one_word.append(True)
        else:
            segments_one_word.append(False)

    srt_time_stamps = []
    # word_index = 0
    forced_aligns_cp = forced_aligns
    for i, segment in enumerate(segments):
        if segments_one_word[i]:
            srt_time_stamps.append((forced_aligns_cp[0].start_time, forced_aligns_cp[0].end_time))
            forced_aligns_cp = forced_aligns_cp[1:]
            # word_index = 0
            continue

        start_time = forced_aligns_cp[0].start_time
        end_time = None


        
        # print("i=", i)
        # print("segments_word_list[i]", segments_word_list[i])
        end_char = segments_word_list[i][-1][-1] # 取最后一个word的最后一个字符
        if is_english_char(end_char): # 检查是否为英文
            end_char = segments_word_list[i][-1]

        while not end_time:
            end_char_count = segments_word_list.count(end_char)

            # print("end_char", end_char)
            # print("end_char_count", end_char_count)
            count_char = 1
            # print("forced_aligns_cp[0]", forced_aligns_cp[0].text)
            # print("len(forced_aligns_cp)", len(forced_aligns_cp))
            for word_jdx in range(0, len(forced_aligns_cp)): # 从前往后检索
                if forced_aligns_cp[word_jdx].text == end_char:
                    if count_char < end_char_count:
                        # 找到了一样的，但不是最后那个
                        count_char += 1
                        # word_index += 1
                    else:
                        # 找到了最后一个字符的force align了
                        end_time = forced_aligns_cp[word_jdx].end_time
                        forced_aligns_cp = forced_aligns_cp[word_jdx+1:]
                        # word_index = 0
                        break

            # 如果检索完整个forced aligns都没有找到，说明匹配失败
            if not end_time:
                if len(segments_word_list[i]) ==1:
                    # 整一段字幕，没有一个字符能在force aligns中找到匹配的。则将第一个字符的end time作为介绍 (几乎不可能发生)
                    raise Exception("No match found for segment: ", segment, "Please Report this issue in Github https://github.com/YanTianlong-01/comfyui_voicebridge")
                    end_time = forced_aligns_cp[0].end_time
                    pass
                # 把最后两个字符合并起来，作为新的最后一个字符，重新检索
              
                segments_word_list[i] = segments_word_list[i][:-2] + [segments_word_list[i][-2]+segments_word_list[i][-1]]
                end_char = segments_word_list[i][-1]

        srt_time_stamps.append((start_time, end_time))
    return srt_time_stamps




def adjust_srt_timestamps(segements, srt_time_stamps):
    '''
    微调字幕
    '''
    concate_segement = ["hi", "hello", "hey", "Hi", "Hello", "Hey", "大家好", "嗨"]
    result_segments = [segements[0]]
    result_srt_time_stamps = [srt_time_stamps[0]]
    
    # for i, (segment, (start_time, end_time)) in enumerate(zip(segements, srt_time_stamps)):
    for i in range(1, len(segements)):
        segment = segements[i]
        (start_time, end_time) = srt_time_stamps[i]
        if segment in concate_segement:
            merged = result_segments[-1] + " " + segment
            result_segments[-1] = merged

            merged_time = (result_srt_time_stamps[-1][0], end_time)
            result_srt_time_stamps[-1] = merged_time
        else:
            result_segments.append(segment)
            result_srt_time_stamps.append((start_time, end_time))
            
    return result_segments, result_srt_time_stamps


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