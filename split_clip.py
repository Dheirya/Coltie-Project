from whisperx.diarize import DiarizationPipeline
from pathlib import Path
import requests
import whisperx
import logging
import ffmpeg
import json
import os

logging.basicConfig(level=logging.INFO)


def ai_chat(prompt):
    url = "https://ai.hackclub.com/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {"messages": [{"role": "user", "content": prompt}]}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        logging.warning(f"Error: {e}")
    except (KeyError, IndexError):
        logging.warning("Error: Unexpected response format")


def diarize(audio_file, out):
    logging.info("Starting diarizing of segments...")
    audio = whisperx.load_audio(audio_file)
    diarize_model = DiarizationPipeline(use_auth_token="TOKEN", device="cpu")
    diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=2)
    return clean(diarize_segments, out)


def save_json(data, filename, out):
    with open(out + filename, "w") as f:
        json.dump(data, f, indent=2)
    logging.info(f"Saved data to {filename}")


def clean(diarize_segments, out):
    logging.info("Starting cleaning of segments...")
    clean_segments = []
    for row in diarize_segments.itertuples(index=False):
        clean_segments.append({"start": float(row.start), "end": float(row.end), "speaker": row.speaker})
    save_json(clean_segments, "diarized.json", out)
    return group(clean_segments, out)


def group(clean_segments, out):
    logging.info("Starting grouping of segments...")
    grouped = []
    for seg in clean_segments:
        start = seg["start"]
        end = seg["end"]
        speaker = seg["speaker"]
        if not grouped:
            grouped.append({"speaker": speaker, "start": start, "end": end})
        else:
            last = grouped[-1]
            if last["speaker"] == speaker:
                last["end"] = end
            else:
                grouped.append({"speaker": speaker, "start": start, "end": end})
    save_json(grouped, "grouped.json", out)
    return grouped


def transcription(start, end, segment_index, video_path, out_directory):
    logging.info('Transcribing a clip...')
    segment_path = os.path.join(out_directory, f"pre_segment_{segment_index + 1:03d}.mp3")
    ffmpeg.input(video_path, ss=start, to=end).output(segment_path, acodec='libmp3lame').run(overwrite_output=True, quiet=True)
    audio = whisperx.load_audio(segment_path)
    result = transcription.model.transcribe(audio, batch_size=16, language="en")
    with open(os.path.join(out_directory, f"transcription_{segment_index:03d}.json"), "w") as f:
        json.dump(result["segments"], f, indent=2)
    return result


def divideText(max_length, text):
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + (1 if current_line else 0) > max_length:
            lines.append(current_line)
            current_line = word
        else:
            if current_line:
                current_line += " " + word
            else:
                current_line = word
    if current_line:
        lines.append(current_line)
    return lines


def addText(video, text, summarized_video):
    font_path = 'font.ttf'
    lines = divideText(30, text)
    line_spacing_factor = 0.05
    start_offset_factor = 0.04
    vf_filters = []
    for i, line in enumerate(lines):
        y_expr = f"h*{start_offset_factor} + {i}*h*{line_spacing_factor}"
        drawtext = (
            f"drawtext=text='{line}':"
            f"fontfile='{font_path}':fontcolor=white:fontsize=h*0.03:"
            f"x=(w-text_w)/2:y={y_expr}:"
            f"box=1:boxcolor=black@0.5:boxborderw=20"
        )
        vf_filters.append(drawtext)
    vf_str = ",".join(vf_filters)
    ffmpeg.input(video).output(summarized_video, vf=vf_str, acodec='copy').run(overwrite_output=True)


def summarize_text(video, text):
    logging.info('Summarizing text with AI...')
    summary = ai_chat(f"Summarize this interview question in under 120 characters (NO QUOTATION MARKS): {text}")
    summarized_video = str(Path(video).with_stem(Path(video).stem + "_final"))
    logging.info("Adding summary text...")
    addText(video, summary, summarized_video)


def cut(in_data, out, video):
    logging.info("Starting cutting of segments...")
    cut_segments = []
    transcriptions = []
    for i, data in enumerate(in_data):
        if i % 2 == 0:
            transcriptions.append(transcription(data['start'], data['end'], i // 2, video, out))
        else:
            cut_segments.append({'start': data['start'], 'end': data['end']})
    os.makedirs(f"{out}/clips", exist_ok=True)
    for idx, seg in enumerate(cut_segments):
        start = seg["start"]
        end = seg["end"]
        duration = end - start
        output_path = f"{out}/clips/clip_{idx + 1:03d}.mov"
        logging.info(f"Cutting: {start:.2f}s → {end:.2f}s → {output_path}")
        (ffmpeg.input(video, ss=start, t=duration).output(output_path, codec="copy").run(overwrite_output=True))
        summarize_text(output_path, transcriptions[idx]["segments"][0]["text"])


def make_dir():
    logging.info("Making output directory...")
    o_dir = f"out/{Path(video_filename).stem}/"
    os.makedirs(o_dir, exist_ok=True)
    return o_dir


if __name__ == "__main__":
    transcription.model = whisperx.load_model("tiny", device="cpu", compute_type="int8", download_root="models/")
    video_filename = 'media/' + input("Enter the video file to perform split operation on: ")
    mp3_filename = video_filename
    out_dir = make_dir()
    logging.info(f"Converting {video_filename} to MP3...")
    cut(diarize(mp3_filename, out_dir), out_dir, mp3_filename)
    logging.info("All operations completed successfully! Check the 'out' directory for results.")
