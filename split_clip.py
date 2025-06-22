from whisperx.diarize import DiarizationPipeline
from pathlib import Path
import whisperx
import ffmpeg
import json
import os


def convert_mp4_to_mp3(input_path: str, output_path: str):
    (ffmpeg.input(input_path).output(output_path, vn=None, acodec='libmp3lame', audio_bitrate='192k').run(overwrite_output=True))


def diarize(audio_file, out):
    print("Starting diarizing of segments...")
    audio = whisperx.load_audio(audio_file)
    diarize_model = DiarizationPipeline(use_auth_token="", device="cpu")
    diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=2)
    return clean(diarize_segments, out)


def save_json(data, filename, out):
    with open(out + filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved data to {filename}")


def clean(diarize_segments, out):
    print("Starting cleaning of segments...")
    clean_segments = []
    for row in diarize_segments.itertuples(index=False):
        clean_segments.append({"start": float(row.start), "end": float(row.end), "speaker": row.speaker})
    save_json(clean_segments, "diarized.json", out)
    return group(clean_segments, out)


def group(clean_segments, out):
    print("Starting grouping of segments...")
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


def cut(in_data, out):
    print("Starting cutting of segments...")
    cut_segments = []
    for i, data in enumerate(in_data):
        if i % 2 == 0:
            cut_segments.append({'start': data['start'], 'end': None})
        else:
            cut_segments[-1]['end'] = data['end']
    os.makedirs(f"{out}/clips", exist_ok=True)
    for idx, seg in enumerate(cut_segments):
        start = seg["start"]
        end = seg["end"]
        duration = end - start
        output_path = f"{out}/clips/clip_{idx + 1:03d}.mov"
        print(f"⏱️ Cutting: {start:.2f}s → {end:.2f}s → {output_path}")
        (ffmpeg.input(full_video_filename, ss=start, t=duration).output(output_path, codec="copy").run(overwrite_output=True))


def make_dir():
    print("Making output directory...")
    o_dir = f"out/{Path(video_filename).stem}/"
    os.makedirs(o_dir, exist_ok=True)
    return o_dir


if __name__ == "__main__":
    full_video_filename = 'media/' + input("Enter the full video file name: ")
    video_filename = 'media/' + input("Enter the video file name to perform split operation on: ")
    mp3_filename = str(Path(video_filename).with_suffix('.mp3'))
    out_dir = make_dir()
    print(f"Converting {video_filename} to MP3...")
    convert_mp4_to_mp3(video_filename, mp3_filename)
    cut(diarize(mp3_filename, out_dir), out_dir)
    print("All operations completed successfully! Check the 'out' directory for results.")
