"""Cut KITE dialogue turns out of the KETI videos.

Each turn becomes a mono 16 kHz float32 wav. The metadata file is
``1_metadata_ser.csv``, with a stratified 80/10/10 train/dev/test split.
The original videos are not modified.

The source tree is the KETI release: one folder per part, and inside each
part a clip folder that holds ``{video_id}.mp4`` and
``{video_id}_interpolation.json``.

    python extractor.py \
      --root_dir /path/to/멀티모달_분야 \
      --output_audio_dir /path/to/KITE \
      --skip_feature_extraction
"""

import argparse
import json
import os
import re
import logging
import subprocess
import pickle  # Added for post-extraction key remapping
from collections import Counter
from datetime import datetime

import imageio_ffmpeg
import numpy as np
import pandas as pd
from scipy.io import wavfile
import cv2

# Try importing process_KITE; provide a mock fallback if it's missing in local environments
try:
    from process_KITE import process as extract_features
except ImportError:
    def extract_features(*args, **kwargs):
        print("[!] Warning: process_KITE feature extraction engine not found. Skipping feature compile.")

# Optional progress bar configuration
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    tqdm = None
    _HAS_TQDM = False

# ---------------------------
# FIND FFMPEG AUTOMATICALLY
# ---------------------------
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()


# ---------------------------
# SPEECH AUDIO EXTRACTION ENGINE
# ---------------------------

def extract_speech_segment_via_ffmpeg(video_path, out_file, start_sec, end_sec, target_sr=16000):
    """
    Cut a mono 16 kHz float32 WAV from the video at the dialogue timestamps.

    Seek is applied *after* opening the input so the cut is timestamp-accurate.
    Input seeking (`-ss` before `-i`) snaps to video keyframes and routinely
    grabs the wrong speaker on KETI's sub-2s utterances.
    """
    try:
        duration = end_sec - start_sec
        if duration <= 0.3:
            return False

        temp_out = out_file + ".tmp.wav"
        cmd = [
            FFMPEG_EXE,
            "-y",
            "-i", video_path,
            "-ss", f"{start_sec:.3f}",
            "-t", f"{duration:.3f}",
            "-vn",
            "-ac", "1",
            "-ar", str(target_sr),
            "-acodec", "pcm_f32le",
            "-af", "highpass=f=80",
            temp_out,
        ]

        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode != 0 or not os.path.exists(temp_out) or os.path.getsize(temp_out) == 0:
            if os.path.exists(temp_out):
                os.remove(temp_out)
            return False

        sr, data = wavfile.read(temp_out)
        data = np.asarray(data, dtype=np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        peak = float(np.max(np.abs(data))) if data.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0
        # Drop near-silent cuts (wrong seek / music bed / no speech).
        if peak < 1e-3 or rms < 1e-4:
            if os.path.exists(temp_out):
                os.remove(temp_out)
            return False

        # Peak-normalize so clip loudness does not leak into MFCCs as a fake emotion cue.
        data = data * (0.99 / peak)

        # if the file is less than 60KB in size, drop it - they contain less spectral information
        # TO BE TESTED TO SEE IF THEY IMPROVE ON PERFORMANCE - REMOVE THIS AFTER TESTING
        # if data.nbytes < 60 * 1024:
        #     if os.path.exists(temp_out):
        #         os.remove(temp_out)
        #     return False


        wavfile.write(out_file, sr, data)
        if os.path.exists(temp_out):
            os.remove(temp_out)
        return True

    except Exception as e:
        print(f"\n-> Extraction saving error on {os.path.basename(video_path)}: {e}")
        for path in (out_file, out_file + ".tmp.wav"):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        return False


def convert_time_to_seconds(time_str):
    if time_str is None:
        return 0.0
    text = str(time_str).strip()
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S"):
        try:
            t = datetime.strptime(text, fmt).time()
            return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
        except ValueError:
            continue
    return 0.0

# get FPS  directly from video file using ffmpeg

def get_actual_fps(video_path, fallback=29.97, output_audio_dir=None):
    """Retrieve actual frame rate from the MP4 container."""
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps and fps > 0:
            return fps
        else:
            # log to a file called fps_errors.log in the same directory as the data path
            logging.basicConfig(filename=os.path.join(output_audio_dir, "fps_errors.log"), level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
            logging.warning(f"FPS extraction failed for {os.path.basename(video_path)}. Using fallback FPS: {fallback}")
    return fallback
# ---------------------------
# KETI PARSERS (DIRECT LABEL UPDATE)
# ---------------------------

def parse_keti_for_audio_ser(json_path, output_audio_dir):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    video_id = data['common_info']['file_id']
    fps = get_actual_fps(os.path.join(os.path.dirname(json_path), f"{video_id}.mp4"), output_audio_dir=output_audio_dir)

    # Parse visual info fallback matrix
    # Parse visual info fallback matrix
    visual_rows = []

    # KETI interpolation frame IDs are absolute/source frame IDs,
    # while dialogue timestamps are relative to the clip.
    all_frame_ids = [
        visual["frame_id"]
        for shot in data["shot_infos"]
        for visual in shot["visual_infos"]
    ]

    base_frame_id = min(all_frame_ids) if all_frame_ids else 0

    for shot in data["shot_infos"]:
        for visual in shot["visual_infos"]:
            frame_id = visual["frame_id"]

            # IMPORTANT:
            # Do NOT use frame_id / fps directly.
            # The dialogue timestamps start at 0 seconds, whereas
            # KETI frame IDs start at an absolute frame number.
            timestamp_secs = (frame_id - base_frame_id) / fps

            for person in visual["persons"]:
                emotions = person.get("person_info", {}).get("emotion", {})

                dominant_emotion = (
                    max(emotions, key=emotions.get)
                    if emotions
                    else None
                )

                visual_rows.append({
                    "frame_id": frame_id,
                    "timestamp_seconds": timestamp_secs,
                    "dominant_emotion": dominant_emotion
                })

    visual_df = pd.DataFrame(visual_rows)
    dialogue_rows = []
    
    if 'dialogue_infos' in data:
        for dial in data['dialogue_infos']:
            start = convert_time_to_seconds(dial['start_time'])
            end = convert_time_to_seconds(dial['end_time'])

            if end <= start:
                continue

            # EMOTION IS EXTRACTED FROM HERE TO BE STORED FOR PSEUDO-LABELLING.
            direct_label = dial.get('emotion') or dial.get('intent') or dial.get('label')

            dialogue_rows.append({
                "video_id": video_id,
                "dialogue_id": dial['dialogue_id'],
                "speaker_id": dial['speaker_id'],
                "utterance": dial['utterance'],
                "start_seconds": start,
                "end_seconds": end,
                "duration_seconds": round(end - start, 3),
                "dialogue_label": direct_label
            })

    return visual_df, pd.DataFrame(dialogue_rows)


def collect_json_files(root_dir):
    json_files = []
    if not os.path.exists(root_dir):
        return json_files
    
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path) and folder_name != "extracted_ser_audio":
            # ONLY TAKE THE INTERPOLATION FILES ALONE.
            json_path = os.path.join(folder_path, f"{folder_name}_interpolation.json")
            if os.path.exists(json_path):
                json_files.append(json_path)
    return json_files


# ---------------------------
# CORE EXTRACTION CONTROLLER
# ---------------------------

def extract_kite_audio(root_dir, output_audio_dir, sample_rate):
    os.makedirs(output_audio_dir, exist_ok=True)
    json_files = collect_json_files(root_dir)
    
    ser_rows = []
    stats = {
        "total_dialogues_found": 0,
        "failed_extractions": 0,
        "skipped_no_label": 0
    }

    if not json_files:
        return pd.DataFrame(), stats

    if _HAS_TQDM:
        file_iter = tqdm(json_files, desc="  └─ Progress", leave=False, unit="json")
    else:
        file_iter = json_files

    for file_path in file_iter:
        try:
            v_df, d_df = parse_keti_for_audio_ser(file_path, output_audio_dir)
            if d_df.empty:
                continue

            video_id = d_df["video_id"].iloc[0]
            video_path = os.path.join(root_dir, video_id, f"{video_id}.mp4")

            if not os.path.exists(video_path):
                continue

            for _, dial in d_df.iterrows():
                stats["total_dialogues_found"] += 1
                start = float(dial["start_seconds"])
                end = float(dial["end_seconds"])
                dial_id = dial["dialogue_id"]

                emotion_label = dial.get("dialogue_label")
                
                if not emotion_label or pd.isna(emotion_label):
                    if not v_df.empty:
                        frames = v_df[(v_df["timestamp_seconds"] >= start) & (v_df["timestamp_seconds"] <= end)]
                        if not frames.empty:
                            emo = frames["dominant_emotion"].dropna().mode()
                            if not emo.empty:
                                emotion_label = emo.iloc[0]
                
                if not emotion_label or pd.isna(emotion_label):
                    stats["skipped_no_label"] += 1
                    # use logger to log the skipped dialogue video_id, dialogue_id to a dropped_dialogues.log file in the output_audio_dir
                    logging.basicConfig(filename=os.path.join(output_audio_dir, "dropped_dialogues.log"), level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
                    logging.warning(f"Skipped dialogue with no label: video_id={video_id}, dialogue_id={dial_id}")
                    continue

                out_file = os.path.join(output_audio_dir, f"{video_id}_dial_{dial_id}.wav")

                if os.path.exists(out_file):
                    try: os.remove(out_file)
                    except: pass

                success = extract_speech_segment_via_ffmpeg(video_path, out_file, start, end, target_sr=sample_rate)

                if success and os.path.exists(out_file):
                    ser_rows.append({
                        "audio_path": out_file, "video_id": video_id, "dialogue_id": dial_id,
                        "speaker_id": dial["speaker_id"], "utterance": dial["utterance"],
                        "duration": dial["duration_seconds"], "emotion_label": emotion_label
                    })
                else:
                    stats["failed_extractions"] += 1

        except Exception as e:
            pass

    return pd.DataFrame(ser_rows), stats


# ---------------------------
# POST-PROCESSING KEYS PATCHER
# ---------------------------

def remap_test_features_in_pickle(pkl_path):
    """
    Corrects the keys inside the test pickle structure.
    Since we pass test data to process_KITE with split_rate=1.0, 
    the library tags them internally as training arrays.
    This safely remaps all 'train' references to 'test'.
    """
    if not os.path.exists(pkl_path):
        return

    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, dict):
            return

        remapped_data = {}
        changes_made = False

        for key, val in data.items():
            if "train" in key.lower():
                # Swap train/training prefixes with test/testing prefixes
                new_key = key.lower().replace("train", "test")
                remapped_data[new_key] = val
                changes_made = True
            else:
                remapped_data[key] = val

        if changes_made:
            with open(pkl_path, "wb") as f:
                pickle.dump(remapped_data, f)
            print(f"   [✔] Internal training-key mapping corrected to testing-key mapping inside: {os.path.basename(pkl_path)}")
    except Exception as e:
        print(f"   [!] Key correction process failed for {os.path.basename(pkl_path)}: {e}")


# ---------------------------
# PIPELINE RUNNER
# ---------------------------

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Float32 Speech Pipeline Engine.")
    # e.g /media/computergeek/SER-datasets/Multimodal_Data_20260416/멀티모달/멀티모달_분야/멀티모달_Part_01"
    parser.add_argument("--root_dir", default="/media/computergeek/SER-datasets/Multimodal_Data_20260416/멀티모달/멀티모달_분야", type=str)
    parser.add_argument("--output_audio_dir", default="data/KITE", type=str)
    parser.add_argument("--features_to_use", default="mfcc", type=str)
    parser.add_argument("--sample_rate", default=16000, type=int)
    parser.add_argument("--nmfcc", default=26, type=int)
    parser.add_argument("--segment_length", default=1.8, type=float)
    parser.add_argument("--train_overlap", default=1.6, type=float)
    parser.add_argument("--test_overlap", default=1.6, type=float)
    parser.add_argument(
        "--split_rate",
        default=0.8,
        type=float,
        help="train split rate (train/dev/test = 0.8/0.1/remainder)",
    )
    parser.add_argument(
        "--dev_rate",
        default=0.1,
        type=float,
        help="dev (validation) split rate; test is the remainder",
    )
    parser.add_argument("--skip_feature_extraction", action="store_true")
    parser.add_argument("--features_file_name", default=None, type=str)
    parser.add_argument(
            "--use_data_part",
            default="all",
            type=str,
            help="specify a part of the data to use (e.g., 'part1', 'part2', etc.)",
        )
    return parser


def main():
    args = build_arg_parser().parse_args()
    output_audio_dir = args.output_audio_dir
    root_dir = args.root_dir

    # if the root directory does not exist, print a warning and exit
    if not os.path.exists(root_dir):
        print(f"[!] Target root directory path does not exist: {root_dir}")
        return
    
    all_dfs = []
    
    global_stats = {
        "total_parts_processed": 0,
        "total_dialogues_found": 0,
        "total_audio_extracted": 0,
        "failed_extractions": 0,
        "skipped_no_label": 0,
        "total_duration_seconds": 0.0
    }



    # print(f"Total parts: {len(parts)}")
    # print(f"Parts: {parts}")

    # for i, part in enumerate(iterator, 1):
    # pass --root_dir to the part_path
    # use data part if specified, --use_data_part as part1 to 11
        # use_data_part is "all
    parts_list = ["Part_01", "Part_02", "Part_03", "Part_04", "Part_05", "Part_06", "Part_07", "Part_08", "Part_09", "Part_10", "Part_11"]        

    print("use_data_part boolean:", args.use_data_part == "all")
    if args.use_data_part == "all":
        if not os.path.exists(args.root_dir):
            print(f"[!] Target root directory path does not exist: {args.root_dir}")
            return

        parts = sorted(os.listdir(args.root_dir))
        parts = [p for p in parts if os.path.isdir(os.path.join(args.root_dir, p))]

        print("\n" + "="*70)
        print(f"INITIALIZING PIPELINE: Processing {len(parts)} sub-directories...")
        print("="*70)

        if _HAS_TQDM and parts:
            if args.use_data_part == "all":
                iterator = tqdm(parts, desc="Global Folders Pipeline", unit="folder")
            else:
                iterator = tqdm(1, desc="Only extracting the specified part", unit="folder")
        else:
            iterator = parts
        print("Processing all parts in the root directory...")
        for i, part in enumerate(iterator, 1):
            part_path = os.path.join(args.root_dir, part)

            if not _HAS_TQDM:
                print(f"[{i}/{len(parts)}] Scanning dataset branch: {part}")

            df, part_stats = extract_kite_audio(part_path, output_audio_dir, args.sample_rate)

            global_stats["total_parts_processed"] += 1
            global_stats["total_dialogues_found"] += part_stats["total_dialogues_found"]
            global_stats["failed_extractions"] += part_stats["failed_extractions"]
            global_stats["skipped_no_label"] += part_stats["skipped_no_label"]

            if not df.empty:
                global_stats["total_audio_extracted"] += len(df)
                global_stats["total_duration_seconds"] += df["duration"].sum()
                all_dfs.append(df)
    elif args.use_data_part in parts_list:
        # generate the part path by attaching use_data_part to the root_dir,
        # e.g Part_01 would be /media/computergeek/SER-datasets/Multimodal_Data_20260416/멀티모달/멀티모달_분야/멀티모달_Part_01
        if not os.path.exists(args.root_dir):
            print(f"[!] Target root directory path does not exist: {args.root_dir}")
            return

        parts = sorted(os.listdir(args.root_dir))
        parts = [p for p in parts if os.path.isdir(os.path.join(args.root_dir, p))]

        print("\n" + "="*70)
        print(f"INITIALIZING PIPELINE: Processing {len(parts)} sub-directories...")
        print("="*70)

        if _HAS_TQDM and parts:
            if args.use_data_part == "all":
                iterator = tqdm(parts, desc="Global Folders Pipeline", unit="folder")
            else:
                iterator = tqdm(1, desc="Only extracting the specified part", unit="folder")
        else:
            iterator = parts
        print(f"Processing only the specified part: {args.use_data_part}")
        full_part_name = f"멀티모달_{args.use_data_part}"
        part_path = os.path.join(args.root_dir, full_part_name)
        print(f"-----Processing part----: {part_path}")

        if not _HAS_TQDM:
            print(f"[{i}/{len(parts)}] Scanning dataset branch: {part}")

        df, part_stats = extract_kite_audio(part_path, output_audio_dir, args.sample_rate)

        global_stats["total_parts_processed"] += 1
        global_stats["total_dialogues_found"] += part_stats["total_dialogues_found"]
        global_stats["failed_extractions"] += part_stats["failed_extractions"]
        global_stats["skipped_no_label"] += part_stats["skipped_no_label"]

        if not df.empty:
            global_stats["total_audio_extracted"] += len(df)
            global_stats["total_duration_seconds"] += df["duration"].sum()
            all_dfs.append(df)

    else:
        # The optionfor data part specified cannot be handled, so we skip processing and log a warning.
        print(f"[!] Warning: The specified data part '{args.use_data_part}' is not recognized. No extraction performed. make sure to specify a valid part name or use 'all' to process all parts.")
        # e.g running python train_KITE.py --use_data_part Part_04 would append "멀티모달_Part_04" to the root_dir and process that part alone.
        print(f"[!] Valid parts are: {', '.join(parts_list)} or 'all' to process all parts.")

        exit(1)

    if all_dfs:
        df = pd.concat(all_dfs, ignore_index=True)
    else:
        df = pd.DataFrame()

    # Capture original dataset counts before splitting
    initial_counts = df["emotion_label"].value_counts() if not df.empty else pd.Series(dtype=int)

    # -----------------------------------------------------------------
    # STRATIFIED DATA SPLITTING (80% train / 10% dev / 10% test)
    # -----------------------------------------------------------------
    train_df = pd.DataFrame()
    dev_df = pd.DataFrame()
    test_df = pd.DataFrame()
    
    if not df.empty:
        # Stratified split that isolates train/dev/test structures
        train_groups = []
        dev_groups = []
        test_groups = []
        train_rate = args.split_rate
        dev_rate = args.dev_rate
        remaining_rate = max(1.0 - train_rate, 1e-12)
        dev_frac_of_remaining = min(max(dev_rate / remaining_rate, 0.0), 1.0)
        
        for emo, group in df.groupby("emotion_label"):
            if len(group) >= 3:
                train_g = group.sample(frac=train_rate, random_state=42)
                leftover = group.drop(train_g.index)
                if leftover.empty:
                    dev_g = leftover
                    test_g = leftover
                else:
                    dev_g = leftover.sample(frac=dev_frac_of_remaining, random_state=42)
                    test_g = leftover.drop(dev_g.index)
            elif len(group) == 2:
                shuffled = group.sample(frac=1.0, random_state=42)
                train_g = shuffled.iloc[[0]]
                dev_g = shuffled.iloc[[1]]
                test_g = shuffled.iloc[0:0]
            else:
                train_g = group
                dev_g = group.iloc[0:0]
                test_g = group.iloc[0:0]
            train_groups.append(train_g)
            dev_groups.append(dev_g)
            test_groups.append(test_g)
            
        train_df = pd.concat(train_groups, ignore_index=True)
        dev_df = pd.concat(dev_groups, ignore_index=True)
        test_df = pd.concat(test_groups, ignore_index=True)
        
        train_df["split"] = "train"
        dev_df["split"] = "dev"
        test_df["split"] = "test"
        
        # Create unified dataframe representation
        df = pd.concat([train_df, dev_df, test_df], ignore_index=True)

    post_train_counts = train_df["emotion_label"].value_counts() if not train_df.empty else pd.Series(dtype=int)
    post_dev_counts = dev_df["emotion_label"].value_counts() if not dev_df.empty else pd.Series(dtype=int)
    post_test_counts = test_df["emotion_label"].value_counts() if not test_df.empty else pd.Series(dtype=int)

    # ---------------------------------------------
    # PIPELINE PERFORMANCE & STATS DISPLAY
    # ---------------------------------------------
    print("\n" + "="*70)
    print("                      DATASET METRICS DASHBOARD                 ")
    print("="*70)
    print(f"Branches Processed     : {global_stats['total_parts_processed']}")
    print(f"Total Dialogue Logs    : {global_stats['total_dialogues_found']}")
    print(f"Skipped (No Label Found): {global_stats['skipped_no_label']}")
    print(f"Failed FFmpeg Writes   : {global_stats['failed_extractions']}")
    print(f"Successful Audio Files : {global_stats['total_audio_extracted']}")
    
    success_rate = 0.0
    if global_stats['total_dialogues_found'] > 0:
        success_rate = (global_stats['total_audio_extracted'] / global_stats['total_dialogues_found']) * 100
    print(f"Pipeline Conversion SR : {success_rate:.2f}%")
    
    total_mins = global_stats['total_duration_seconds'] / 60.0
    print(f"Total Audio Duration   : {global_stats['total_duration_seconds']:.2f} sec ({total_mins:.2f} mins)")
    print("-"*70)
    
    print("EMOTION LABEL DISTRIBUTION (RAW ARCHIVE):")
    if not initial_counts.empty:
        for emo, count in initial_counts.items():
            print(f"  - {emo:<15} : {count} clips")
    else:
        print("  (No speech samples matching criteria found)")

    if not post_train_counts.empty:
        print("-"*70)
        print("TRAINING SET DISTRIBUTION:")
        for emo, count in post_train_counts.items():
            print(f"  - {emo:<15} : {count} clips")

    if not post_dev_counts.empty:
        print("-"*70)
        print("DEV SET DISTRIBUTION (MODEL SELECTION):")
        for emo, count in post_dev_counts.items():
            print(f"  - {emo:<15} : {count} clips")
            
    if not post_test_counts.empty:
        print("-"*70)
        print("PRISTINE TEST DISTRIBUTION (LEAK-PROOF BENCHMARK):")
        for emo, count in post_test_counts.items():
            print(f"  - {emo:<15} : {count} clips")
    print("="*70 + "\n")

    # If dataset contains no targets or features are flagged to skip, stop here
    if df.empty or args.skip_feature_extraction:
        out_csv = os.path.join(output_audio_dir, "1_metadata_ser.csv")
        if not df.empty:
            df.to_csv(out_csv, index=False, encoding="utf-8-sig")
            train_df.to_csv(os.path.join(output_audio_dir, "metadata_train.csv"), index=False, encoding="utf-8-sig")
            if not dev_df.empty:
                dev_df.to_csv(os.path.join(output_audio_dir, "metadata_dev.csv"), index=False, encoding="utf-8-sig")
            test_df.to_csv(os.path.join(output_audio_dir, "metadata_test.csv"), index=False, encoding="utf-8-sig")
            print(f"Saved split-aware output configuration sheets -> {output_audio_dir}")
        print("Pipeline execution complete (Audio extraction and split stage completed).")
        return

    # Commit structured layout files to disk
    df.to_csv(os.path.join(output_audio_dir, "1_metadata_ser.csv"), index=False, encoding="utf-8-sig")

    print("\n[✔] Speech segments have been extracted from dialogues and Metadata CSV files generated successfully.")
    # next step step is splitting, augmentation and training, which is handled by train_KITE.py
    print("[✔] Proceeding to feature extraction and model training stage...")

if __name__ == "__main__":
    main()
