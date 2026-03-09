import warnings
warnings.filterwarnings("ignore", message="Ignoring cached namespace 'core'")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hdmf_zarr import NWBZarrIO

# =========================
# 1. Load metadata
# =========================
metadata = pd.read_csv('/data/metadata/bci_metadata.csv')

# 五个 subject_id
subject_ids = [731015, 740369, 754303, 766719, 767715]

# =========================
# 2. Loop through each subject
# =========================
for subject_id in subject_ids:
    mouse_meta = metadata[metadata['subject_id'] == subject_id].sort_values(by='session_number')

    print(f"\n========== Mouse {subject_id} ==========")
    print(mouse_meta[['name', 'session_number', 'session_date', 'ophys_fov']])

    # 如果这个鼠没有数据，就跳过
    if len(mouse_meta) == 0:
        print(f"No metadata found for mouse {subject_id}")
        continue

    # =========================
    # 3. Prepare containers
    # =========================
    all_threshold = []
    all_trials = []
    session_boundaries = []
    session_labels = []
    used_session_numbers = []

    trial_offset = 0
    colors = plt.cm.Set2(np.linspace(0, 1, len(mouse_meta)))

    plt.figure(figsize=(14, 8))

    # =========================
    # 4. Loop through sessions
    # =========================
    for color, (_, row) in zip(colors, mouse_meta.iterrows()):
        session_name = row['name']
        session_num = row['session_number']

        session_dir = os.path.join('/data/', 'brain-computer-interface-v2', session_name)

        if not os.path.exists(session_dir):
            print(f"Skipped missing session folder: {session_dir}")
            continue

        nwb_candidates = [f for f in os.listdir(session_dir) if 'nwb' in f]
        if len(nwb_candidates) == 0:
            print(f"No NWB file found in: {session_dir}")
            continue

        nwb_file = nwb_candidates[0]
        nwb_path = os.path.join(session_dir, nwb_file)

        try:
            with NWBZarrIO(str(nwb_path), 'r') as io:
                nwbfile = io.read()
                trials = nwbfile.stimulus["Trials"].to_dataframe()

                # threshold crossing time
                threshold_time = np.array(trials["threshold_crossing_times"], dtype=float)

                # 去掉 NaN
                mask = np.isfinite(threshold_time)
                threshold_time = threshold_time[mask]

                if len(threshold_time) == 0:
                    print(f"No valid threshold data in session: {session_name}")
                    continue

                # cumulative trial number
                trial_num = np.arange(trial_offset + 1, trial_offset + len(threshold_time) + 1)

                # raw
                plt.plot(
                    trial_num,
                    threshold_time,
                    'o-',
                    color=color,
                    alpha=0.35,
                    markersize=3,
                    linewidth=1,
                    label=f"Session {int(session_num)} raw"
                )

                # rolling average
                rolling = pd.Series(threshold_time).rolling(window=10, min_periods=1).mean()

                plt.plot(
                    trial_num,
                    rolling,
                    color=color,
                    linewidth=3,
                    label=f"Session {int(session_num)} rolling avg"
                )

                # save
                all_threshold.extend(threshold_time)
                all_trials.extend(trial_num)
                session_boundaries.append(trial_num[-1])
                session_labels.append((trial_num[0] + trial_num[-1]) / 2)
                used_session_numbers.append(session_num)

                trial_offset += len(threshold_time)

                print(f"Added session {int(session_num)} | trials used: {len(threshold_time)}")

        except Exception as e:
            print(f"Error reading {session_name}: {e}")
    # =========================
    # 5. Add session separators
    # =========================
    for boundary in session_boundaries[:-1]:
        plt.axvline(boundary + 0.5, color='gray', linestyle='--', alpha=0.6)

    # 图顶端标注 session
    ymax = plt.ylim()[1]
    for label_x, session_num in zip(session_labels, used_session_numbers):
        if pd.notna(session_num):
            plt.text(label_x, ymax * 0.98, f"S{int(session_num)}",
                     ha='center', va='top', fontsize=11)

    # =========================
    # 6. Final formatting
    # =========================
    plt.xlabel("Cumulative Trial Number", fontsize=12)
    plt.ylabel("Threshold Crossing Time (s)", fontsize=12)
    plt.title(f"Mouse {subject_id}: Threshold Crossing Time Across Sessions", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()