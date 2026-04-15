# ================================
# HARD ENV LIMITS (MUST BE FIRST)
# ================================
import os

# ---- GPU control ----
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_NUMA_DISABLED"] = "1"

# ---- HARD THREAD LIMITS ----
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_GPU_THREAD_COUNT"] = "1"
os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "0"

# ================================
# IMPORTS (AFTER ENV VARS)
# ================================
import tensorflow as tf
import cv2
import numpy as np
import random
import gc
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    confusion_matrix,
)
from config import CFG
checkpoint_path = "/lustre/home/rchandraghosh/deep-video-classifier-main/Archive/best_model_fold_3.h5"

# ================================
# DISABLE OPENCV THREADS (CRITICAL)
# ================================
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# ================================
# TENSORFLOW THREAD CONTROL
# ================================
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.data.experimental.enable_debug_mode()

# ================================
# CONFIG
# ================================


# Base dataset root
# BASE = "/ibiscostorage/rchandraghosh/video_dataset/testing_dataset"
BASE = "/ibiscostorage/rchandraghosh/video_dataset/testing_dataset"

# ================================
# ALL TEST CATEGORIES
# Each entry: (category_name, original_folder, watermarked_folder, output_txt)
# ================================
TEST_CATEGORIES = [
    (
        "1 - Recompression",
        f"{BASE}/1_recompression/original",
        f"{BASE}/1_recompression/watermarked",
        "metrics_recompression.txt",
    ),
    (
        "2 - Transcoding",
        f"{BASE}/2_transcoding/original",
        f"{BASE}/2_transcoding/watermarked",
        "metrics_transcoding.txt",
    ),
    (
        "3 - Resizing",
        f"{BASE}/3_resizing/original",
        f"{BASE}/3_resizing/watermarked",
        "metrics_resizing.txt",
    ),
    (
        "4 - Cropping",
        f"{BASE}/4_cropping/original",
        f"{BASE}/4_cropping/watermarked",
        "metrics_cropping.txt",
    ),
    (
        "6 - Temporal Edits",
        f"{BASE}/6_temporal_edits/original",
        f"{BASE}/6_temporal_edits/watermarked",
        "metrics_temporal_edits.txt",
    ),
    # (
    #     "original",
    #     f"{BASE}/original",
    #     f"{BASE}/watermark",
    #     "metrics_original.txt",
    # ),
]


# ================================
# FRAME PREPROCESSING
# ================================
def format_frames(frame, output_size):
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame


# ================================
# VIDEO → FRAMES
# ================================
def frames_from_video_file(video_path, n_frames, output_size, frame_step):
    src = cv2.VideoCapture(str(video_path))
    if not src.isOpened():
        return None

    video_length = int(src.get(cv2.CAP_PROP_FRAME_COUNT))
    need_length  = 1 + (n_frames - 1) * frame_step

    start = 0 if need_length > video_length else random.randint(
        0, max(0, video_length - need_length)
    )
    src.set(cv2.CAP_PROP_POS_FRAMES, start)

    frames = []
    ret, frame = src.read()
    if not ret:
        src.release()
        return None

    frames.append(format_frames(frame, output_size))

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
            if not ret:
                break
        if ret:
            frames.append(format_frames(frame, output_size))
        else:
            frames.append(tf.zeros_like(frames[0]))

    src.release()
    frames = tf.stack(frames)
    frames = tf.reverse(frames, axis=[-1])   # BGR → RGB
    return frames.numpy()

def build_model(input_shape=(10, 224, 224, 3)):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv3D(32, 3, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling3D((1, 2, 2))(x)

    x = tf.keras.layers.Conv3D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 2))(x)

    x = tf.keras.layers.Conv3D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 2))(x)

    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs)




# ================================
# SINGLE VIDEO PREDICTION
# ================================
def predict_video(video_path, model):
    frames = frames_from_video_file(
        video_path,
        n_frames=CFG.n_frames,
        output_size=CFG.output_size,
        frame_step=CFG.frame_step,
    )
    if frames is None:
        return None, None

    frames = np.expand_dims(frames, axis=0)
    prob   = model(frames, training=False).numpy()[0][0]
    label  = int(prob > 0.5)
    return label, float(prob)


# ================================
# COLLECT PREDICTIONS FROM ONE FOLDER
# ================================
def collect_predictions(folder_path, ground_truth_label, model):
    true_labels  = []
    pred_labels  = []
    pred_probs   = []
    failed_files = []

    video_extensions = (".mp4", ".avi", ".mov", ".mkv")

    for root, _, files in os.walk(folder_path):
        for file in sorted(files):
            if not file.lower().endswith(video_extensions):
                continue

            video_path = os.path.join(root, file)
            print(f"  Processing: {file}")

            try:
                pred_label, prob = predict_video(video_path, model)
                if pred_label is not None:
                    true_labels.append(ground_truth_label)
                    pred_labels.append(pred_label)
                    pred_probs.append(prob)
                else:
                    failed_files.append(file)
            except Exception as e:
                print(f"    ⚠️  Failed: {file} — {e}")
                failed_files.append(file)

            gc.collect()

    return true_labels, pred_labels, pred_probs, failed_files


# ================================
# COMPUTE METRICS + WRITE REPORT
# ================================
def compute_and_save_metrics(
    category_name,
    original_folder,
    watermarked_folder,
    metrics_txt,
    model,
):
    print(f"\n{'='*55}")
    print(f"  CATEGORY: {category_name}")
    print(f"{'='*55}")

    # --- Collect from original folder (label = 0) ---
    print(f"\n[1/2] Reading ORIGINAL folder...")
    t1, p1, prob1, fail1 = collect_predictions(original_folder, 0, model)

    # --- Collect from watermarked folder (label = 1) ---
    print(f"\n[2/2] Reading WATERMARKED folder...")
    t2, p2, prob2, fail2 = collect_predictions(watermarked_folder, 1, model)

    # --- Combine both folders ---
    true_labels  = t1 + t2
    pred_labels  = p1 + p2
    pred_probs   = prob1 + prob2
    failed_files = fail1 + fail2

    if not true_labels:
        print("⚠️  No valid predictions — skipping metrics.")
        return

    # --- Compute metrics ---
    acc       = accuracy_score(true_labels, pred_labels)
    f1        = f1_score(true_labels, pred_labels, zero_division=0)
    recall    = recall_score(true_labels, pred_labels, zero_division=0)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    cm        = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    unique = set(true_labels)
    if len(unique) > 1:
        auc     = roc_auc_score(true_labels, pred_probs)
        auc_str = f"{auc:.4f}"
    else:
        auc_str = "N/A (only one class present)"

    total   = len(true_labels)
    correct = sum(t == p for t, p in zip(true_labels, pred_labels))

    # --- Build report ---
    lines = [
        "=" * 55,
        "  VIDEO WATERMARK DETECTION — METRICS REPORT",
        "=" * 55,
        f"  Timestamp          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Category           : {category_name}",
        f"  Original folder    : {original_folder}",
        f"  Watermarked folder : {watermarked_folder}",
        "-" * 55,
        f"  Total videos       : {total}",
        f"    Original         : {len(t1)}",
        f"    Watermarked      : {len(t2)}",
        f"  Correct            : {correct}",
        f"  Failed / skipped   : {len(failed_files)}",
        "-" * 55,
        f"  Accuracy           : {acc:.4f}  ({acc*100:.2f}%)",
        f"  Precision          : {precision:.4f}",
        f"  Recall             : {recall:.4f}",
        f"  F1 Score           : {f1:.4f}",
        f"  AUC-ROC            : {auc_str}",
        "-" * 55,
        "  Confusion Matrix",
        "  Rows = True label  |  Cols = Predicted label",
        f"  {'':>15} | {'original':>10} {'watermarked':>12}",
        f"  {'original':>15} | {tn:>10}  {fp:>10}",
        f"  {'watermarked':>15} | {fn:>10}  {tp:>10}",
        "-" * 55,
        f"  TP (watermarked correctly detected)  : {tp}",
        f"  TN (original correctly detected)     : {tn}",
        f"  FP (original wrongly called wm)      : {fp}",
        f"  FN (watermarked missed)              : {fn}",
    ]

    if failed_files:
        lines += ["-" * 55, "  Failed files:"]
        for f in failed_files:
            lines.append(f"    - {f}")

    lines.append("=" * 55)

    report = "\n".join(lines)
    print("\n" + report + "\n")

    with open(metrics_txt, "w") as fh:
        fh.write(report + "\n")

    print(f"✅ Metrics saved to: {metrics_txt}\n")


# ================================
# MAIN
# ================================
if __name__ == "__main__":
    # --- Load model ---
    model = build_model()
    dummy = np.zeros((1, CFG.n_frames, *CFG.output_size, 3))
    model(dummy)
    model.load_weights(checkpoint_path)
    print("✅ Model loaded successfully!")

    # ----------------------------------------------------------------
    # Option A: Run ALL categories (default)
    # ----------------------------------------------------------------
    categories_to_run = TEST_CATEGORIES

    # ----------------------------------------------------------------
    # Option B: Run only specific categories
    # Uncomment below and comment out Option A
    #
    # Index: 0=Recompression, 1=Transcoding, 2=Resizing,
    #        3=Cropping,       4=Temporal Edits
    # ----------------------------------------------------------------
    # categories_to_run = [TEST_CATEGORIES[0]]                        # Recompression only
    # categories_to_run = [TEST_CATEGORIES[0], TEST_CATEGORIES[1]]    # Recompression + Transcoding
    # categories_to_run = [TEST_CATEGORIES[2], TEST_CATEGORIES[3]]    # Resizing + Cropping

    # ----------------------------------------------------------------
    for category_name, orig_folder, wm_folder, out_txt in categories_to_run:
        compute_and_save_metrics(
            category_name=category_name,
            original_folder=orig_folder,
            watermarked_folder=wm_folder,
            metrics_txt=out_txt,
            model=model,
        )

    print("\n🎉 All categories completed!")