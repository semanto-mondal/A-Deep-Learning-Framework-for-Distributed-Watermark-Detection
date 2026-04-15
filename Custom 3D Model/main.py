import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_NUMA_DISABLED"] = "1"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "0"

import tensorflow as tf
import gc

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.experimental.set_synchronous_execution(True)


import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from data_loader import VideoDataset
from model import cusModel, PreModel
from trainer import Trainer
from config import CFG

# -----------------------------
# Global config
# -----------------------------
N_SPLITS = 5
RESULTS_DIR = "kfold_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def build_model():
    if CFG.model_select == "custom_model":
        return cusModel.build()
    elif CFG.model_select == "pre_model":
        model = PreModel.build()
        dummy = tf.random.normal((1, CFG.n_frames, 224, 224, 3))
        model(dummy)
        return model
    else:
        raise ValueError("Unknown model")

def main():
    # -----------------------------
    # Load full dataset into memory
    # -----------------------------
    dataset = VideoDataset()
    X, y = dataset.get_arrays()

    print(f"Total samples: {len(y)}")

    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=CFG.random_state
    )

    fold_results = []

    # -----------------------------
    # K-FOLD LOOP
    # -----------------------------
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        fold_id = fold + 1

        print("\n" + "=" * 60)
        print(f"🚀 VALIDATION {fold_id} / {N_SPLITS}")
        print("=" * 60)

        print(f"Train samples: {len(train_idx)}")
        print(f"Validation samples: {len(val_idx)}")

        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]


        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
            .shuffle(CFG.batch_size * 4) \
            .batch(CFG.batch_size)

        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)) \
            .batch(CFG.batch_size)

        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = 1
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

        train_ds = train_ds.with_options(options)
        val_ds = val_ds.with_options(options)


        # Build new model (IMPORTANT)
        model = build_model()

        # Unique paths per fold
        fold_ckpt = f"{RESULTS_DIR}/best_model_fold_{fold+1}.h5"
        plot_dir = f"{RESULTS_DIR}/plots_fold_{fold+1}"

        trainer = Trainer(
            model=model,
            train_ds=train_ds,
            valid_ds=val_ds,
            checkpoint_path=fold_ckpt,
            plot_dir=plot_dir,
            run_name=f"fold_{fold+1}"
        )

        history = trainer.train()
        val_loss, val_acc = trainer.evaluate()

        # Save results
        fold_results.append({
            "fold": fold + 1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "best_model": fold_ckpt
        })

        # Cleanup
        tf.keras.backend.clear_session()
        gc.collect()

    # -----------------------------
    # Save final results
    # -----------------------------
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(f"{RESULTS_DIR}/kfold_summary.csv", index=False)

    print("\n✅ K-Fold Cross Validation Complete")
    print(results_df)
    print("\n📊 Mean Accuracy:", results_df["val_accuracy"].mean())
    print("📉 Std Accuracy:", results_df["val_accuracy"].std())


if __name__ == "__main__":
    main()
