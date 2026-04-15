import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from config import CFG
import os

class Trainer:
    def __init__(self, model, train_ds, valid_ds, checkpoint_path=None, plot_dir="plots", run_name=None):
        self.model = model
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.plot_dir = plot_dir
        self.run_name = run_name
        os.makedirs(self.plot_dir, exist_ok=True)  # create folder if not exists
        # allow override per-run checkpoint path
        from config import CFG
        self.checkpoint_path = checkpoint_path or CFG.checkpoint_path

    def train(self):
        from config import CFG
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            self.checkpoint_path, monitor="val_accuracy", mode="max", save_best_only=True
        )

        history = self.model.fit(
            self.train_ds, epochs=CFG.epochs, validation_data=self.valid_ds, callbacks=[checkpoint]
        )

        # load best weights from this run's checkpoint
        try:
            self.model.load_weights(self.checkpoint_path)
        except Exception:
            pass

        self.plot_history(history)
        return history

    def evaluate(self):
        val_loss, val_acc = self.model.evaluate(self.valid_ds)
        print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.4f}")
        return val_loss, val_acc

    def plot_history(self, history):
        run_prefix = f"{self.run_name}_" if self.run_name else ""
        for metrics in [("loss", "val_loss"), ("accuracy", "val_accuracy")]:
            df = pd.DataFrame(history.history, columns=metrics)
            ax = df.plot(title=f"{metrics[0]} vs {metrics[1]}")
            fig = ax.get_figure()

            save_path = os.path.join(self.plot_dir, f"{run_prefix}{metrics[0]}_plot.png")
            fig.savefig(save_path)
            print(f"✅ Saved plot: {save_path}")

            plt.close(fig)  # close to avoid memory leak
