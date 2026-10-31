import os
import sys
import time
import json
import tensorflow as tf
from tensorflow.keras import mixed_precision

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_mnv2_lrlf import AuxCamPortraitNet
from augment import PortraitAugmenter
from losses import compute_luma, reconstruct_alpha, PortraitMattingLoss, compute_metrics
from prepare_dataset import load_tfrecord_dataset
from export_int8 import export_int8_model

def run_qat_training(
    stage2_weights="/kaggle/working/checkpoints_stage2/best_stage2.weights.h5",
    train_pattern="data/tfrecords/train_*.tfrecord",
    val_pattern="data/tfrecords/val_*.tfrecord",
    output_dir="/kaggle/working/checkpoints_qat",
    epochs=10,
    batch_size=32,
    lr=1e-5
):
    print("=== Initializing Stage QAT Training (10 Epochs @ 1e-5) ===")
    os.makedirs(output_dir, exist_ok=True)

    model = AuxCamPortraitNet(input_shape=(384, 512, 3), decoder_ch=96, batch_size=None)

    if os.path.exists(stage2_weights):
        model.load_weights(stage2_weights)
        print(f"Loaded Stage 2 weights from {stage2_weights}")
    else:
        print(f"Notice: Stage 2 weights not found at {stage2_weights}, continuing with base weights.")

    for layer in model.layers:
        layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = PortraitMattingLoss()
    augmenter = PortraitAugmenter()

    train_files = tf.io.gfile.glob(train_pattern)
    if not train_files:
        print("Notice: No TFRecords found. Generating synthetic dataset for verification...")
        def synth_gen():
            for _ in range(64):
                img = tf.random.uniform([384, 512, 3], 0.0, 1.0)
                alpha = tf.cast(img[..., 0:1] > 0.5, tf.float32)
                yield img, alpha
        train_ds = tf.data.Dataset.from_generator(synth_gen, output_signature=(
            tf.TensorSpec(shape=(384, 512, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(384, 512, 1), dtype=tf.float32)
        )).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = train_ds
    else:
        train_ds = load_tfrecord_dataset(train_pattern, batch_size=batch_size, is_training=True)
        train_ds = train_ds.map(augmenter.augment_train, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_ds = load_tfrecord_dataset(val_pattern, batch_size=batch_size, is_training=False)
        val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    @tf.function
    def train_step(x_rgb, y_alpha):
        luma = compute_luma(x_rgb)
        with tf.GradientTape() as tape:
            coeffs_low = model(x_rgb, training=True)
            alpha_pred = reconstruct_alpha(coeffs_low, luma)
            loss_val = loss_fn(y_alpha, alpha_pred)

        grads = tape.gradient(loss_val, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_val, alpha_pred

    best_val_iou = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss_total = 0.0
        train_steps = 0

        for x_batch, y_batch in train_ds:
            loss_v, pred = train_step(x_batch, y_batch)
            train_loss_total += loss_v.numpy()
            train_steps += 1

        avg_train_loss = train_loss_total / max(1, train_steps)

        # Val
        val_loss_total = 0.0
        val_iou_total = 0.0
        val_sad_total = 0.0
        val_steps = 0

        for x_val, y_val in val_ds:
            luma = compute_luma(x_val)
            coeffs_low = model(x_val, training=False)
            v_pred = reconstruct_alpha(coeffs_low, luma)
            v_loss = loss_fn(y_val, v_pred)
            iou, sad = compute_metrics(y_val, v_pred)
            val_loss_total += v_loss.numpy()
            val_iou_total += iou.numpy()
            val_sad_total += sad.numpy()
            val_steps += 1

        avg_val_loss = val_loss_total / max(1, val_steps)
        avg_val_iou = val_iou_total / max(1, val_steps)
        avg_val_sad = val_sad_total / max(1, val_steps)

        dt = time.time() - t0
        print(f"QAT Epoch {epoch:02d}/{epochs:02d} ({dt:.1f}s) | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | IoU: {avg_val_iou:.4f} | SAD: {avg_val_sad:.2f}")

        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            best_path = os.path.join(output_dir, "best_qat.weights.h5")
            model.save_weights(best_path)

    final_weights = os.path.join(output_dir, "best_qat.weights.h5")
    final_tflite = os.path.join(output_dir, "portrait_v1_int8.tflite")
    export_int8_model(weights_path=final_weights, output_path=final_tflite)
    print(f"=== QAT Complete! Exported -> {final_tflite} ===")

if __name__ == "__main__":
    run_qat_training(epochs=1)
