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

def run_stage1_training(
    train_pattern="data/tfrecords/train_*.tfrecord",
    val_pattern="data/tfrecords/val_*.tfrecord",
    output_dir="/kaggle/working/checkpoints_stage1",
    epochs=20,
    batch_size=32,
    lr=1e-3
):
    print("=== Initializing Stage 1 Training (Encoder Frozen, 20 Epochs) ===")
    os.makedirs(output_dir, exist_ok=True)

    # Set mixed precision for NVIDIA T4 throughput
    mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision policy:", mixed_precision.global_policy())

    # Build model
    model = AuxCamPortraitNet(input_shape=(384, 512, 3), decoder_ch=96, batch_size=None)
    
    # Freeze encoder stages (Conv2D and Depthwise up to Stage 4)
    # The encoder comprises layers up to the ASPP branches
    encoder_layer_count = 0
    for layer in model.layers:
        if "aspp" in layer.name or "deep" in layer.name or "fuse" in layer.name or "coeffs" in layer.name or "low_proj" in layer.name:
            layer.trainable = True
        else:
            # First ~30 layers are backbone stages 0-4
            layer.trainable = False
            encoder_layer_count += 1
            
    print(f"Total layers: {len(model.layers)}, Frozen encoder layers: {encoder_layer_count}")
    print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights])}")

    # Optimizer & Loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    loss_fn = PortraitMattingLoss()

    # Data pipelines
    augmenter = PortraitAugmenter()
    
    # Check if files exist, fallback to synthetic data if testing
    train_files = tf.io.gfile.glob(train_pattern)
    if not train_files:
        print("Notice: No TFRecords found at pattern. Generating synthetic dataset for graph verification...")
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

    # Metrics history
    history = []
    best_val_iou = 0.0

    @tf.function
    def train_step(x_rgb, y_alpha):
        luma = compute_luma(x_rgb)
        with tf.GradientTape() as tape:
            coeffs_low = model(x_rgb, training=True)
            coeffs_low = tf.cast(coeffs_low, tf.float32)
            alpha_pred = reconstruct_alpha(coeffs_low, luma)
            loss_val = loss_fn(y_alpha, alpha_pred)
            scaled_loss = optimizer.scale_loss(loss_val)

        grads = tape.gradient(scaled_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_val, alpha_pred

    @tf.function
    def val_step(x_rgb, y_alpha):
        luma = compute_luma(x_rgb)
        coeffs_low = model(x_rgb, training=False)
        coeffs_low = tf.cast(coeffs_low, tf.float32)
        alpha_pred = reconstruct_alpha(coeffs_low, luma)
        loss_val = loss_fn(y_alpha, alpha_pred)
        return loss_val, alpha_pred

    global_step = 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss_total = 0.0
        train_steps = 0

        for x_batch, y_batch in train_ds:
            loss_v, pred = train_step(x_batch, y_batch)
            train_loss_total += loss_v.numpy()
            train_steps += 1
            global_step += 1

            if global_step % 500 == 0:
                ckpt_path = os.path.join(output_dir, f"model_step_{global_step}.weights.h5")
                model.save_weights(ckpt_path)
                print(f"  [Step {global_step}] Checkpoint saved -> {ckpt_path}")

        avg_train_loss = train_loss_total / max(1, train_steps)

        # Validation loop
        val_loss_total = 0.0
        val_iou_total = 0.0
        val_sad_total = 0.0
        val_steps = 0

        for x_val, y_val in val_ds:
            v_loss, v_pred = val_step(x_val, y_val)
            iou, sad = compute_metrics(y_val, v_pred)
            val_loss_total += v_loss.numpy()
            val_iou_total += iou.numpy()
            val_sad_total += sad.numpy()
            val_steps += 1

        avg_val_loss = val_loss_total / max(1, val_steps)
        avg_val_iou = val_iou_total / max(1, val_steps)
        avg_val_sad = val_sad_total / max(1, val_steps)

        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{epochs:02d} ({dt:.1f}s) | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f} | Val SAD: {avg_val_sad:.2f}")

        record = {
            "epoch": epoch,
            "train_loss": float(avg_train_loss),
            "val_loss": float(avg_val_loss),
            "val_iou": float(avg_val_iou),
            "val_sad": float(avg_val_sad),
            "time_sec": dt
        }
        history.append(record)

        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            best_path = os.path.join(output_dir, "best_stage1.weights.h5")
            model.save_weights(best_path)
            print(f"  ⭐ New best Val IoU: {best_val_iou:.4f}! Saved -> {best_path}")

    # Save final stage1 weights
    final_path = os.path.join(output_dir, "stage1_final.weights.h5")
    model.save_weights(final_path)
    with open(os.path.join(output_dir, "history_stage1.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"=== Stage 1 Training Complete! Best Val IoU: {best_val_iou:.4f} ===")
    return model

if __name__ == "__main__":
    run_stage1_training(epochs=1) # 1 test epoch if run locally
