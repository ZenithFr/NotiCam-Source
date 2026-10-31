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

def run_stage2_training(
    stage1_weights="/kaggle/working/checkpoints_stage1/best_stage1.weights.h5",
    train_pattern="data/tfrecords/train_*.tfrecord",
    val_pattern="data/tfrecords/val_*.tfrecord",
    output_dir="/kaggle/working/checkpoints_stage2",
    epochs=80,
    batch_size=32,
    initial_lr=1e-4
):
    print("=== Initializing Stage 2 Training (Full Unfreeze, 80 Epochs Cosine Decay) ===")
    os.makedirs(output_dir, exist_ok=True)

    mixed_precision.set_global_policy('mixed_float16')

    # Build model
    model = AuxCamPortraitNet(input_shape=(384, 512, 3), decoder_ch=96, batch_size=None)

    # Load weights from Stage 1 if available
    if os.path.exists(stage1_weights):
        model.load_weights(stage1_weights)
        print(f"Loaded Stage 1 weights from {stage1_weights}")
    else:
        print(f"Notice: {stage1_weights} not found, initializing from scratch or intermediate checkpoint.")

    # Fully unfreeze all layers
    for layer in model.layers:
        layer.trainable = True

    print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights])}")

    # Cosine decay schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=epochs * 400, # approx 400 steps per epoch for ~13k images
        alpha=0.01 # decays to 1e-6
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    loss_fn = PortraitMattingLoss()

    augmenter = PortraitAugmenter()
    
    train_files = tf.io.gfile.glob(train_pattern)
    if not train_files:
        print("Notice: No TFRecords found. Generating synthetic dataset for graph verification...")
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
        curr_lr = optimizer.inner_optimizer.learning_rate(global_step).numpy() if hasattr(optimizer, 'inner_optimizer') else 0.0
        print(f"Epoch {epoch:02d}/{epochs:02d} ({dt:.1f}s) | LR: {curr_lr:.2e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f} | Val SAD: {avg_val_sad:.2f}")

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
            best_path = os.path.join(output_dir, "best_stage2.weights.h5")
            model.save_weights(best_path)
            print(f"  ⭐ New best Val IoU: {best_val_iou:.4f}! Saved -> {best_path}")

    final_path = os.path.join(output_dir, "stage2_final.weights.h5")
    model.save_weights(final_path)
    with open(os.path.join(output_dir, "history_stage2.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"=== Stage 2 Complete! Best Val IoU: {best_val_iou:.4f} ===")
    return model

if __name__ == "__main__":
    run_stage2_training(epochs=1)
