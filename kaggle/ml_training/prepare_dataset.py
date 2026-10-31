import os
import tensorflow as tf
from PIL import Image

TARGET_H = 384
TARGET_W = 512

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(rgb_bytes, alpha_bytes):
    feature = {
        'image_raw': _bytes_feature(rgb_bytes),
        'alpha_raw': _bytes_feature(alpha_bytes),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def process_and_write_tfrecords(pairs, output_prefix, shard_size=2000):
    """
    pairs: list of (rgb_path, alpha_path)
    output_prefix: e.g. "data/tfrecords/train"
    """
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    os.makedirs(os.path.dirname(os.path.abspath(output_prefix)), exist_ok=True)
    total = len(pairs)
    print(f"Processing {total} image-alpha pairs into {output_prefix} TFRecords...")

    writer = None
    written = 0
    failed = 0
    try:
        for rgb_p, alpha_p in pairs:
            try:
                # Close files promptly rather than keeping decoded originals open
                # across thousands of samples.
                with Image.open(rgb_p) as source:
                    img = source.convert("RGB").resize(
                        (TARGET_W, TARGET_H), Image.Resampling.BILINEAR)
                with Image.open(alpha_p) as source:
                    alpha = source.convert("L").resize(
                        (TARGET_W, TARGET_H), Image.Resampling.BILINEAR)
                serialized = serialize_example(img.tobytes(), alpha.tobytes())
            except (OSError, ValueError) as exc:
                failed += 1
                print(f"Warning: Failed processing {rgb_p}: {exc}")
                continue

            if written % shard_size == 0:
                if writer is not None:
                    writer.close()
                shard_path = f"{output_prefix}_{written // shard_size:03d}.tfrecord"
                writer = tf.io.TFRecordWriter(shard_path)
                print(f"  Starting shard {shard_path} ({written}/{total})...")
            writer.write(serialized)
            written += 1
    finally:
        if writer is not None:
            writer.close()
    if written == 0 and total:
        raise ValueError(f"No usable image-alpha pairs for {output_prefix} ({failed} failed)")
    print(f"Finished writing {written} samples in {(written + shard_size - 1) // shard_size} shards; {failed} skipped.")

def parse_tfrecord_fn(example_proto):
    feature_description = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'alpha_raw': tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_raw(parsed['image_raw'], tf.uint8)
    image = tf.reshape(image, [TARGET_H, TARGET_W, 3])
    image = tf.cast(image, tf.float32) / 255.0

    alpha = tf.io.decode_raw(parsed['alpha_raw'], tf.uint8)
    alpha = tf.reshape(alpha, [TARGET_H, TARGET_W, 1])
    alpha = tf.cast(alpha, tf.float32) / 255.0
    return image, alpha

def load_tfrecord_dataset(file_pattern, batch_size=32, is_training=True):
    paths = sorted(tf.io.gfile.glob(file_pattern))
    if not paths:
        raise FileNotFoundError(f"No TFRecords matched {file_pattern!r}")
    files = tf.data.Dataset.from_tensor_slices(paths)
    if is_training:
        files = files.shuffle(len(paths))
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    if is_training:
        dataset = dataset.shuffle(2048)
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

if __name__ == "__main__":
    print("Dataset preparation module loaded. Call process_and_write_tfrecords(pairs, prefix) with your dataset paths.")
