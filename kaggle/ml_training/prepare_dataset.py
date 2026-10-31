import os
import glob
import random
import tensorflow as tf
from PIL import Image
import numpy as np

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
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    total = len(pairs)
    print(f"Processing {total} image-alpha pairs into {output_prefix} TFRecords...")

    shard_idx = 0
    writer = None

    for i, (rgb_p, alpha_p) in enumerate(pairs):
        if i % shard_size == 0:
            if writer:
                writer.close()
            shard_path = f"{output_prefix}_{shard_idx:03d}.tfrecord"
            writer = tf.io.TFRecordWriter(shard_path)
            shard_idx += 1
            print(f"  Starting shard {shard_path} ({i}/{total})...")

        try:
            # Load and resize RGB
            img = Image.open(rgb_p).convert("RGB")
            img = img.resize((TARGET_W, TARGET_H), Image.Resampling.BILINEAR)

            # Load and resize Alpha
            alpha = Image.open(alpha_p).convert("L")
            alpha = alpha.resize((TARGET_W, TARGET_H), Image.Resampling.BILINEAR)

            rgb_bytes = img.tobytes()
            alpha_bytes = alpha.tobytes()

            serialized = serialize_example(rgb_bytes, alpha_bytes)
            writer.write(serialized)
        except Exception as e:
            print(f"Warning: Failed processing {rgb_p}: {e}")

    if writer:
        writer.close()
    print(f"Finished writing {shard_idx} shards for {output_prefix}.")

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
    files = tf.data.Dataset.list_files(file_pattern, shuffle=is_training)
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    if is_training:
        dataset = dataset.shuffle(2048)
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

if __name__ == "__main__":
    print("Dataset preparation module loaded. Call process_and_write_tfrecords(pairs, prefix) with your dataset paths.")
