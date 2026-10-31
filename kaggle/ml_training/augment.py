import tensorflow as tf
import math

class PortraitAugmenter:
    """
    Augmentation module for AuxCam Portrait Net.
    Applies spatial transforms to both RGB image and alpha ground truth,
    and applies ISP simulation transforms strictly to RGB image.
    """
    def __init__(self, target_h=384, target_w=512):
        self.target_h = target_h
        self.target_w = target_w

        # Precompute radial distance grid for vignette [H, W, 1]
        y = tf.linspace(-1.0, 1.0, target_h)
        x = tf.linspace(-1.0, 1.0, target_w)
        yy, xx = tf.meshgrid(y, x, indexing='ij')
        # Normalized radius: r = sqrt(xx^2 + yy^2), r_max = sqrt(2.0)
        r_sq = (xx**2 + yy**2) / 2.0
        self.vignette_mask = tf.expand_dims(1.0 - 0.10 * r_sq, axis=-1) # [H, W, 1]

    @tf.function
    def apply_spatial(self, image, alpha):
        """
        image: [H, W, 3] float32 in [0, 1]
        alpha: [H, W, 1] float32 in [0, 1]
        """
        combined = tf.concat([image, alpha], axis=-1) # [H, W, 4]

        # 1. Random horizontal flip (50%)
        combined = tf.image.random_flip_left_right(combined)

        # 2. Random scale & crop (scale 0.9 - 1.1)
        scale = tf.random.uniform([], 0.9, 1.1)
        scaled_h = tf.cast(tf.cast(self.target_h, tf.float32) * scale, tf.int32)
        scaled_w = tf.cast(tf.cast(self.target_w, tf.float32) * scale, tf.int32)
        combined_scaled = tf.image.resize(combined, [scaled_h, scaled_w], method='bilinear')

        # Crop or pad to target [384, 512]
        combined = tf.image.resize_with_crop_or_pad(combined_scaled, self.target_h, self.target_w)

        # 3. Small rotation (+/- 5 degrees)
        angle = tf.random.uniform([], -5.0 * (math.pi / 180.0), 5.0 * (math.pi / 180.0))
        # Use simple shear/affine or rotation if available; tf.image.rot90 / affine
        # In TF native, rotate can be done via tf.raw_ops.ImageProjectiveTransformV3 or small roll
        # For small +/-5 deg, tf.image affine is clean:
        cos_a = tf.cos(angle)
        sin_a = tf.sin(angle)
        # Center of rotation
        cx = tf.cast(self.target_w, tf.float32) / 2.0
        cy = tf.cast(self.target_h, tf.float32) / 2.0
        # Transform matrix mapping output coords to input coords:
        # x_in = cos(a)*(x - cx) - sin(a)*(y - cy) + cx
        # y_in = sin(a)*(x - cx) + cos(a)*(y - cy) + cy
        a0 = cos_a
        a1 = -sin_a
        a2 = cx - cos_a * cx + sin_a * cy
        b0 = sin_a
        b1 = cos_a
        b2 = cy - sin_a * cx - cos_a * cy
        transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])
        combined_4d = tf.expand_dims(combined, axis=0)
        combined_rot = tf.raw_ops.ImageProjectiveTransformV3(
            images=combined_4d,
            transforms=tf.expand_dims(transform, axis=0),
            output_shape=[self.target_h, self.target_w],
            interpolation="BILINEAR",
            fill_mode="REFLECT",
            fill_value=0.0
        )
        combined = tf.squeeze(combined_rot, axis=0)

        image_out = combined[..., :3]
        alpha_out = combined[..., 3:]
        return image_out, alpha_out

    @tf.function
    def apply_isp_simulation(self, image):
        """
        Applies camera ISP simulation:
        - Gamma jitter [0.95, 1.10]
        - Per-channel gain jitter [0.97, 1.03]
        - Gaussian noise sigma in [0.005, 0.02]
        - JPEG quality [85, 98] (50% prob)
        - Radial vignette
        """
        # 1. Gamma jitter
        gamma = tf.random.uniform([], 0.95, 1.10)
        image = tf.math.pow(tf.maximum(image, 1e-6), gamma)

        # 2. Per-channel gain jitter
        gain_r = tf.random.uniform([], 0.97, 1.03)
        gain_g = tf.random.uniform([], 0.97, 1.03)
        gain_b = tf.random.uniform([], 0.97, 1.03)
        gains = tf.stack([gain_r, gain_g, gain_b])
        image = image * gains

        # 3. Gaussian sensor noise
        sigma = tf.random.uniform([], 0.005, 0.02)
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=sigma)
        image = image + noise

        # 4. Radial vignette
        image = image * self.vignette_mask

        # 5. JPEG compression jitter (50% prob)
        do_jpeg = tf.random.uniform([]) > 0.5
        quality = tf.random.uniform([], 85, 98, dtype=tf.int32)
        image = tf.cond(do_jpeg, lambda: tf.image.adjust_jpeg_quality(image, quality), lambda: image)

        return tf.clip_by_value(image, 0.0, 1.0)

    @tf.function
    def augment_train(self, image, alpha):
        # image: [H, W, 3], alpha: [H, W, 1]
        image, alpha = self.apply_spatial(image, alpha)
        image = self.apply_isp_simulation(image)
        alpha = tf.clip_by_value(alpha, 0.0, 1.0)
        return image, alpha
