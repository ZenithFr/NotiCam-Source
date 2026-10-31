import tensorflow as tf

def compute_luma(rgb_image):
    """
    Computes luma matching the exact Vulkan shader definition:
    luma = 0.299*R + 0.587*G + 0.114*B on gamma-encoded RGB in [0, 1].
    rgb_image: [B, H, W, 3]
    returns: [B, H, W, 1]
    """
    weights = tf.constant([0.299, 0.587, 0.114], dtype=rgb_image.dtype)
    weights = tf.reshape(weights, [1, 1, 1, 3])
    return tf.reduce_sum(rgb_image * weights, axis=-1, keepdims=True)

def reconstruct_alpha(coeffs_low, luma_high, target_h=384, target_w=512):
    """
    Reconstructs full-resolution alpha from bilateral slice coefficients:
    coeffs_low: [B, 96, 128, 2] containing [A_low, B_low]
    luma_high:  [B, 384, 512, 1]
    returns:    [B, 384, 512, 1]
    """
    # Upsample coefficients to full resolution using bilinear interpolation
    coeffs_high = tf.image.resize(coeffs_low, [target_h, target_w], method='bilinear')
    a_high = coeffs_high[..., 0:1]
    b_high = coeffs_high[..., 1:2]
    alpha_pred = a_high * luma_high + b_high
    return tf.clip_by_value(alpha_pred, 0.0, 1.0)

def sobel_edges(x):
    """
    Computes horizontal and vertical Sobel gradients:
    x: [B, H, W, 1]
    returns: [B, H, W, 2]
    """
    sobel_x = tf.constant([[-1.0, 0.0, 1.0],
                           [-2.0, 0.0, 2.0],
                           [-1.0, 0.0, 1.0]], dtype=x.dtype)
    sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])

    sobel_y = tf.constant([[-1.0, -2.0, -1.0],
                           [ 0.0,  0.0,  0.0],
                           [ 1.0,  2.0,  1.0]], dtype=x.dtype)
    sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])

    grad_x = tf.nn.conv2d(x, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
    grad_y = tf.nn.conv2d(x, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
    return tf.concat([grad_x, grad_y], axis=-1)

def laplacian_pyramid_loss(y_pred, y_true, num_levels=3):
    """
    Computes Laplacian Pyramid L1 loss across scales to penalize high-frequency edge errors.
    """
    loss = 0.0
    curr_pred = y_pred
    curr_true = y_true

    # Gaussian kernel 5x5
    kernel_1d = tf.constant([1.0, 4.0, 6.0, 4.0, 1.0], dtype=y_pred.dtype) / 16.0
    kernel_2d = tf.tensordot(kernel_1d, kernel_1d, axes=0)
    kernel_4d = tf.reshape(kernel_2d, [5, 5, 1, 1])

    for i in range(num_levels):
        # Blur
        blur_pred = tf.nn.conv2d(curr_pred, kernel_4d, strides=[1, 1, 1, 1], padding='SAME')
        blur_true = tf.nn.conv2d(curr_true, kernel_4d, strides=[1, 1, 1, 1], padding='SAME')

        # Laplacian high-pass band
        lap_pred = curr_pred - blur_pred
        lap_true = curr_true - blur_true

        weight = 2.0 ** (-i)
        loss += weight * tf.reduce_mean(tf.abs(lap_pred - lap_true))

        # Downsample for next octave
        curr_pred = tf.nn.avg_pool2d(blur_pred, ksize=2, strides=2, padding='SAME')
        curr_true = tf.nn.avg_pool2d(blur_true, ksize=2, strides=2, padding='SAME')

    return loss

class PortraitMattingLoss(tf.keras.losses.Loss):
    """
    Composite Loss for AuxCam Guided Filter Portrait Matting:
    loss = L1(alpha_pred, gt) + 0.5 * L1_Sobel(alpha_pred, gt) + 0.5 * LaplacianPyramid(alpha_pred, gt)
    """
    def __init__(self, name="portrait_matting_loss"):
        super().__init__(name=name)

    def call(self, y_true_alpha, alpha_pred):
        """
        y_true_alpha: [B, H, W, 1] ground truth alpha in [0, 1]
        alpha_pred:   [B, H, W, 1] reconstructed alpha in [0, 1]
        """
        # 1. Base L1 loss
        loss_l1 = tf.reduce_mean(tf.abs(alpha_pred - y_true_alpha))

        # 2. Sobel edge gradient L1 loss
        sobel_pred = sobel_edges(alpha_pred)
        sobel_true = sobel_edges(y_true_alpha)
        loss_sobel = tf.reduce_mean(tf.abs(sobel_pred - sobel_true))

        # 3. Laplacian Pyramid multi-scale loss
        loss_lap = laplacian_pyramid_loss(alpha_pred, y_true_alpha, num_levels=3)

        return loss_l1 + 0.5 * loss_sobel + 0.5 * loss_lap

def compute_metrics(y_true_alpha, alpha_pred):
    """
    Evaluates Intersection over Union (IoU) and Sum of Absolute Differences (SAD).
    """
    # Threshold at 0.5 for binary IoU
    pred_bin = tf.cast(alpha_pred >= 0.5, tf.float32)
    true_bin = tf.cast(y_true_alpha >= 0.5, tf.float32)
    intersection = tf.reduce_sum(pred_bin * true_bin, axis=[1, 2, 3])
    union = tf.reduce_sum(tf.clip_by_value(pred_bin + true_bin, 0.0, 1.0), axis=[1, 2, 3])
    iou = tf.reduce_mean((intersection + 1e-6) / (union + 1e-6))

    # SAD in 1e3
    sad = tf.reduce_mean(tf.reduce_sum(tf.abs(alpha_pred - y_true_alpha), axis=[1, 2, 3])) / 1000.0

    return iou, sad
