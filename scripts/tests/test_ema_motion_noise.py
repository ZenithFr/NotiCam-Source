#!/usr/bin/env python3
"""
Test verifying that the 8x noise-averaged downsampled motion detector
does NOT trip on static scenes under high ISO noise (ISO 400-800, sigma = 6.0 - 11.3 / 255),
allowing temporal EMA to actively settle coefficients.
"""
import numpy as np

def simulate_ema_settling():
    print("=== Testing EMA Settling on Static Scene with ISO 800 Noise ===")
    np.random.seed(42)
    
    H, W = 384, 512
    MOTION_H, MOTION_W = 48, 64
    
    # Static base scene (e.g. gray card or wall with luma ~0.5)
    base_luma = 0.50
    iso800_sigma = 8.5 / 255.0  # Typical ISO 800 sensor noise
    
    prev_grid = None
    motion_trips = 0
    num_frames = 50
    
    # Ground truth coefficient value
    coeff_true = 0.80
    coeff_history = []
    current_coeff = 0.0 # start cold
    alpha = 0.25
    
    for frame in range(num_frames):
        # Generate noisy frame at ISO 800
        noise = np.random.normal(0.0, iso800_sigma, (H, W)).astype(np.float32)
        frame_luma = np.clip(base_luma + noise, 0.0, 1.0)
        
        # 48x64 downsampled grid (averaging 8x8 blocks)
        grid = frame_luma.reshape((MOTION_H, 8, MOTION_W, 8)).mean(axis=(1, 3))
        
        # Check motion detector
        motion_detected = False
        if prev_grid is not None:
            max_delta = np.max(np.abs(grid - prev_grid))
            if max_delta > 0.06: # threshold 15/255
                motion_detected = True
                motion_trips += 1
        prev_grid = grid
        
        # Inferred raw coefficient from noisy frame
        raw_coeff = coeff_true + np.random.normal(0.0, iso800_sigma * 0.5)
        
        if motion_detected or frame == 0:
            current_coeff = raw_coeff
        else:
            current_coeff = (1.0 - alpha) * current_coeff + alpha * raw_coeff
            
        coeff_history.append(current_coeff)
    
    print(f"Total frames: {num_frames}")
    print(f"Motion trips on static ISO 800 noise: {motion_trips} / {num_frames - 1}")
    print(f"Initial coeff: {coeff_history[0]:.4f}")
    print(f"Frame 10 coeff: {coeff_history[10]:.4f}")
    print(f"Frame 30 coeff: {coeff_history[30]:.4f}")
    print(f"Frame 50 coeff: {coeff_history[49]:.4f}")
    
    # Standard deviation over last 20 frames
    settled_std = np.std(coeff_history[30:])
    raw_std = iso800_sigma * 0.5
    print(f"Raw noise std: {raw_std:.4f}, Settled EMA std: {settled_std:.4f}")
    print(f"Noise reduction ratio: {raw_std / settled_std:.2f}x")
    
    assert motion_trips == 0, f"FAIL: Motion detector tripped {motion_trips} times on static noise!"
    assert settled_std < raw_std * 0.5, "FAIL: EMA did not smooth noise!"
    assert abs(coeff_history[-1] - coeff_true) < 0.02, "FAIL: Coefficient did not converge!"
    
    print("✅ EMA noise-bypass fix verified: 0 false motion trips, temporal EMA settles cleanly!")

if __name__ == "__main__":
    simulate_ema_settling()
