#!/usr/bin/env python3
"""
Unit test asserting physically correct CoC diameter & radius calculations.
Validates absence of unit hazard (focal length mm vs subject distance m).
"""
import sys

def compute_coc_radius(focal_mm, f_stop, sensor_width_mm, image_width_px, s1_focus_m, s2_depth_m, bokeh_gain=1.0):
    if s2_depth_m < s1_focus_m:
        return 0.0  # Near-field suppression: portrait bokeh is background-only
    f_mm = float(focal_mm)
    N = float(f_stop)
    S1_mm = float(s1_focus_m) * 1000.0
    S2_mm = float(s2_depth_m) * 1000.0
    
    aperture_diam_mm = f_mm / N
    mag = f_mm / max(S1_mm - f_mm, 1.0)
    coc_diam_mm = aperture_diam_mm * mag * (abs(S2_mm - S1_mm) / S2_mm)
    coc_diam_px = (coc_diam_mm / float(sensor_width_mm)) * float(image_width_px)
    coc_radius_px = 0.5 * coc_diam_px * float(bokeh_gain)
    return coc_radius_px

def main():
    print("=== Testing CoC Unit Hazard & Physical Ground Truth ===")
    
    # Specification parameters:
    # f = 5.56 mm, N = 1.8, sensor_width = 8.16 mm, width = 1440 px
    # S1 = 0.6 m (focus), S2 = 5.0 m (background), bokeh_gain = 1.0
    r_5m = compute_coc_radius(
        focal_mm=5.56,
        f_stop=1.8,
        sensor_width_mm=8.16,
        image_width_px=1440,
        s1_focus_m=0.6,
        s2_depth_m=5.0,
        bokeh_gain=1.0
    )
    print(f"CoC radius at S2=5.0m, gain=1.0: {r_5m:.4f} px (expected ~2.24 px)")
    assert 2.23 <= r_5m <= 2.25, f"FAIL: Expected ~2.24 px, got {r_5m:.4f} px"

    # S2 -> infinity (1e6 m)
    r_inf = compute_coc_radius(
        focal_mm=5.56,
        f_stop=1.8,
        sensor_width_mm=8.16,
        image_width_px=1440,
        s1_focus_m=0.6,
        s2_depth_m=1e6,
        bokeh_gain=1.0
    )
    print(f"CoC radius at S2=inf, gain=1.0: {r_inf:.4f} px (expected ~2.55 px)")
    assert 2.54 <= r_inf <= 2.56, f"FAIL: Expected ~2.55 px, got {r_inf:.4f} px"

    # With default bokeh_gain = 5.0
    r_gain5 = compute_coc_radius(
        focal_mm=5.56,
        f_stop=1.8,
        sensor_width_mm=8.16,
        image_width_px=1440,
        s1_focus_m=0.6,
        s2_depth_m=5.0,
        bokeh_gain=5.0
    )
    print(f"CoC radius at S2=5.0m, gain=5.0: {r_gain5:.4f} px (expected ~11.22 px)")
    assert 11.20 <= r_gain5 <= 11.24, f"FAIL: Expected ~11.22 px, got {r_gain5:.4f} px"

    # Near-field suppression test (S2 = 0.3m, S1 = 0.6m)
    r_fore = compute_coc_radius(
        focal_mm=5.56,
        f_stop=1.8,
        sensor_width_mm=8.16,
        image_width_px=1440,
        s1_focus_m=0.6,
        s2_depth_m=0.3,
        bokeh_gain=5.0
    )
    print(f"CoC radius at S2=0.3m (foreground): {r_fore:.4f} px (expected 0.0 px - suppressed)")
    assert r_fore == 0.0, f"FAIL: Expected 0.0 px, got {r_fore:.4f} px"

    print("✅ CoC unit test passed with zero unit hazards and verified near-field suppression!")

if __name__ == "__main__":
    main()
