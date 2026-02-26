"""
Test Script per Video Stabilization
Script di test per verificare il funzionamento dei moduli principali.
Può essere usato per debug senza lanciare l'intera app Streamlit.
"""

import numpy as np
import cv2
from motion_estimation import (
    estimate_motion_sift,
    estimate_motion_raft
)
from trajectory_smoothing import (
    moving_average_smooth,
    kalman_smooth,
    compute_trajectory_from_transforms,
    KalmanTrajectoryFilter
)
from video_stabilization import (
    resize_frame,
    apply_transform_with_zoom,
    get_video_info
)


def test_motion_estimation():
    """Test della stima del movimento con SIFT."""
    print("=" * 50)
    print("TEST: Motion Estimation (SIFT)")
    print("=" * 50)
    
    # Crea due frame di test con pattern visibili per SIFT
    img1 = np.zeros((480, 640), dtype=np.uint8)
    img2 = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(img1, (200, 200), 50, 255, -1)
    cv2.circle(img2, (210, 205), 50, 255, -1)  # Piccolo spostamento
    
    # Stima movimento con SIFT
    dx, dy, da, new_kp, new_desc = estimate_motion_sift(img1, img2)
    print(f"✓ Movimento SIFT stimato: dx={dx:.2f}, dy={dy:.2f}, da={np.degrees(da):.2f}°")
    
    print("✅ Test Motion Estimation completato!\n")


def test_trajectory_smoothing():
    """Test dello smoothing delle traiettorie."""
    print("=" * 50)
    print("TEST: Trajectory Smoothing")
    print("=" * 50)
    
    # Crea una traiettoria di test (sinusoide con rumore)
    n_frames = 200
    t = np.linspace(0, 4 * np.pi, n_frames)
    
    # Traiettoria smooth di base + rumore (jitter)
    x = 100 * np.sin(t) + np.random.normal(0, 10, n_frames)
    y = 100 * np.cos(t) + np.random.normal(0, 10, n_frames)
    angle = 0.1 * np.sin(2 * t) + np.random.normal(0, 0.05, n_frames)
    
    trajectory = np.column_stack([x, y, angle])
    print(f"✓ Traiettoria creata: {trajectory.shape}")
    
    # Test Media Mobile
    smoothed_ma = moving_average_smooth(trajectory, radius=30)
    print(f"✓ Media Mobile applicata: {smoothed_ma.shape}")
    
    # Calcola riduzione del rumore
    noise_original = np.std(np.diff(trajectory[:, 0]))
    noise_ma = np.std(np.diff(smoothed_ma[:, 0]))
    print(f"  - Rumore originale: {noise_original:.2f}")
    print(f"  - Rumore MA: {noise_ma:.2f}")
    print(f"  - Riduzione: {(1 - noise_ma/noise_original)*100:.1f}%")
    
    # Test Kalman Filter
    smoothed_kalman = kalman_smooth(trajectory, process_noise=0.01, measurement_noise=1.0)
    print(f"✓ Kalman Filter applicato: {smoothed_kalman.shape}")
    
    noise_kalman = np.std(np.diff(smoothed_kalman[:, 0]))
    print(f"  - Rumore Kalman: {noise_kalman:.2f}")
    print(f"  - Riduzione: {(1 - noise_kalman/noise_original)*100:.1f}%")
    
    print("✅ Test Trajectory Smoothing completato!\n")


def test_kalman_filter():
    """Test specifico del Filtro di Kalman."""
    print("=" * 50)
    print("TEST: Kalman Filter Dettagliato")
    print("=" * 50)
    
    # Crea un filtro
    kf = KalmanTrajectoryFilter(process_noise=0.01, measurement_noise=1.0)
    print("✓ Filtro Kalman inizializzato")
    
    # Simula una sequenza di misure con jitter
    measurements = []
    true_position = 0.0
    for i in range(50):
        true_position += 5.0  # Movimento costante
        measured = true_position + np.random.normal(0, 10)  # Aggiungi jitter
        measurements.append([measured, 0.0, 0.0])
    
    # Applica il filtro
    filtered = []
    for meas in measurements:
        filt = kf.update(meas)
        filtered.append(filt[0])
    
    # Confronta
    meas_noise = np.std(np.diff([m[0] for m in measurements]))
    filt_noise = np.std(np.diff(filtered))
    print(f"✓ Rumore misure: {meas_noise:.2f}")
    print(f"✓ Rumore filtrato: {filt_noise:.2f}")
    print(f"✓ Miglioramento: {(1 - filt_noise/meas_noise)*100:.1f}%")
    
    print("✅ Test Kalman Filter completato!\n")


def test_frame_processing():
    """Test del processing dei frame."""
    print("=" * 50)
    print("TEST: Frame Processing")
    print("=" * 50)
    
    # Crea un frame di test
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    print(f"✓ Frame originale: {frame.shape}")
    
    # Test ridimensionamento
    resized = resize_frame(frame, width=640)
    print(f"✓ Frame ridimensionato: {resized.shape}")
    
    # Test trasformazione con zoom
    stabilized = apply_transform_with_zoom(resized, dx=10, dy=-5, da=0.05, zoom_factor=1.1)
    print(f"✓ Frame stabilizzato: {stabilized.shape}")
    
    print("✅ Test Frame Processing completato!\n")


def test_trajectory_computation():
    """Test del calcolo della traiettoria."""
    print("=" * 50)
    print("TEST: Trajectory Computation")
    print("=" * 50)
    
    # Crea trasformazioni di test
    transforms = [
        (1.0, 0.5, 0.01),
        (1.2, 0.3, 0.02),
        (0.8, 0.7, -0.01),
        (1.1, 0.4, 0.015)
    ]
    
    # Calcola traiettoria
    trajectory = compute_trajectory_from_transforms(transforms)
    print(f"✓ Traiettoria calcolata: {trajectory.shape}")
    print(f"  - Prima posizione: {trajectory[0]}")
    print(f"  - Ultima posizione: {trajectory[-1]}")
    print(f"  - Traiettoria cumulativa verificata: {np.allclose(trajectory[0], transforms[0])}")
    
    print("✅ Test Trajectory Computation completato!\n")


def run_all_tests():
    """Esegue tutti i test."""
    print("\n" + "=" * 50)
    print("INIZIO TEST SUITE")
    print("=" * 50 + "\n")
    
    try:
        test_motion_estimation()
    except Exception as e:
        print(f"❌ Test Motion Estimation fallito: {e}\n")
    
    try:
        test_trajectory_smoothing()
    except Exception as e:
        print(f"❌ Test Trajectory Smoothing fallito: {e}\n")
    
    try:
        test_kalman_filter()
    except Exception as e:
        print(f"❌ Test Kalman Filter fallito: {e}\n")
    
    try:
        test_frame_processing()
    except Exception as e:
        print(f"❌ Test Frame Processing fallito: {e}\n")
    
    try:
        test_trajectory_computation()
    except Exception as e:
        print(f"❌ Test Trajectory Computation fallito: {e}\n")
    
    print("=" * 50)
    print("TEST SUITE COMPLETATA")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()
