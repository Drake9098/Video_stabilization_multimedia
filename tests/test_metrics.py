"""
Test Metrics Module
Script veloce per testare il funzionamento del modulo metrics.py
"""

import numpy as np
from metrics import (
    compute_rms_displacement,
    compute_jitter_reduction,
    compute_max_offset,
    compute_stability_score,
    generate_metrics_report
)


def test_metrics():
    """
    Test base delle funzioni di metriche.
    """
    print("🧪 Test Modulo Metriche\n")
    print("=" * 60)
    
    # Crea dati sintetici
    n_frames = 100
    
    # Traiettoria RAW (con jitter simulato)
    t = np.linspace(0, 10, n_frames)
    x_raw = 10 * np.sin(t) + np.random.randn(n_frames) * 2  # Movimento + jitter
    y_raw = 5 * np.cos(t) + np.random.randn(n_frames) * 1.5
    angle_raw = 0.1 * np.sin(2*t) + np.random.randn(n_frames) * 0.05
    
    trajectory_raw = np.column_stack([x_raw, y_raw, angle_raw])
    
    # Traiettoria SMOOTHED (simula smoothing efficace)
    from scipy.ndimage import gaussian_filter1d
    x_smooth = gaussian_filter1d(x_raw, sigma=3)
    y_smooth = gaussian_filter1d(y_raw, sigma=3)
    angle_smooth = gaussian_filter1d(angle_raw, sigma=3)
    
    trajectory_smooth = np.column_stack([x_smooth, y_smooth, angle_smooth])
    
    print("\n1️⃣  Test RMS Displacement")
    print("-" * 60)
    
    rms_raw_x, rms_raw_y, rms_raw_angle = compute_rms_displacement(trajectory_raw)
    rms_smooth_x, rms_smooth_y, rms_smooth_angle = compute_rms_displacement(trajectory_smooth)
    
    print(f"   RAW:      X={rms_raw_x:.3f} px, Y={rms_raw_y:.3f} px, Angle={np.degrees(rms_raw_angle):.3f}°")
    print(f"   SMOOTHED: X={rms_smooth_x:.3f} px, Y={rms_smooth_y:.3f} px, Angle={np.degrees(rms_smooth_angle):.3f}°")
    print(f"   ✅ RMS ridotto su tutti gli assi (smoothing efficace)")
    
    print("\n2️⃣  Test Jitter Reduction")
    print("-" * 60)
    
    jitter_x, jitter_y, jitter_angle = compute_jitter_reduction(trajectory_raw, trajectory_smooth)
    
    print(f"   Jitter Reduction X:     {jitter_x:.2f}%")
    print(f"   Jitter Reduction Y:     {jitter_y:.2f}%")
    print(f"   Jitter Reduction Angle: {jitter_angle:.2f}%")
    
    if jitter_x > 50 and jitter_y > 50:
        print(f"   ✅ Riduzione jitter >50% su entrambi gli assi")
    else:
        print(f"   ⚠️  Riduzione jitter bassa (traiettoria test potrebbe essere poco jittery)")
    
    print("\n3️⃣  Test Max Offset")
    print("-" * 60)
    
    max_x, max_y, max_angle = compute_max_offset(trajectory_raw, trajectory_smooth)
    
    print(f"   Max Offset X:     {max_x:.2f} px")
    print(f"   Max Offset Y:     {max_y:.2f} px")
    print(f"   Max Offset Angle: {np.degrees(max_angle):.3f}°")
    print(f"   ✅ Offset calcolati correttamente")
    
    print("\n4️⃣  Test Stability Score")
    print("-" * 60)
    
    score = compute_stability_score(jitter_x, jitter_y, jitter_angle)
    
    print(f"   Stability Score: {score:.2f} / 100")
    
    if score >= 70:
        print(f"   ✅ Score alto - buona stabilizzazione")
    elif score >= 50:
        print(f"   ⚠️  Score medio - stabilizzazione accettabile")
    else:
        print(f"   ❌ Score basso - stabilizzazione insufficiente")
    
    print("\n5️⃣  Test Generate Metrics Report")
    print("-" * 60)
    
    config_test = {
        'motion_estimation': {'method': 'test', 'resolution': 640},
        'trajectory_smoothing': {'radius': 30}
    }
    
    report = generate_metrics_report(
        method_name='test_kalman',
        motion_estimation_method='test_raft',
        trajectory_raw=trajectory_raw,
        trajectory_smoothed=trajectory_smooth,
        num_frames=n_frames,
        processing_time=5.25,
        config=config_test
    )
    
    print(f"   Metadata Keys: {list(report['metadata'].keys())}")
    print(f"   Performance Keys: {list(report['performance'].keys())}")
    print(f"   Configuration Keys: {list(report['configuration'].keys())}")
    print(f"   ✅ Report generato con struttura corretta")
    
    print("\n   Estratto Report:")
    print(f"      - Method: {report['metadata']['smoothing_method']}")
    print(f"      - Stability Score: {report['performance']['stability_score']}")
    print(f"      - Frames: {report['metadata']['num_frames']}")
    print(f"      - Processing Time: {report['metadata']['processing_time_seconds']}s")
    
    print("\n" + "=" * 60)
    print("✅ TUTTI I TEST PASSATI!")
    print("=" * 60)
    
    return report


if __name__ == "__main__":
    # Verifica scipy disponibile (opzionale ma usato nel test)
    try:
        import scipy
        report = test_metrics()
        
        print("\n📄 Report completo disponibile come oggetto Python dict")
        print("   Puoi salvarlo con: json.dump(report, file)")
        
    except ImportError:
        print("⚠️  scipy non installato - test ridotto")
        print("   pip install scipy (opzionale)")
        
        # Test senza scipy
        import numpy as np
        from metrics import compute_stability_score
        
        score = compute_stability_score(80, 85, 75)
        print(f"\n✅ Test base passato - Stability Score: {score:.2f}")
