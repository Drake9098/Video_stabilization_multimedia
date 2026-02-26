"""
Generate Test Video
Script per generare un video di test con shake simulato.
Utile per testare la stabilizzazione senza avere un video reale.
"""

import cv2
import numpy as np
import os


def generate_test_video(output_path='test_video.mp4', 
                       duration=10, 
                       fps=30, 
                       width=640, 
                       height=480,
                       shake_intensity=10.0):
    """
    Genera un video di test con shape geometriche in movimento e shake simulato.
    
    Args:
        output_path: Percorso del video di output
        duration: Durata in secondi
        fps: Frame per secondo
        width: Larghezza del video
        height: Altezza del video
        shake_intensity: Intensità dello shake (0-20, più alto = più shake)
    """
    
    n_frames = duration * fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Generazione video di test...")
    print(f"- Durata: {duration}s ({n_frames} frame)")
    print(f"- Risoluzione: {width}x{height}")
    print(f"- Intensità shake: {shake_intensity}")
    
    # Parametri per le forme
    circle_pos = [width // 4, height // 2]
    square_pos = [3 * width // 4, height // 2]
    
    for i in range(n_frames):
        # Crea frame base
        frame = np.ones((height, width, 3), dtype=np.uint8) * 30  # Grigio scuro
        
        # Aggiungi griglia di riferimento
        for x in range(0, width, 50):
            cv2.line(frame, (x, 0), (x, height), (60, 60, 60), 1)
        for y in range(0, height, 50):
            cv2.line(frame, (0, y), (width, y), (60, 60, 60), 1)
        
        # Simula movimento degli oggetti (panning smooth)
        t = i / fps
        circle_pos[0] = int(width // 4 + 50 * np.sin(t))
        circle_pos[1] = int(height // 2 + 30 * np.cos(2 * t))
        
        square_pos[0] = int(3 * width // 4 - 40 * np.sin(t * 1.5))
        square_pos[1] = int(height // 2 - 25 * np.cos(t * 0.8))
        
        # Disegna cerchio
        cv2.circle(frame, tuple(circle_pos), 40, (0, 255, 0), -1)
        cv2.circle(frame, tuple(circle_pos), 40, (0, 200, 0), 2)
        
        # Disegna quadrato
        cv2.rectangle(
            frame,
            (square_pos[0] - 40, square_pos[1] - 40),
            (square_pos[0] + 40, square_pos[1] + 40),
            (255, 0, 0),
            -1
        )
        cv2.rectangle(
            frame,
            (square_pos[0] - 40, square_pos[1] - 40),
            (square_pos[0] + 40, square_pos[1] + 40),
            (200, 0, 0),
            2
        )
        
        # Aggiungi testo
        cv2.putText(
            frame,
            f"Frame {i + 1}/{n_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # SIMULA SHAKE (jitter ad alta frequenza + rotazione)
        # Questo è il "problema" che la stabilizzazione deve risolvere
        
        # Shake traslazionale (tremore della mano)
        shake_x = np.random.normal(0, shake_intensity)
        shake_y = np.random.normal(0, shake_intensity)
        
        # Shake rotazionale (piccole rotazioni casuali)
        shake_angle = np.random.normal(0, shake_intensity / 10.0)  # In gradi
        
        # Aggiungi anche un trend lento (drift)
        drift_x = shake_intensity * 0.5 * np.sin(t * 2)
        drift_y = shake_intensity * 0.5 * np.cos(t * 3)
        
        # Applica la trasformazione di shake al frame
        center = (width / 2, height / 2)
        
        # Matrice di rotazione + traslazione
        M = cv2.getRotationMatrix2D(center, shake_angle, 1.0)
        M[0, 2] += shake_x + drift_x
        M[1, 2] += shake_y + drift_y
        
        # Applica transformazione
        shaky_frame = cv2.warpAffine(
            frame,
            M,
            (width, height),
            borderMode=cv2.BORDER_REFLECT
        )
        
        # Scrivi frame
        out.write(shaky_frame)
        
        # Progress bar
        if (i + 1) % 30 == 0:
            progress = (i + 1) / n_frames * 100
            print(f"Progresso: {progress:.1f}% ({i + 1}/{n_frames} frame)")
    
    out.release()
    print(f"\n✅ Video generato: {output_path}")
    print(f"Dimensione file: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def generate_multiple_test_videos():
    """Genera vari video di test con diversi livelli di shake."""
    
    print("=" * 60)
    print("GENERAZIONE VIDEO DI TEST")
    print("=" * 60 + "\n")
    
    # Video con shake leggero
    print("\n1. Video con shake LEGGERO")
    print("-" * 60)
    generate_test_video(
        output_path='test_video_light.mp4',
        duration=5,
        shake_intensity=5.0
    )
    
    # Video con shake medio
    print("\n2. Video con shake MEDIO")
    print("-" * 60)
    generate_test_video(
        output_path='test_video_medium.mp4',
        duration=5,
        shake_intensity=10.0
    )
    
    # Video con shake forte
    print("\n3. Video con shake FORTE")
    print("-" * 60)
    generate_test_video(
        output_path='test_video_heavy.mp4',
        duration=5,
        shake_intensity=15.0
    )
    
    print("\n" + "=" * 60)
    print("GENERAZIONE COMPLETATA!")
    print("=" * 60)
    print("\nPuoi ora usare questi video per testare l'app Streamlit:")
    print("  streamlit run app.py")
    print("\nE caricare uno dei video generati.")


if __name__ == "__main__":
    # Genera un singolo video di test con shake medio
    generate_test_video(
        output_path='test_video.mp4',
        duration=10,
        fps=30,
        shake_intensity=10.0
    )
    
    # Per generare tutti e tre i video, decommenta la riga seguente:
    # generate_multiple_test_videos()
