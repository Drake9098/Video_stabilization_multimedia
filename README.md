# 🎥 Video Stabilization Dashboard

Progetto universitario per il corso di Multimedia. Confronta in modo interattivo algoritmi di **motion estimation** e **trajectory smoothing** applicati alla stabilizzazione video, tramite una dashboard Streamlit.

---

## Panoramica

Il sistema stima il movimento globale tra frame consecutivi, costruisce la traiettoria della camera nel tempo e la liscia per eliminare le vibrazioni ad alta frequenza (jitter). Ogni frame viene poi riproiettato applicando la correzione calcolata, con uno zoom adattivo che elimina i bordi neri risultanti.

Sono implementati e confrontati due approcci distinti per ciascuna fase della pipeline:

**Motion Estimation**

- **SIFT** — estrae keypoint invarianti a scala e rotazione, li matcha con Brute Force + Lowe's ratio test e stima la trasformazione affine via RANSAC. Veloce, non richiede GPU.
- **RAFT** — rete neurale profonda che calcola il flusso ottico denso tra frame. Più preciso in scene complesse, richiede PyTorch.

**Trajectory Smoothing**

- **Moving Average** — convoluzione con finestra mobile uniforme. Semplice ed efficace, ma introduce latenza perché considera frame sia passati che futuri.
- **Kalman Filter** — filtro a 6 stati `[x, y, θ, vx, vy, vθ]` con modello a velocità costante. Predice la posizione futura, riducendo la latenza e ottenendo una stabilizzazione più reattiva.

---

## Struttura del progetto

```
├── app.py                    # Dashboard Streamlit (entry point)
├── video_stabilization.py    # Pipeline principale: lettura, transform, zoom adattivo
├── motion_estimation.py      # SIFT e RAFT
├── trajectory_smoothing.py   # Moving Average e Kalman Filter
├── metrics.py                # Calcolo metriche: RMS, jitter reduction, stability score
├── compare_experiments.py    # Confronto tabellare tra esperimenti salvati in JSON
├── requirements.txt
├── json_files/               # Risultati metriche di esperimenti precedenti
├── stabilized/               # Video stabilizzati in output
├── unstabilized/             # Video originali
└── tests/
```

---

## Installazione

Clona il repository e installa le dipendenze in un ambiente virtuale:

```bash
git clone <url-repo>
cd Video_stabilization_multimedia
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

PyTorch non è strettamente necessario se si usa solo SIFT. Per abilitare RAFT, installa manualmente la versione compatibile con il tuo hardware:

```bash
# CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# GPU (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Avvio

```bash
streamlit run app.py
```

La dashboard si apre nel browser. Da lì è possibile:

- Caricare un video da stabilizzare
- Scegliere il metodo di motion estimation (SIFT o RAFT) e regolarne i parametri
- Scegliere il metodo di smoothing (Moving Average o Kalman) con i relativi iperparametri
- Visualizzare il confronto affiancato tra video originale e stabilizzato
- Consultare le metriche quantitative e i grafici delle traiettorie
- Esportare i risultati in JSON per analisi successive

---

## Metriche

| Metrica                     | Descrizione                                                                            |
| --------------------------- | -------------------------------------------------------------------------------------- |
| **Stability Score**         | Indice composito [0–100] che bilancia riduzione del jitter e fedeltà al moto originale |
| **Jitter Reduction**        | Riduzione percentuale della varianza del moto incrementale rispetto al video raw       |
| **RMS Displacement**        | Ampiezza media degli spostamenti residui (pixel per X/Y, gradi per l'angolo)           |
| **Max Compensation Offset** | Massimo spostamento applicato per compensare le vibrazioni                             |

---

## Confronto esperimenti

Il file `compare_experiments.py` genera tabelle comparative a partire da più file JSON:

```bash
python compare_experiments.py json_files/exp1_sift_baseline.json json_files/exp2_raft_baseline.json
```

I risultati degli esperimenti già condotti sono salvati nella cartella `json_files/`.

---

## Dipendenze principali

- [Streamlit](https://streamlit.io/) — interfaccia web interattiva
- [OpenCV](https://opencv.org/) — elaborazione video, SIFT, trasformazioni affini, Kalman Filter
- [NumPy](https://numpy.org/) / [Pandas](https://pandas.pydata.org/) — calcolo numerico e tabelle
- [Matplotlib](https://matplotlib.org/) — grafici traiettorie
- [PyTorch](https://pytorch.org/) + [TorchVision](https://pytorch.org/vision/) — modello RAFT (opzionale)
