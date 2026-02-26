"""
Compare Experiments Script
Script per confrontare automaticamente metriche di esperimenti multipli.
Utile per generare tabelle comparative per la relazione.
"""

import json
import argparse
from pathlib import Path
import pandas as pd


def load_metrics(json_path):
    """
    Carica metriche da file JSON.
    
    Args:
        json_path: Path del file JSON
        
    Returns:
        Dict con metriche
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_key_metrics(metrics):
    """
    Estrae le metriche chiave per il confronto.
    
    Args:
        metrics: Dict completo delle metriche
        
    Returns:
        Dict con metriche essenziali
    """
    meta = metrics['metadata']
    perf = metrics['performance']
    
    return {
        'Method': f"{meta['smoothing_method']} + {meta['motion_estimation_method']}",
        'Stability Score': perf['stability_score'],
        'Jitter Reduction X (%)': perf['jitter_reduction_percent']['x'],
        'Jitter Reduction Y (%)': perf['jitter_reduction_percent']['y'],
        'Jitter Reduction Angle (%)': perf['jitter_reduction_percent']['angle'],
        'RMS X (px)': perf['rms_displacement']['smoothed']['x_px'],
        'RMS Y (px)': perf['rms_displacement']['smoothed']['y_px'],
        'RMS Angle (°)': perf['rms_displacement']['smoothed']['angle_deg'],
        'Max Offset X (px)': perf['max_compensation_offset']['x_px'],
        'Max Offset Y (px)': perf['max_compensation_offset']['y_px'],
        'Processing Time (s)': meta.get('processing_time_seconds', 'N/A'),
        'Frames': meta['num_frames']
    }


def compare_experiments(json_paths, output_format='table'):
    """
    Confronta metriche di esperimenti multipli.
    
    Args:
        json_paths: Lista di path ai file JSON
        output_format: 'table', 'csv', o 'latex'
        
    Returns:
        DataFrame con confronto
    """
    experiments = []
    
    for json_path in json_paths:
        try:
            metrics = load_metrics(json_path)
            key_metrics = extract_key_metrics(metrics)
            experiments.append(key_metrics)
        except Exception as e:
            print(f"⚠️  Errore loading {json_path}: {e}")
    
    if not experiments:
        print("❌ Nessun esperimento caricato con successo")
        return None
    
    df = pd.DataFrame(experiments)
    
    # Ordina per Stability Score (descending)
    df = df.sort_values('Stability Score', ascending=False)
    
    return df


def print_comparison_table(df):
    """
    Stampa tabella formattata per terminale.
    
    Args:
        df: DataFrame con metriche
    """
    print("\n" + "=" * 120)
    print("🔬 CONFRONTO ESPERIMENTI - METRICHE QUANTITATIVE")
    print("=" * 120)
    print(df.to_string(index=False))
    print("=" * 120)
    
    # Identifica migliore
    best_idx = df['Stability Score'].idxmax()
    best_method = df.loc[best_idx, 'Method']
    best_score = df.loc[best_idx, 'Stability Score']
    
    print(f"\n🏆 MIGLIORE: {best_method} con Stability Score = {best_score:.2f}")
    print()


def save_as_csv(df, output_path):
    """
    Salva confronto come CSV.
    
    Args:
        df: DataFrame con metriche
        output_path: Path output CSV
    """
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ CSV salvato in: {output_path}")


def save_as_latex(df, output_path):
    """
    Salva confronto come tabella LaTeX.
    
    Args:
        df: DataFrame con metriche
        output_path: Path output .tex
    """
    latex_str = df.to_latex(index=False, float_format="%.2f", escape=False)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_str)
    
    print(f"✅ LaTeX salvato in: {output_path}")


def main():
    """
    Script principale per confronto esperimenti da linea di comando.
    """
    parser = argparse.ArgumentParser(
        description='Confronta metriche di esperimenti multipli',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Confronta 2 esperimenti
  python compare_experiments.py metrics_kalman.json metrics_ma.json
  
  # Esporta come CSV
  python compare_experiments.py *.json --csv comparison.csv
  
  # Esporta come LaTeX
  python compare_experiments.py *.json --latex comparison.tex
        """
    )
    
    parser.add_argument(
        'json_files',
        nargs='+',
        help='File JSON con metriche da confrontare'
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        help='Salva confronto come CSV'
    )
    
    parser.add_argument(
        '--latex',
        type=str,
        help='Salva confronto come tabella LaTeX'
    )
    
    parser.add_argument(
        '--sort-by',
        type=str,
        default='Stability Score',
        help='Colonna per ordinamento (default: Stability Score)'
    )
    
    args = parser.parse_args()
    
    print("\n🔍 Caricamento esperimenti...")
    print(f"File trovati: {len(args.json_files)}")
    
    # Confronta
    df = compare_experiments(args.json_files)
    
    if df is None:
        return
    
    # Ri-ordina se richiesto
    if args.sort_by in df.columns:
        df = df.sort_values(args.sort_by, ascending=False)
    
    # Stampa tabella
    print_comparison_table(df)
    
    # Export se richiesto
    if args.csv:
        save_as_csv(df, args.csv)
    
    if args.latex:
        save_as_latex(df, args.latex)
    
    # Riepilogo
    print("\n📊 RIEPILOGO:")
    print(f"   - Esperimenti confrontati: {len(df)}")
    print(f"   - Metodi testati: {df['Method'].nunique()}")
    print(f"   - Score range: {df['Stability Score'].min():.1f} - {df['Stability Score'].max():.1f}")
    
    # Statistiche aggiuntive
    print("\n📈 STATISTICHE:")
    print(f"   - Avg Stability Score: {df['Stability Score'].mean():.2f}")
    print(f"   - Avg Jitter Reduction X: {df['Jitter Reduction X (%)'].mean():.1f}%")
    print(f"   - Avg Jitter Reduction Y: {df['Jitter Reduction Y (%)'].mean():.1f}%")
    print()


if __name__ == "__main__":
    main()
