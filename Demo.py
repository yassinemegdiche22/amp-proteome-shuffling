import os
import random
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

def parse_fasta(file_path):
    """Lit un fichier FASTA et retourne un dictionnaire {id: sequence}."""
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            sequences[current_id] = "".join(current_seq)
    return sequences

def search_peptides(proteome, peptides):
    """Effectue la recherche exacte des peptides dans le protéome."""
    hits = []
    for pep_id, pep_seq in peptides.items():
        pep_seq_clean = pep_seq.upper()
        for prot_id, prot_seq in proteome.items():
            prot_seq_clean = prot_seq.upper()
            start_idx = 0
            while True:
                idx = prot_seq_clean.find(pep_seq_clean, start_idx)
                if idx == -1:
                    break
                hits.append({
                    'peptide_id': pep_id,
                    'peptide_seq': pep_seq_clean,
                    'protein_id': prot_id,
                    'position_1_based': idx + 1
                })
                start_idx = idx + 1
    return pd.DataFrame(hits)

def run_pipeline():
    os.makedirs("results", exist_ok=True)
    
    print("1. Chargement des données...")
    proteome = parse_fasta("Proteom.fasta")
    peptides = parse_fasta("general_amps.fasta") # Remplace par le nom exact de ton fichier FASTA de peptides
    
    # Restreindre aux 1000 premiers peptides pour le test
    peptides = dict(list(peptides.items())[:1000])
    
    print(f"   - Protéines chargées : {len(proteome)}")
    print(f"   - Peptides testés (sous-ensemble) : {len(peptides)}")
    
    print("\n2. Recherche des hits observés...")
    df_obs = search_peptides(proteome, peptides)
    df_obs.to_csv("results/raw_hits.csv", index=False)
    obs_counts = df_obs['peptide_id'].value_counts().to_dict() if not df_obs.empty else {}
    
    print(f"   - Total de hits bruts trouvés : {len(df_obs)}")
    
    print("\n3. Exécution du modèle nul (shuffling)...")
    num_iterations = 100  # Tu pourras augmenter (ex: 50 ou 100) pour plus de précision par la suite
    null_hits_dict = {pep_id: [] for pep_id in peptides}
    
    for it in range(num_iterations):
        print(f"   - Itération {it + 1}/{num_iterations}...")
        
        # Mélange du protéome
        shuffled_proteome = {}
        for prot_id, seq in proteome.items():
            seq_list = list(seq)
            random.shuffle(seq_list)
            shuffled_proteome[prot_id] = "".join(seq_list)
            
        # Comptage sur le protéome mélangé
        df_null = search_peptides(shuffled_proteome, peptides)
        null_counts = df_null['peptide_id'].value_counts().to_dict() if not df_null.empty else {}
        
        for pep_id in peptides:
            null_hits_dict[pep_id].append(null_counts.get(pep_id, 0))
            
    print("\n4. Calcul des p-valeurs empiriques et du FDR...")
    results = []
    for pep_id, pep_seq in peptides.items():
        obs = obs_counts.get(pep_id, 0)
        null_vals = null_hits_dict[pep_id]
        mean_null = np.mean(null_vals) if null_vals else 0
        
        # P-valeur empirique avec correction de Phipson & Smyth (+1)
        greater_or_equal = sum(1 for x in null_vals if x >= obs)
        empirical_p_val = (greater_or_equal + 1) / (num_iterations + 1)
        
        results.append({
            'peptide_id': pep_id,
            'length': len(pep_seq),
            'observed_hits': obs,
            'expected_chance_hits': mean_null,
            'empirical_p_value': empirical_p_val
        })
        
    df_results = pd.DataFrame(results)
    
    # Correction FDR (Benjamini-Hochberg) stratifiée par longueur
    df_results['adj_p_value_fdr'] = 1.0
    for length, group in df_results.groupby('length'):
        pvals = group['empirical_p_value'].values
        if len(pvals) > 0:
            _, corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
            df_results.loc[group.index, 'adj_p_value_fdr'] = corrected
            
    # Sauvegarde du tableau complet
    df_results.to_csv("results/statistical_summary_fdr.csv", index=False)
    
    # Filtrage des peptides statistiquement significatifs (FDR < 0.05)
    df_significant = df_results[df_results['adj_p_value_fdr'] < 0.05]
    df_significant.to_csv("results/significant_peptides.csv", index=False)
    
    print("\nPipeline terminé avec succès !")
    print(f" - Tableau global : results/statistical_summary_fdr.csv")
    print(f" - Peptides significatifs enregistrés : {len(df_significant)} (dans results/significant_peptides.csv)")

if __name__ == "__main__":
    run_pipeline()