import anndata
import scipy
import numpy as np
import pandas as pd
import logging
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
from sketchKH import *
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, silhouette_score
import seaborn as sns
from scipy.stats import spearmanr

def test_preservation(adata, features_set1, features_set2, n_clusters=4):
    """
    Test both cell state and trajectory preservation between two feature sets
    
    Parameters:
    -----------
    adata : AnnData object
        Input data
    features_set1 : list
        First set of features (non-attention)
    features_set2 : list
        Second set of features (attention)
    n_clusters : int
        Number of expected cell states/clusters
    """
    results = {}
    
    # 1. Cell State Preservation Tests
    results['cell_state'] = test_cell_state_preservation(
        adata, features_set1, features_set2, n_clusters
    )
    
    # 2. Trajectory Preservation Tests
    results['trajectory'] = test_trajectory_preservation(
        adata, features_set1, features_set2
    )
    
    return results

def test_cell_state_preservation(adata, features_set1, features_set2, n_clusters):
    """Test cell state preservation through clustering comparison"""
    
    # Get data for both feature sets
    X1 = adata[:, features_set1].X
    X2 = adata[:, features_set2].X
    
    # Store the feature matrices in adata.obsm for visualization
    adata.obsm['X_set1'] = X1.toarray() if scipy.sparse.issparse(X1) else X1
    adata.obsm['X_set2'] = X2.toarray() if scipy.sparse.issparse(X2) else X2
    
    # Perform clustering on both feature sets using their respective features
    kmeans1 = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans2 = KMeans(n_clusters=n_clusters, random_state=42)
    
    # Use the specific feature sets for clustering
    labels1 = kmeans1.fit_predict(adata.obsm['X_set1'])
    labels2 = kmeans2.fit_predict(adata.obsm['X_set2'])
    
    # Calculate metrics using the specific feature sets
    ari_score = adjusted_rand_score(labels1, labels2)
    sil_score1 = silhouette_score(adata.obsm['X_set1'], labels1)
    sil_score2 = silhouette_score(adata.obsm['X_set2'], labels2)
    
    # Visualize clustering comparison
    fig = plt.figure(figsize=(20, 5))
    
    # Store cluster labels in adata.obs
    adata.obs['set1_clusters'] = labels1
    adata.obs['set2_clusters'] = labels2
    
    # Plot UMAP for first feature set using only those features
    plt.subplot(141)
    sc.pp.neighbors(adata, use_rep='X_set1', n_neighbors=15)
    sc.tl.umap(adata)
    adata.obsm['X_umap_set1'] = adata.obsm['X_umap'].copy()
    sc.pl.umap(adata, color='set1_clusters', 
               title=f'Non-attention clusters\nTop 10 features:\n{", ".join(features_set1[:10])}', 
               show=False, ax=plt.gca())
    
    # Plot feature ranking for set 1
    plt.subplot(142)
    top_features1 = features_set1[:10]
    plt.barh(range(10), np.arange(10, 0, -1))
    plt.yticks(range(10), top_features1)
    plt.title('Top 10 Non-attention Features\n(by ranking)')
    plt.xlabel('Rank (lower is better)')
    
    # Plot UMAP for second feature set using only those features
    plt.subplot(143)
    sc.pp.neighbors(adata, use_rep='X_set2', n_neighbors=15)
    sc.tl.umap(adata)
    adata.obsm['X_umap_set2'] = adata.obsm['X_umap'].copy()
    sc.pl.umap(adata, color='set2_clusters', 
               title=f'Attention clusters\nTop 10 features:\n{", ".join(features_set2[:10])}', 
               show=False, ax=plt.gca())
    
    # Plot feature ranking for set 2
    plt.subplot(144)
    top_features2 = features_set2[:10]
    plt.barh(range(10), np.arange(10, 0, -1))
    plt.yticks(range(10), top_features2)
    plt.title('Top 10 Attention Features\n(by ranking)')
    plt.xlabel('Rank (lower is better)')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'ARI': ari_score,
        'silhouette_score1': sil_score1,
        'silhouette_score2': sil_score2,
        'top_features1': top_features1,
        'top_features2': top_features2
    }

def test_trajectory_preservation(adata, features_set1, features_set2):
    """Test trajectory preservation through pseudotime correlation"""
    
    # Get data for both feature sets and convert to dense arrays if sparse
    X1 = adata[:, features_set1].X
    X1 = X1.toarray() if scipy.sparse.issparse(X1) else X1
    X2 = adata[:, features_set2].X
    X2 = X2.toarray() if scipy.sparse.issparse(X2) else X2
    
    # Store the matrices in adata.obsm
    adata.obsm['X_set1'] = X1
    adata.obsm['X_set2'] = X2
    
    # Calculate pseudotime for first feature set
    sc.pp.neighbors(adata, use_rep='X_set1', n_neighbors=15)
    sc.tl.diffmap(adata)
    adata.uns['iroot'] = np.argmin(adata.obsm['X_diffmap'][:, 0])
    sc.tl.dpt(adata)
    pt1 = adata.obs['dpt_pseudotime'].values
    
    # Calculate pseudotime for second feature set
    sc.pp.neighbors(adata, use_rep='X_set2', n_neighbors=15)
    sc.tl.diffmap(adata)
    adata.uns['iroot'] = np.argmin(adata.obsm['X_diffmap'][:, 0])
    sc.tl.dpt(adata)
    pt2 = adata.obs['dpt_pseudotime'].values
    
    # Calculate correlation
    correlation, pvalue = spearmanr(pt1, pt2)
    
    # Visualize pseudotime correlation
    plt.figure(figsize=(10, 5))
    
    # Scatter plot
    plt.subplot(121)
    plt.scatter(pt1, pt2, alpha=0.5)
    plt.xlabel('Non-attention pseudotime')
    plt.ylabel('Attention pseudotime')
    plt.title(f'Pseudotime Correlation\nr = {correlation:.3f}')
    
    # Distribution plot
    plt.subplot(122)
    plt.hist(pt1, alpha=0.5, label='Non-attention', bins=50, density=True)
    plt.hist(pt2, alpha=0.5, label='Attention', bins=50, density=True)
    plt.xlabel('Pseudotime')
    plt.ylabel('Density')
    plt.title('Pseudotime Distributions')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'correlation': correlation,
        'pvalue': pvalue
    }

def main():
    # Load your data
    adata = sc.read_h5ad('/Users/georgeweale/delve/data/adata_RPE.h5ad')
    
    # Read feature sets from CSV
    features_df = pd.read_csv('cyril/combined_features.csv')
    features_knn = features_df['features_knn'].dropna().tolist()
    features_knn_weighted = features_df['features_knn_weighted'].dropna().tolist()
    
    # Run tests
    results = test_preservation(
        adata,
        features_knn,
        features_knn_weighted,
        n_clusters=4  # Adjust based on your expected number of cell states
    )
    
    # Print results
    print("\nCell State Preservation Results:")
    print(f"ARI Score: {results['cell_state']['ARI']:.3f}")
    print(f"Non-attention Silhouette Score: {results['cell_state']['silhouette_score1']:.3f}")
    print(f"Attention Silhouette Score: {results['cell_state']['silhouette_score2']:.3f}")
    
    print("\nTrajectory Preservation Results:")
    print(f"Pseudotime Correlation: {results['trajectory']['correlation']:.3f}")
    print(f"P-value: {results['trajectory']['pvalue']:.3e}")

if __name__ == '__main__':
    main() 