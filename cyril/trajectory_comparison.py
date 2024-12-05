import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from scipy.stats import kendalltau
import anndata

def compare_trajectories(adata, feature_rankings_path, n_top_features=50):
    """
    Compare trajectories generated using different feature sets
    
    Parameters
    ----------
    adata: AnnData object
        Contains expression data
    feature_rankings_path: str
        Path to CSV containing feature rankings
    n_top_features: int
        Number of top features to use
    """
    # Load feature rankings
    rankings = pd.read_csv(feature_rankings_path)
    
    # Get top features for each method
    knn_features = rankings['features_knn'][:n_top_features].tolist()
    weighted_features = rankings['features_knn_weighted'][:n_top_features].tolist()
    
    # Create separate AnnData objects with selected features
    adata_knn = adata[:, knn_features].copy()
    adata_weighted = adata[:, weighted_features].copy()
    
    # Process both datasets
    trajectories = {}
    for name, data in [('KNN', adata_knn), ('Weighted KNN', adata_weighted)]:
        # Normalize and process
        sc.pp.normalize_total(data)
        sc.pp.log1p(data)
        
        # Compute PCA
        sc.pp.pca(data)
        
        # Compute neighborhood graph
        sc.pp.neighbors(data)
        
        # Compute UMAP embedding
        sc.tl.umap(data)
        
        # Compute diffusion pseudotime
        sc.tl.diffmap(data)
        data.uns['iroot'] = 0
        sc.tl.dpt(data)
        
        trajectories[name] = data
        
    return trajectories

def visualize_trajectory_comparison(trajectories, feature_rankings_path, n_top_features):
    """
    Visualize and compare trajectories
    
    Parameters
    ----------
    trajectories: dict
        Dictionary containing processed AnnData objects
    feature_rankings_path: str
        Path to CSV containing feature rankings
    n_top_features: int
        Number of top features to use
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Plot UMAP embeddings
    for idx, (name, adata) in enumerate(trajectories.items()):
        sc.pl.umap(adata, color='dpt_pseudotime', ax=axes[0, idx], 
                  title=f'{name} Trajectory', show=False)
    
    # Plot pseudotime correlation
    pseudotime_knn = trajectories['KNN'].obs['dpt_pseudotime']
    pseudotime_weighted = trajectories['Weighted KNN'].obs['dpt_pseudotime']
    
    correlation = np.corrcoef(pseudotime_knn, pseudotime_weighted)[0,1]
    
    axes[1, 0].scatter(pseudotime_knn, pseudotime_weighted, alpha=0.5)
    axes[1, 0].set_xlabel('KNN Pseudotime')
    axes[1, 0].set_ylabel('Weighted KNN Pseudotime')
    axes[1, 0].set_title(f'Pseudotime Correlation (r={correlation:.3f})')
    
    # Plot feature overlap
    rankings = pd.read_csv(feature_rankings_path)
    top_features_knn = set(rankings['features_knn'][:n_top_features])
    top_features_weighted = set(rankings['features_knn_weighted'][:n_top_features])
    
    venn_data = venn.venn2([top_features_knn, top_features_weighted], 
                          set_labels=('KNN', 'Weighted KNN'))
    axes[1, 1].set_title('Feature Overlap')
    
    plt.tight_layout()
    return fig

def compute_trajectory_metrics(trajectories):
    """
    Compute quantitative metrics comparing trajectories
    
    Parameters
    ----------
    trajectories: dict
        Dictionary containing processed AnnData objects
    """
    metrics = {}
    
    # Pseudotime correlation
    tau, p_value = kendalltau(trajectories['KNN'].obs['dpt_pseudotime'],
                             trajectories['Weighted KNN'].obs['dpt_pseudotime'])
    metrics['pseudotime_correlation'] = tau
    metrics['correlation_pvalue'] = p_value
    
    # Path conservation score
    # Measure how well neighborhood relationships are preserved
    knn_neighbors = trajectories['KNN'].obsp['connectivities']
    weighted_neighbors = trajectories['Weighted KNN'].obsp['connectivities']
    conservation = (knn_neighbors != 0) == (weighted_neighbors != 0)
    metrics['path_conservation'] = conservation.mean()
    
    return metrics

def main():
    # Load data
    adata = anndata.read_h5ad('/Users/georgeweale/delve/data/adata_RPE.h5ad')
    feature_rankings_path = '/Users/georgeweale/delve/cyril/combined_features.csv'
    
    # Compare trajectories
    trajectories = compare_trajectories(adata, feature_rankings_path)
    
    # Visualize comparison
    fig = visualize_trajectory_comparison(trajectories, feature_rankings_path, n_top_features)
    plt.savefig('trajectory_comparison.png')
    
    # Compute metrics
    metrics = compute_trajectory_metrics(trajectories)
    print("\nTrajectory Comparison Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")

if __name__ == '__main__':
    main() 