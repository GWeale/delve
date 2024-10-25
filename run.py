import anndata
from delve import delve_fs
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()  # Required for multiprocessing
    
    # Load your data
    adata = anndata.read_h5ad('data/adata_RPE.h5ad')
    
    # Perform feature selection
    delta_mean, modules, ranked_features = delve_fs(
          adata=adata,
          n_pcs=50,
          k=10,
          num_subsamples=1000,
          n_clusters=5,
          random_state=0,
          n_jobs=-1,
          null_iterations=1000
    )

    # Output the results
    print("Delta Mean:", delta_mean)
    print("Modules:", modules)
    print("Ranked Features:", ranked_features)
