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
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity, rbf_kernel
from scipy.sparse import csr_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from umap import UMAP
from scipy.sparse.csgraph import minimum_spanning_tree
from matplotlib.gridspec import GridSpec 
import scanpy as sc  # for PAGA
import networkx as nx
from scipy.stats import spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

#### DELVE FUNCTIONS ####

def delve_fs(adata = None,
            k: int = 10,
            num_subsamples: int = 1000,
            n_clusters: int = 5,
            null_iterations: int = 1000,
            random_state: int = 0,
            n_random_state: int = 10,
            n_pcs = None,
            n_jobs: int = -1):
    """Performs DELVE feature selection 
        - step 1: identifies dynamic seed features to construct a between-cell affinity graph according to dynamic cell state progression
        - step 2: ranks features according to their total variation in signal along the approximate trajectory graph using the Laplacian score
    
    Parameters
    adata: anndata.AnnData
        annotated data object where adata.X is the attribute for preprocessed data (dimensions = cells x features)
    k: int (default = 10)
        number of nearest neighbors for between cell affinity kNN graph construction
    num_subsamples: int (default = 1000)
        number of neighborhoods to subsample when estimating feature dynamics  
    n_clusters: int (default = 5)
        number of feature modules
    null_iterations: int (default = 1000)
        number of iterations for gene-wise permutation testing
    random_state: int (default = 0)
        random seed parameter
    n_random_state: int (default = 10)
        number of kmeans clustering initializations
    n_pcs: int (default = None)
        number of principal components to compute pairwise Euclidean distances for between-cell affinity graph construction. If None, uses adata.X
    n_jobs = int (default = -1)
        number of tasks
    ----------
    Returns
    delta_mean: pd.DataFrame
        dataframe containing average pairwise change in expression of all features across subsampled neighborhoods (dimensions = num_subsamples x features)
    modules: pd.DataFrame
        dataframe containing feature-cluster assignments and permutation p-values (dimensions = features x 2)
    selected_features: pd.DataFrame
        dataframe containing ranked features and Laplacian scores following feature selection (dimensions = features x 1)
    ----------
    """
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    elif n_jobs < -1:
        n_jobs = mp.cpu_count() + 1 + n_jobs

    X, feature_names, obs_names = parse_input(adata) #parse anndata

    try:
        logging.info(f'Step 1: identifying dynamic feature modules')
        sub_idx, _, delta_mean, modules  = seed_select(X = X, feature_names = feature_names, obs_names = obs_names, k = k, num_subsamples = num_subsamples,
                                                    n_clusters = n_clusters, null_iterations = null_iterations, random_state = random_state,
                                                    n_random_state = n_random_state, n_pcs = n_pcs, n_jobs = n_jobs)

        logging.info(f'Step 2: performing feature selection')
        dyn_feats = np.asarray(modules.index[modules['cluster_id'] != 'static'])
        selected_features = feature_select(X = X[sub_idx, :], feature_names = feature_names, dyn_feats = dyn_feats, k = k, n_pcs = n_pcs, n_jobs = n_jobs)
        return delta_mean, modules, selected_features

    except TypeError: #no dynamic seed features were identified
        return None, None, None

def seed_select(X = None,
                feature_names = None,
                obs_names = None,
                k: int = 10, 
                num_subsamples: int = 1000,
                n_clusters: int = 5,
                null_iterations: int = 1000,
                random_state: int = 0,
                n_random_state: int = 10,
                n_pcs = None,
                n_jobs: int = -1):
    """Identifies dynamic seed clusters
    Parameters
    X: np.ndarray (default = None)
        array containing normalized and preprocessed data (dimensions = cells x features)
    feature_names: np.ndarray (default = None)
        array containing feature names
    obs_names: np.ndarray (default = None)
        array containing cell names   
    k: int (default = 10)
        number of nearest neighbors for between cell affinity kNN graph construction
    num_subsamples: int (default = 1000)
        number of neighborhoods to subsample when estimating feature dynamics  
    n_clusters: int (default = 5)
        number of feature modules
    null_iterations: int (default = 1000)
        number of iterations for gene-wise permutation testing
    random_state: int (default = 0)
        random seed parameter
    n_random_state: int (default = 10)
        number of kmeans clustering initializations
    n_pcs: int (default = None)
        number of principal components to compute pairwise Euclidean distances for between-cell affinity graph construction. If None, uses adata.X
    n_jobs = int (default = -1)
        number of tasks
    ----------
    Returns
    sub_idx: np.andarray
        array containing indices of subsampled neighborhoods
    adata_sub: anndata.AnnData
        annotated data object containing subsampled means (dimensions = num_subsamples x features)
    delta_mean: pd.DataFrame
        dataframe containing average pairwise change in expression of all features across subsampled neighborhoods (dimensions = num_subsamples x features)
    modules: pd.DataFrame
        dataframe containing feature-cluster assignments and permutation p-values (dimensions = features x 2)
    ----------
    """                
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    else:
        n_jobs == mp.cpu_count() + 1 + n_jobs

    p = mp.Pool(n_jobs)
    
    np.random.seed(random_state)
    random_state_arr = np.random.randint(0, 1000000, n_random_state)

    logging.info(f'estimating feature dynamics')
    sub_idx, adata_sub, delta_mean = delta_exp(X = X, feature_names = feature_names, obs_names = obs_names, k = k, num_subsamples = num_subsamples, random_state = random_state, n_pcs = n_pcs, n_jobs = n_jobs)

    #identify modules
    mapping_df = pd.DataFrame(index = feature_names)
    pval_df = pd.DataFrame(index = feature_names)
    dyn_feats = []
    random_state_idx = []
    for result in tqdm(p.imap(partial(_run_cluster, delta_mean, feature_names, n_clusters, null_iterations), random_state_arr), 
                            total = n_random_state, desc = 'clustering features and performing feature-wise permutation testing'):        
        if result is not None:
            mapping_df = pd.concat([mapping_df, result[0]], axis = 1)
            pval_df = pd.concat([pval_df, result[1]], axis = 1)
            dyn_feats.append(result[2])
            random_state_idx.append(result[3])

    if len(dyn_feats) == 0:
        logging.warning(f'No feature clusters have a dynamic variance greater than null. Consider changing the number of clusters or the subsampling size.')
    else:
        dyn_feats = list(np.unique(list(set.intersection(*map(set,dyn_feats)))))
        if len(dyn_feats) == 0:
            logging.warning(f'No features were considered dynamically-expressed across runs.')
        else:
            modules = _annotate_clusters(mapping_df = mapping_df, dyn_feats = dyn_feats, pval_df = pval_df, random_state_idx = random_state_idx[-1])  
            n_dynamic_clusters = len(np.unique(modules['cluster_id'][modules['cluster_id'] != 'static']))
            logging.info(f'identified {n_dynamic_clusters} dynamic cluster(s)')
            return sub_idx, adata_sub, delta_mean, modules

def feature_select(X = None,
                    feature_names = None,
                    dyn_feats = None,
                    k: int = 10,
                    n_pcs = None, 
                    n_jobs: int = -1):
    """Ranks features along dynamic seed graph using the Laplacian score: https://papers.nips.cc/paper/2005/file/b5b03f06271f8917685d14cea7c6c50a-Paper.pdf
    Parameters
    X: np.ndarray (default = None)
        array containing normalized and preprocessed data (dimensions = cells x features)
    feature_names: np.ndarray (default = None)
        array containing feature names
    dyn_feats: np.ndarray (default = None)
        array containing features that are dynamically expressed. Can consider replacing this with a set of known regulators.
    k: int (default = 10)
        number of nearest neighbors for between cell affinity kNN graph construction
    n_pcs: int (default = None)
        number of principal components to compute pairwise Euclidean distances for between-cell affinity graph construction. If None, uses adata.X
    n_jobs = int (default = -1)
        number of tasks
    ----------
    Returns
    selected_features: pd.DataFrame
        dataframe containing ranked features and Laplacian scores for feature selection (dimensions = features x 1)
    ----------
    """
    f_idx = np.where(np.isin(feature_names, dyn_feats) == True)[0] #index of feature names to construct seed graph
    W = construct_affinity(X = X[:, f_idx], k = k, n_pcs = n_pcs, n_jobs = n_jobs) #constructs graph using dynamic seed features
    scores = laplacian_score(X = X, W = W)
    selected_features = pd.DataFrame(scores, index = feature_names, columns = ['DELVE'])
    selected_features = selected_features.sort_values(by = 'DELVE', ascending = True)

    return selected_features

def delta_exp(X = None,
            feature_names = None, 
            obs_names = None,
            k: int = 10,
            num_subsamples: int = 1000,
            random_state: int = 0,
            n_pcs = None,
            n_jobs: int = -1):
    """Estimates change in expression of features across representative cellular neighborhoods
    Parameters
    X: np.ndarray (default = None)
        array containing normalized and preprocessed data (dimensions = cells x features)
    feature_names: np.ndarray (default = None)
        array containing feature names
    obs_names: np.ndarray (default = None)
        array containing cell names   
    k: int (default = 10)
        number of nearest neighbors for between cell affinity kNN graph construction
    num_subsamples: int (default = 1000)
        number of neighborhoods to subsample when estimating feature dynamics  
    random_state: int (default = 0)
        random seed parameter
    n_pcs: int (default = None)
        number of principal components for between-cell affinity graph computation. if None, uses adata.X to find pairwise Euclidean distances 
    n_jobs = int (default = -1)
        number of tasks
    ----------
    Returns
    sub_idx: np.ndarray
        array containing indices of subsampled neighborhoods
    adata_sub: anndata.AnnData
        annotated data object containing subsampled means (dimensions = num_subsamples x features)
    delta_mean: pd.DataFrame (dimensions = num_subsamples x features)
        array containing average pairwise change in expression of all features across subsampled neighborhoods (dimensions = num_subsamples x features)
    ----------
    """
    #construct between cell affinity kNN graph according to all profiled features
    W = construct_affinity(X = X, k = k, n_pcs = n_pcs, n_jobs = -1)

    #compute neighborhood means
    n_bool = W.astype(bool)
    n_mean = (X.transpose() @ n_bool) / np.asarray(n_bool.sum(1)).reshape(1,-1)
    n_mean = pd.DataFrame(n_mean.transpose(), index = obs_names, columns = feature_names)

    #perform subsampling of means to get representative neighborhoods using kernel herding sketching: https://dl.acm.org/doi/abs/10.1145/3535508.3545539, https://github.com/CompCy-lab/SketchKH
    sub_idx, adata_sub = sketch(anndata.AnnData(n_mean), num_subsamples = num_subsamples, frequency_seed = random_state, n_jobs = n_jobs)

    #compute the average pairwise change in the expression across all neighborhoods for all features
    subsampled_means = np.asarray(adata_sub.X, dtype = np.float32)
    delta_mean = subsampled_means.reshape(-1, 1, subsampled_means.shape[1]) - subsampled_means.reshape(1, -1,subsampled_means.shape[1])
    delta_mean = delta_mean.sum(axis = 1) * (1 / (subsampled_means.shape[0] - 1))
    delta_mean = pd.DataFrame(delta_mean[np.argsort(adata_sub.obs.index)], index = adata_sub.obs.index[np.argsort(adata_sub.obs.index)], columns = adata_sub.var_names) #resort according to subsampled indices

    return sub_idx[0], adata_sub, delta_mean

def laplacian_score(X = None,
                    W = None):
    """Computes the Laplacian score
    Parameters
    X: np.ndarray (default = None)
        array containing normalized and preprocessed data (dimensions = cells x features)
    W: np.ndarray (default = None)
        adjacency matrix containing between-cell affinity weights
    ----------
    Returns
    l_score: np.ndarray
        array containing laplacian score for all features (dimensions = features)
    ----------
    """
    n_samples, n_features = X.shape
    
    # Ensure W has correct dimensions
    if W.shape[0] != n_samples or W.shape[1] != n_samples:
        raise ValueError(f"Affinity matrix W has shape {W.shape} but expected ({n_samples}, {n_samples})")
    
    # Compute degree matrix
    D = np.array(W.sum(axis = 1))
    D = scipy.sparse.diags(np.transpose(D), [0])
    
    # Convert D to dense if it's sparse
    D_array = D.toarray()
    
    # Compute graph laplacian
    L = D_array - W.toarray() if scipy.sparse.issparse(W) else D_array - W

    # Ones vector: 1 = [1,···,1]'
    ones = np.ones((n_samples, 1))  # Changed to column vector

    # Feature vector: fr = [fr1,...,frm]'
    fr = X.copy()

    # Construct fr_t = fr - (fr' D 1/ 1' D 1) 1
    numerator = np.matmul(np.matmul(np.transpose(fr), D_array), ones)
    denominator = np.matmul(np.matmul(np.transpose(ones), D_array), ones)
    ratio = numerator / denominator
    fr_t = fr - np.matmul(ones, np.transpose(ratio))

    # Compute laplacian score Lr = fr_t' L fr_t / fr_t' D fr_t
    l_score = np.sum(fr_t * (np.matmul(L, fr_t)), axis=0) / np.sum(fr_t * (np.matmul(D_array, fr_t)), axis=0)

    return l_score

def construct_affinity(X = None,
                        k: int = 10,
                        radius: int = 3,
                        n_pcs = None,
                        n_jobs: int = -1):
    """Computes between cell affinity knn graph using heat kernel
    Parameters
    X: np.ndarray (default = None)
        Data (dimensions = cells x features)
    k: int (default = None)
        Number of nearest neighbors
    radius: int (default = 3)
        Neighbor to compute per cell distance for heat kernel bandwidth parameter
    n_pcs: int (default = None)
        number of principal components to compute pairwise Euclidean distances for between-cell affinity graph construction. If None, uses adata.X
    n_jobs: int (default = -1)
        Number of tasks  
    ----------
    Returns
    W: np.ndarray
        sparse symmetric matrix containing between cell similarity (dimensions = cells x cells)
    ----------
    """
    if n_pcs is not None:
        n_comp = min(n_pcs, X.shape[1])
        pca_op = PCA(n_components=n_comp, random_state = 0)
        X_ = pca_op.fit_transform(X)
    else:
        X_ = X.copy()

    # find kNN
    knn_tree = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric='euclidean', n_jobs=n_jobs).fit(X_)
    dist, nn = knn_tree.kneighbors()  # dist = cells x knn (no self interactions)

    # transform distances using heat kernel
    s = heat_kernel(dist, radius = radius) # -||x_i - x_j||^2 / 2*sigma_i**2
    rows = np.repeat(np.arange(X.shape[0]), k)
    cols = nn.reshape(-1)
    W = scipy.sparse.csr_matrix((s.reshape(-1), (rows, cols)), shape=(X.shape[0], X.shape[0]))

    # make symmetric
    bigger = W.transpose() > W
    W = W - W.multiply(bigger) + W.transpose().multiply(bigger)

    return W

def heat_kernel(dist = None,
                radius: int = 3):
    """Transforms distances into weights using heat kernel
    Parameters
    dist: np.ndarray (default = None)
        distance matrix (dimensions = cells x k)
    radius: np.int (default = 3)
        defines the per-cell bandwidth parameter (distance to the radius nn)
    ----------
    Returns
    s: np.ndarray
        array containing between cell similarity (dimensions = cells x k)
    ----------
    """ 
    epsilon = 1e-10        
    sigma = dist[:, [radius]]  # per cell bandwidth parameter (distance to the radius nn)
    s = np.exp(-1 * (dist**2)/ (2.*(sigma**2 + epsilon ))) # -||x_i - x_j||^2 / 2*sigma_i**2
    return s

def parse_input(adata: anndata.AnnData):
    """Accesses and parses data from adata object
    Parameters
    adata: anndata.AnnData
        annotated data object where adata.X is the attribute for preprocessed data
    ----------
    Returns
    X: np.ndarray
        array of data (dimensions = cells x features)
    feature_names: np.ndarray
        array of feature names
    obs_names: np.ndarray
        array of cell names   
    ----------
    """
    try:
        if isinstance(adata, anndata.AnnData):
            X = adata.X.copy()
        if isinstance(X, scipy.sparse.csr_matrix):
            X = np.asarray(X.todense())

        feature_names = np.asarray(adata.var_names)
        obs_names = np.asarray(adata.obs_names)
        return X, feature_names, obs_names
    except NameError:
        return None

def _run_cluster(delta_mean, feature_names, n_clusters, null_iterations, state):
    """Multiprocessing function for identifying feature modules and performing gene-wise permutation testing
    Parameters
    delta_mean: pd.DataFrame
        dataframe containing average pairwise change in expression of all features across subsampled neighborhoods (dimensions = num_subsamples x features)
    feature_names: np.ndarray (default = None)
        array containing feature names
    n_clusters: int (default = 5)
        number of feature modules
    null_iterations: int (default = 1000)
        number of iterations for gene-wise permutation testing
    state: int (default = 0)
        random seed parameter
    ----------
    Returns
    mapping_df: pd.DataFrame
        dataframe containing feature to cluster assignments
    pval_df: pd.DataFrame
        dataframe containing the permutation p-values 
    dyn_feats: np.ndarray
        array containing features identified as dynamically-expressed following permutation testing
    state: int
        random seed parameter
    ----------
    """     
    #perform clustering     
    clusters = KMeans(n_clusters = n_clusters, random_state = state, init = 'k-means++', n_init = 10).fit_predict(delta_mean.transpose())
    feats = {i:feature_names[np.where(clusters == i)[0]] for i in np.unique(clusters)}

    #record feature-cluster assignment to find intersection across runs
    mapping = np.full((len(feature_names), 1), 'NaN')
    for id, feature in feats.items():
        mapping[np.isin(feature_names, feature)] = str(id)  
    mapping_df = pd.DataFrame(mapping, index = feature_names, columns = [state])

    #compute variance-based permutation test
    seed_var = np.array([np.var(delta_mean.iloc[:, np.isin(feature_names, feats[i])], axis = 1, ddof = 1).mean() for i in range(n_clusters)])
    null_var = []
    pval_df = pd.DataFrame(index = feature_names, columns = [state])
    for f in range(0, len(feats)):
        null_var_ = np.array([np.var(delta_mean.iloc[:, np.isin(feature_names, np.random.choice(feature_names, len(feats[f]), replace = False))], axis = 1, ddof=1).mean() for i in range(null_iterations)])
        permutation_pval = 1 - (len(np.where(seed_var[f] > null_var_)[0]) + 1) / (null_iterations + 1)
        pval_df.loc[feats[f]] = permutation_pval
        null_var.append(np.mean(null_var_))

    dynamic_id = np.where(seed_var > np.array(null_var))[0] #select dynamic clusters over null variance threshold

    if len(dynamic_id) != 0:
        dyn_feats = np.concatenate([v for k, v in feats.items() if k in np.array(list(feats.keys()))[dynamic_id]])
        return mapping_df, pval_df, dyn_feats, state

def _annotate_clusters(mapping_df = None,
                        dyn_feats = None,
                        pval_df = None,
                        random_state_idx: int = None):
    """Annotates clusters as dynamic or static according to feature-wise permutation testing within clusters
    Parameters
    mapping_df: pd.DataFrame
        dataframe containing feature-cluster ids from KMeans clustering across random trials (dimensions = features x n_random_state)
    dyn_feats:  np.ndarray
        array containing features considered to be dynamically expressed across runs
    random_state_idx:  int (default = None)
        id of random state column id in mapping DataFrame to obtain cluster ids  
    ----------
    Returns
    modules: pd.DataFrame
        dataframe containing annotated feature-cluster assignment and permutation p-values (dimensions = features x 2)
    ----------
    """
    cluster_id = np.unique(mapping_df.values)
    dynamic_id = np.unique(mapping_df.loc[dyn_feats].loc[:, random_state_idx])
    static_id = cluster_id[~np.isin(cluster_id, dynamic_id)]

    cats = {id_: 'static' for id_ in static_id}
    cats.update({id_: f'dynamic {i}' if len(dynamic_id) > 1 else 'dynamic' for i, id_ in enumerate(dynamic_id)})

    modules = pd.Categorical(pd.Series(mapping_df.loc[:, random_state_idx].astype('str')).map(cats))
    modules = pd.DataFrame(modules, index = mapping_df.index, columns = ['cluster_id'])
    modules[~np.isin(modules.index, dyn_feats)] = 'static'
    modules['cluster_permutation_pval'] = pval_df.median(1) #median across all random trials
    return modules



### 


#The function performs step 1 completely but stops in step 2 so the output is just the reduced X matrix composed of the selected neighbourhoods
#The reduced matrix can then be used as the input the different algo used to build affinity graphs

def extract_affinity(adata = None,
            k: int = 10,
            num_subsamples: int = 1000,
            n_clusters: int = 5,
            null_iterations: int = 1000,
            random_state: int = 0,
            n_random_state: int = 10,
            n_pcs = None,
            n_jobs: int = -1):
    """Performs DELVE feature selection 
        - step 1: identifies dynamic seed features to construct a between-cell affinity graph according to dynamic cell state progression
        - step 2: ranks features according to their total variation in signal along the approximate trajectory graph using the Laplacian score
    
    Parameters
    adata: anndata.AnnData
        annotated data object where adata.X is the attribute for preprocessed data (dimensions = cells x features)
    k: int (default = 10)
        number of nearest neighbors for between cell affinity kNN graph construction
    num_subsamples: int (default = 1000)
        number of neighborhoods to subsample when estimating feature dynamics  
    n_clusters: int (default = 5)
        number of feature modules
    null_iterations: int (default = 1000)
        number of iterations for gene-wise permutation testing
    random_state: int (default = 0)
        random seed parameter
    n_random_state: int (default = 10)
        number of kmeans clustering initializations
    n_pcs: int (default = None)
        number of principal components to compute pairwise Euclidean distances for between-cell affinity graph construction. If None, uses adata.X
    n_jobs = int (default = -1)
        number of tasks
    ----------
    Returns
    delta_mean: pd.DataFrame
        dataframe containing average pairwise change in expression of all features across subsampled neighborhoods (dimensions = num_subsamples x features)
    modules: pd.DataFrame
        dataframe containing feature-cluster assignments and permutation p-values (dimensions = features x 2)
    selected_features: pd.DataFrame
        dataframe containing ranked features and Laplacian scores following feature selection (dimensions = features x 1)
    ----------
    """
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    elif n_jobs < -1:
        n_jobs = mp.cpu_count() + 1 + n_jobs

    X, feature_names, obs_names = parse_input(adata) #parse anndata


    logging.info(f'Step 1: identifying dynamic feature modules')
    sub_idx, _, delta_mean, modules  = seed_select(X = X, feature_names = feature_names, obs_names = obs_names, k = k, num_subsamples = num_subsamples,
                                                n_clusters = n_clusters, null_iterations = null_iterations, random_state = random_state,
                                                n_random_state = n_random_state, n_pcs = n_pcs, n_jobs = n_jobs)

    logging.info(f'Step 2: performing feature selection')
    dyn_feats = np.asarray(modules.index[modules['cluster_id'] != 'static'])
    f_idx = np.where(np.isin(feature_names, dyn_feats) == True)[0]
    X_reduced = X[:, f_idx]

    return X, X_reduced, feature_names


def rest_of_step2(adata):

    k = 10
    n_pcs = None
    n_jobs = -1
    X, X_reduced, feature_names = extract_affinity(adata=adata, k=10, num_subsamples=1000, n_clusters=5, random_state=0, n_jobs=-1)

    W = construct_affinity(X = X_reduced, k = k, n_pcs = n_pcs, n_jobs = n_jobs) #constructs graph using dynamic seed features
    print(W)
    scores = laplacian_score(X = X, W = W)
    selected_features = pd.DataFrame(scores, index = feature_names, columns = ['DELVE'])
    selected_features = selected_features.sort_values(by = 'DELVE', ascending = True)

    return selected_features


def find_selected_features(W , X, feature_names):
    scores = laplacian_score(X = X, W = W)
    selected_features = pd.DataFrame(scores, index = feature_names, columns = ['DELVE'])
    selected_features = selected_features.sort_values(by = 'DELVE', ascending = True)

    return selected_features


### Initial kNN affinity graph

def kNN_affinity(X_reduced, k: int = 10, radius: int = 3, n_pcs = None, n_jobs: int = -1):

    W = construct_affinity(X = X_reduced, k = k, n_pcs = n_pcs, n_jobs = n_jobs, radius=radius)
    return W


### K-means affinity graph

def construct_affinity_kmeans(X=None, n_clusters: int =10, radius: int =3, n_pcs=None):
    """
    Constructs an affinity graph using KMeans clustering and heat kernel.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix (cells x features).
    n_clusters : int
        Number of clusters for KMeans.
    radius : int
        Radius parameter for the heat kernel.
    n_pcs : int or None
        Number of principal components to use for affinity construction.
    
    Returns
    -------
    W : scipy.sparse.csr_matrix
        Sparse symmetric matrix representing within-cluster affinities.
    """
    
    # Optionally apply PCA to reduce dimensionality
    if n_pcs is not None:
        n_comp = min(n_pcs, X.shape[1])
        pca_op = PCA(n_components=n_comp, random_state=0)
        X_ = pca_op.fit_transform(X)
    else:
        X_ = X.copy()

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_)
    labels = kmeans.labels_  # Get cluster labels for each cell
    
    rows, cols, data = [], [], []

    # Compute pairwise distances within each cluster and apply heat kernel
    for cluster in range(n_clusters):
        cluster_indices = np.where(labels == cluster)[0]  # Get indices of points in this cluster
        if len(cluster_indices) > 1:  # Skip clusters with only one point
            cluster_distances = euclidean_distances(X_[cluster_indices])  # Pairwise distances
            affinity_matrix = heat_kernel(cluster_distances, radius=radius)  # Apply heat kernel
            
            # Store non-zero affinities
            for i in range(len(cluster_indices)):
                for j in range(i+1, len(cluster_indices)):  # Avoid self-loops
                    rows.append(cluster_indices[i])
                    cols.append(cluster_indices[j])
                    data.append(affinity_matrix[i, j])
                    # Since it's symmetric, also store the reverse pair
                    rows.append(cluster_indices[j])
                    cols.append(cluster_indices[i])
                    data.append(affinity_matrix[i, j])

    # Create sparse affinity matrix
    W = csr_matrix((data, (rows, cols)), shape=(X.shape[0], X.shape[0]))

    return W

def kmeans_affinity(X_reduced, n_clusters:int =10, n_pcs = None, radius:int =3):

    W_kmeans = construct_affinity_kmeans(X=X_reduced, n_clusters=n_clusters, radius=radius, n_pcs=n_pcs)
    return W_kmeans

### RBF affinity graph

def rbf_affinity(X, gamma=None, threshold:int =1e-4):
    """
    Computes a sparse affinity graph using RBF (Gaussian) kernel.
    
    Parameters:
        X : np.ndarray
            Data matrix (samples x features)
        gamma : float, optional
            Free parameter of the RBF kernel. If None, uses 1/n_features.
        threshold : float
            Minimum value of affinity to retain in the sparse matrix.
    
    Returns:
        W_rbf : scipy.sparse.csr_matrix
            Sparse affinity matrix (samples x samples)
    """
    # Compute RBF kernel to get the full affinity matrix
    affinity_matrix = rbf_kernel(X, gamma=gamma)
    
    # Set values below the threshold to zero
    affinity_matrix[affinity_matrix < threshold] = 0
    
    # Create a sparse matrix from the non-zero entries
    W_rbf = csr_matrix(affinity_matrix)
    
    return W_rbf







### Similarity measures

def compute_cosine_similarity(W1, W2):
    """Compute cosine similarity between two affinity matrices W1 and W2."""
    W1_flat = W1.todense().A.flatten()
    W2_flat = W2.todense().A.flatten()

    return cosine_similarity(W1_flat.reshape(1, -1), W2_flat.reshape(1, -1))[0][0]

def construct_affinity_monocle3(X, k=10, n_pcs=50, radius=3):
    """
    Constructs affinity graph using Monocle3-style approach with UMAP and MST.
    """
    # UMAP embedding
    umap = UMAP(n_neighbors=k, n_components=2, random_state=0)
    X_umap = umap.fit_transform(X)
    
    # Construct kNN graph in UMAP space
    knn_graph = NearestNeighbors(n_neighbors=k)
    knn_graph.fit(X_umap)
    dist_matrix = knn_graph.kneighbors_graph(mode='distance')
    
    # Get minimum spanning tree
    mst = minimum_spanning_tree(dist_matrix)
    
    # Convert MST to CSR format if it isn't already
    mst = mst.tocsr()
    
    # Create a copy of the MST structure with modified weights
    W = mst.copy()
    W.data = np.exp(-W.data**2 / (2 * radius**2))
    
    # Make symmetric
    bigger = W.transpose() > W
    W = W - W.multiply(bigger) + W.transpose().multiply(bigger)
    
    return W

def construct_affinity_slingshot(X, n_clusters=10, radius=3):
    """
    Constructs affinity graph using Slingshot-style approach with clusters.
    """
    # Initial clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    
    # Compute pairwise distances between cluster centers
    cluster_distances = euclidean_distances(centers)
    
    # Create sparse matrix for affinity graph
    n_samples = X.shape[0]
    rows, cols, data = [], [], []
    
    # Connect points within clusters and between adjacent clusters
    for i in range(n_clusters):
        cluster_i_indices = np.where(labels == i)[0]
        
        # Within-cluster connections
        for idx1 in cluster_i_indices:
            for idx2 in cluster_i_indices:
                if idx1 < idx2:
                    dist = np.linalg.norm(X[idx1] - X[idx2])
                    affinity = np.exp(-dist**2 / (2 * radius**2))
                    rows.extend([idx1, idx2])
                    cols.extend([idx2, idx1])
                    data.extend([affinity, affinity])
    
    # Create sparse matrix
    W = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
    return W

def construct_affinity_paga(X, n_neighbors=15):
    """
    Constructs affinity graph using PAGA approach.
    """
    # Create anndata object
    adata = sc.AnnData(X)
    
    # Compute neighborhood graph
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    
    # Run Leiden clustering
    sc.tl.leiden(adata)
    
    # Run PAGA
    sc.tl.paga(adata, groups='leiden')
    
    # Get cluster-level connectivity matrix
    cluster_connectivity = adata.uns['paga']['connectivities']
    
    # Get cell cluster assignments
    cell_clusters = pd.Categorical(adata.obs['leiden']).codes
    
    # Create cell-level connectivity matrix
    n_cells = X.shape[0]
    rows, cols, data = [], [], []
    
    # For each pair of connected clusters
    cluster_pairs = np.array(cluster_connectivity.nonzero()).T
    for c1, c2 in cluster_pairs:
        # Get cells in each cluster
        cells_c1 = np.where(cell_clusters == c1)[0]
        cells_c2 = np.where(cell_clusters == c2)[0]
        
        # Connection strength between clusters
        strength = cluster_connectivity[c1, c2]
        
        # Connect all cells between clusters
        for cell1 in cells_c1:
            for cell2 in cells_c2:
                if cell1 != cell2:  # Avoid self-loops
                    rows.extend([cell1, cell2])
                    cols.extend([cell2, cell1])
                    data.extend([strength, strength])
    
    # Create sparse matrix
    W = csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
    
    return W

def compare_affinity_methods(X, feature_names, k=10, n_clusters=10, radius=3, n_pcs=50):
    """
    Compare different affinity graph construction methods.
    """
    # Get affinity matrices from different methods
    W_knn = kNN_affinity(X, k=k, radius=radius)
    W_kmeans = kmeans_affinity(X, n_clusters=n_clusters, radius=radius)
    W_monocle = construct_affinity_monocle3(X, k=k, n_pcs=n_pcs, radius=radius)
    W_slingshot = construct_affinity_slingshot(X, n_clusters=n_clusters, radius=radius)
    W_paga = construct_affinity_paga(X, n_neighbors=k)
    
    # Calculate feature rankings for each method
    features_knn = find_selected_features(W_knn, X, feature_names)
    features_kmeans = find_selected_features(W_kmeans, X, feature_names)
    features_monocle = find_selected_features(W_monocle, X, feature_names)
    features_slingshot = find_selected_features(W_slingshot, X, feature_names)
    features_paga = find_selected_features(W_paga, X, feature_names)

    # Set style for better visualization
    plt.style.use('default')
    
    # Create main comparison plot with enhanced spacing
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[1.5, 1.5, 1])
    
    # Enhanced color palette for methods
    method_colors = {
        'kNN': '#FF6B6B',      # Coral Red
        'k-means': '#4ECDC4',  # Turquoise
        'Monocle3': '#45B7D1', # Sky Blue
        'Slingshot': '#96CEB4',# Sage Green
        'PAGA': '#FFB347'     # Orange
    }

    # 1. UMAP visualization with affinity graphs
    ax_umap = fig.add_subplot(gs[0, :])
    embedder = UMAP(n_components=2, random_state=0)
    X_umap = embedder.fit_transform(X)
    
    # Plot base UMAP with enhanced scatter
    scatter = ax_umap.scatter(X_umap[:, 0], X_umap[:, 1], 
                            c=np.arange(X_umap.shape[0]), 
                            cmap='viridis', 
                            s=2, alpha=0.7)
    ax_umap.set_title('UMAP Visualization with Different Affinity Graphs', 
                     fontsize=14, pad=20)
    
    # Add affinity connections with enhanced visibility
    methods = {
        'kNN': W_knn,
        'k-means': W_kmeans,
        'Monocle3': W_monocle,
        'Slingshot': W_slingshot,
        'PAGA': W_paga
    }
    
    for method_name, W in methods.items():
        rows, cols = W.nonzero()
        # Intelligent subsampling based on matrix density
        n_samples = min(300, len(rows))
        mask = np.random.choice(len(rows), size=n_samples, replace=False)
        
        for i, j in zip(rows[mask], cols[mask]):
            ax_umap.plot([X_umap[i, 0], X_umap[j, 0]], 
                        [X_umap[i, 1], X_umap[j, 1]], 
                        c=method_colors[method_name], 
                        linewidth=0.3, 
                        alpha=0.4, 
                        label=method_name)
    
    # Enhanced legend
    handles = [plt.Line2D([0], [0], color=color, label=method, linewidth=2) 
              for method, color in method_colors.items()]
    leg = ax_umap.legend(handles=handles, 
                        loc='center left', 
                        bbox_to_anchor=(1.02, 0.5),
                        frameon=True,
                        fontsize=10)
    leg.get_frame().set_alpha(0.9)

    # 2. Feature ranking correlation heatmap
    ax_corr = fig.add_subplot(gs[1, :2])
    
    all_features = pd.DataFrame({
        'kNN': features_knn['DELVE'],
        'k-means': features_kmeans['DELVE'],
        'Monocle3': features_monocle['DELVE'],
        'Slingshot': features_slingshot['DELVE'],
        'PAGA': features_paga['DELVE']
    })
    
    corr_matrix = all_features.corr()
    
    # Enhanced heatmap
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='RdYlBu_r', 
                vmin=-1, 
                vmax=1, 
                ax=ax_corr,
                annot_kws={'size': 10},
                square=True,
                fmt='.2f')
    
    ax_corr.set_title('Feature Ranking Correlation Between Methods', 
                      fontsize=14, pad=20)
    ax_corr.set_xticklabels(ax_corr.get_xticklabels(), rotation=45, ha='right')
    ax_corr.set_yticklabels(ax_corr.get_yticklabels(), rotation=0)

    # 3. Top features comparison with colored shared genes
    ax_top = fig.add_subplot(gs[1, 2])
    
    # Get top 10 features from each method
    top_features = pd.DataFrame({
        'kNN': features_knn.nlargest(10, 'DELVE').index,
        'k-means': features_kmeans.nlargest(10, 'DELVE').index,
        'Monocle3': features_monocle.nlargest(10, 'DELVE').index,
        'Slingshot': features_slingshot.nlargest(10, 'DELVE').index,
        'PAGA': features_paga.nlargest(10, 'DELVE').index
    })
    
    # Create color mapping for shared genes
    all_genes = set()
    for col in top_features.columns:
        all_genes.update(top_features[col])
    
    # Create a colormap for shared genes
    n_unique_genes = len(all_genes)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_unique_genes))
    gene_to_color = dict(zip(all_genes, colors))
    
    # Create cell colors matrix
    cell_colors = np.zeros((top_features.shape[0], top_features.shape[1], 4))
    for i in range(top_features.shape[0]):
        for j in range(top_features.shape[1]):
            gene = top_features.iloc[i, j]
            cell_colors[i, j] = gene_to_color[gene]
    
    # Enhanced table with colored cells
    ax_top.axis('tight')
    ax_top.axis('off')
    table = ax_top.table(cellText=top_features.values,
                        colLabels=top_features.columns,
                        loc='center',
                        cellLoc='center',
                        cellColours=cell_colors)
    
    # Enhance table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.8)
    
    # Style the header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#E6E6E6')
        cell.set_edgecolor('#FFFFFF')
    
    ax_top.set_title('Top 10 Features by Method\n(Same colors indicate shared genes)', 
                     fontsize=14, pad=20)

    # Adjust layout
    plt.tight_layout()
    
    # Save high-resolution figure
    plt.savefig('affinity_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'features': {
            'kNN': features_knn,
            'k-means': features_kmeans,
            'Monocle3': features_monocle,
            'Slingshot': features_slingshot,
            'PAGA': features_paga
        },
        'affinity_matrices': methods
    }

def compare_graph_metrics(methods):
    """
    Compare different affinity graphs using various graph metrics.
    
    Parameters:
    methods: dict
        Dictionary containing the affinity matrices from different methods
    """
    # Initialize metrics dictionary
    metrics = {
        'Density': [],
        'Average Degree': [],
        'Average Clustering': [],
        'Number of Components': [],
        'Average Path Length': []
    }
    
    method_names = []
    
    # Calculate metrics for each method
    for method_name, W in methods.items():
        print(f"Analyzing {method_name}...")
        
        # Convert sparse matrix to networkx graph
        G = nx.from_scipy_sparse_array(W)
        
        # Calculate metrics
        metrics['Density'].append(nx.density(G))
        metrics['Average Degree'].append(np.mean([d for n, d in G.degree()]))
        metrics['Average Clustering'].append(nx.average_clustering(G))
        metrics['Number of Components'].append(nx.number_connected_components(G))
        
        # Calculate average path length (only for the largest component to avoid inf values)
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        metrics['Average Path Length'].append(nx.average_shortest_path_length(subgraph))
        
        method_names.append(method_name)
    
    # Create visualization
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Radar plot of normalized metrics
    ax_radar = fig.add_subplot(gs[0, 0], projection='polar')
    
    # Normalize metrics for radar plot
    metrics_norm = {}
    for metric in metrics:
        values = metrics[metric]
        min_val = min(values)
        max_val = max(values)
        if max_val - min_val > 0:
            metrics_norm[metric] = [(v - min_val) / (max_val - min_val) for v in values]
        else:
            metrics_norm[metric] = [1 for v in values]
    
    # Number of metrics
    num_metrics = len(metrics)
    angles = [n / float(num_metrics) * 2 * np.pi for n in range(num_metrics)]
    angles += angles[:1]
    
    # Plot for each method
    colors = plt.cm.rainbow(np.linspace(0, 1, len(method_names)))
    for i, method in enumerate(method_names):
        values = [metrics_norm[metric][i] for metric in metrics_norm.keys()]
        values += values[:1]
        ax_radar.plot(angles, values, color=colors[i], linewidth=2, label=method)
        ax_radar.fill(angles, values, color=colors[i], alpha=0.25)
    
    # Set radar chart labels
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(list(metrics.keys()))
    ax_radar.set_title('Normalized Graph Metrics Comparison')
    ax_radar.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))
    
    # 2. Bar plot of raw metrics
    ax_bar = fig.add_subplot(gs[0, 1])
    x = np.arange(len(method_names))
    width = 0.15
    multiplier = 0
    
    for metric, values in metrics.items():
        offset = width * multiplier
        rects = ax_bar.bar(x + offset, values, width, label=metric)
        multiplier += 1
    
    ax_bar.set_ylabel('Value')
    ax_bar.set_title('Raw Graph Metrics')
    ax_bar.set_xticks(x + width * 2)
    ax_bar.set_xticklabels(method_names, rotation=45)
    ax_bar.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # 3. Degree distribution comparison
    ax_degree = fig.add_subplot(gs[1, 0])
    for i, method in enumerate(method_names):
        G = nx.from_scipy_sparse_array(methods[method])
        degrees = [d for n, d in G.degree()]
        ax_degree.hist(degrees, bins=50, alpha=0.5, label=method, color=colors[i])
    
    ax_degree.set_xlabel('Degree')
    ax_degree.set_ylabel('Frequency')
    ax_degree.set_title('Degree Distribution')
    ax_degree.legend()
    ax_degree.set_yscale('log')
    
    # 4. Clustering coefficient distribution
    ax_cluster = fig.add_subplot(gs[1, 1])
    for i, method in enumerate(method_names):
        G = nx.from_scipy_sparse_array(methods[method])
        clustering_coeffs = list(nx.clustering(G).values())
        ax_cluster.hist(clustering_coeffs, bins=50, alpha=0.5, label=method, color=colors[i])
    
    ax_cluster.set_xlabel('Clustering Coefficient')
    ax_cluster.set_ylabel('Frequency')
    ax_cluster.set_title('Clustering Coefficient Distribution')
    ax_cluster.legend()
    ax_cluster.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('graph_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for metric in metrics:
        print(f"\n{metric}:")
        for method, value in zip(method_names, metrics[metric]):
            print(f"{method}: {value:.4f}")

def compare_methods_simple(methods, features_dict, X_umap):
    """
    Create simple visualizations comparing the different methods.
    
    Parameters:
    methods: dict
        Dictionary of affinity matrices from different methods
    features_dict: dict
        Dictionary of feature rankings from different methods
    X_umap: array
        UMAP coordinates of the data
    """
    fig = plt.figure(figsize=(20, 10))
    
    # 1. Edge Density Comparison (Simple Bar Plot)
    plt.subplot(231)
    densities = []
    for method, W in methods.items():
        density = W.nnz / (W.shape[0] * W.shape[1])
        densities.append(density)
    
    plt.bar(methods.keys(), densities)
    plt.title('Edge Density by Method')
    plt.xticks(rotation=45)
    plt.ylabel('Density')
    
    # 2. Feature Overlap Analysis (Top 50 features)
    plt.subplot(232)
    top_n = 50
    feature_sets = {method: set(features['DELVE'].nlargest(top_n).index) 
                   for method, features in features_dict.items()}
    
    overlap_matrix = np.zeros((len(methods), len(methods)))
    for i, (method1, set1) in enumerate(feature_sets.items()):
        for j, (method2, set2) in enumerate(feature_sets.items()):
            overlap = len(set1.intersection(set2)) / top_n
            overlap_matrix[i, j] = overlap
    
    sns.heatmap(overlap_matrix, 
                xticklabels=methods.keys(), 
                yticklabels=methods.keys(),
                annot=True, 
                fmt='.2f', 
                cmap='YlOrRd')
    plt.title(f'Feature Overlap (Top {top_n})')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # 3. Connectivity Pattern Visualization
    plt.subplot(233)
    connectivity_patterns = []
    for W in methods.values():
        # Calculate average connectivity at different distances in UMAP space
        distances = euclidean_distances(X_umap)
        connected = W.toarray() > 0
        avg_connectivity = []
        distance_bins = np.linspace(0, np.percentile(distances, 90), 20)
        
        for i in range(len(distance_bins)-1):
            mask = (distances > distance_bins[i]) & (distances <= distance_bins[i+1])
            avg_connectivity.append(np.mean(connected[mask]))
        
        connectivity_patterns.append(avg_connectivity)
    
    for method, pattern in zip(methods.keys(), connectivity_patterns):
        plt.plot(distance_bins[1:], pattern, label=method, marker='o')
    plt.title('Connectivity vs Distance')
    plt.xlabel('UMAP Distance')
    plt.ylabel('Connection Probability')
    plt.legend()
    
    # 4. Node Degree Distribution (Box Plot)
    plt.subplot(234)
    degrees_data = []
    method_labels = []
    for method, W in methods.items():
        degrees = np.array(W.sum(axis=1)).flatten()
        degrees_data.extend(degrees)
        method_labels.extend([method] * len(degrees))
    
    sns.boxplot(x=method_labels, y=degrees_data)
    plt.title('Node Degree Distribution')
    plt.xticks(rotation=45)
    plt.ylabel('Degree')
    
    # 5. Feature Score Distribution
    plt.subplot(235)
    for method, features in features_dict.items():
        sns.kdeplot(features['DELVE'], label=method)
    plt.title('Feature Score Distribution')
    plt.xlabel('DELVE Score')
    plt.ylabel('Density')
    plt.legend()
    
    # 6. Sparsity Pattern
    plt.subplot(236)
    sparsity_data = []
    for method, W in methods.items():
        non_zero_rows = np.diff(W.indptr)
        sparsity_data.append(non_zero_rows)
    
    plt.boxplot(sparsity_data, labels=methods.keys())
    plt.title('Connections per Cell')
    plt.xticks(rotation=45)
    plt.ylabel('Number of Connections')
    
    plt.tight_layout()
    plt.savefig('method_comparison_simple.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print some summary statistics
    print("\nSummary Statistics:")
    print("\nAverage number of connections per cell:")
    for method, W in methods.items():
        avg_connections = W.nnz / W.shape[0]
        print(f"{method}: {avg_connections:.2f}")
    
    print("\nFeature overlap between methods:")
    for method1, set1 in feature_sets.items():
        for method2, set2 in feature_sets.items():
            if method1 < method2:
                overlap = len(set1.intersection(set2))
                print(f"{method1} vs {method2}: {overlap} features in common")

def main():
    adata = anndata.read_h5ad('/Users/georgeweale/delve/data/adata_RPE.h5ad')

    # Extract data
    X, X_reduced, feature_names = extract_affinity(adata=adata, k=10, 
                                                 num_subsamples=1000, 
                                                 n_clusters=5, 
                                                 random_state=0, 
                                                 n_jobs=-1)

    # Calculate affinity matrices using GAT
    W_gat = construct_affinity_gat(X_reduced, k=10)
    
    # Calculate other affinity matrices
    W_knn = kNN_affinity(X_reduced, k=10)
    W_kmeans = kmeans_affinity(X_reduced, n_clusters=10)
    W_monocle = construct_affinity_monocle3(X_reduced, k=10)
    W_slingshot = construct_affinity_slingshot(X_reduced, n_clusters=10)
    W_paga = construct_affinity_paga(X_reduced, n_neighbors=10)

    # Update methods dictionary
    methods = {
        'GAT': W_gat,
        'kNN': W_knn,
        'k-means': W_kmeans,
        'Monocle3': W_monocle,
        'Slingshot': W_slingshot,
        'PAGA': W_paga
    }

    # Calculate feature rankings
    features_knn = find_selected_features(W_knn, X, feature_names)
    features_kmeans = find_selected_features(W_kmeans, X, feature_names)
    features_monocle = find_selected_features(W_monocle, X, feature_names)
    features_slingshot = find_selected_features(W_slingshot, X, feature_names)
    features_paga = find_selected_features(W_paga, X, feature_names)

    # Set style for better visualization
    plt.style.use('default')
    
    # Create main comparison plot with enhanced spacing
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[1.5, 1.5, 1])
    
    # Enhanced color palette for methods
    method_colors = {
        'kNN': '#FF6B6B',      # Coral Red
        'k-means': '#4ECDC4',  # Turquoise
        'Monocle3': '#45B7D1', # Sky Blue
        'Slingshot': '#96CEB4',# Sage Green
        'PAGA': '#FFB347'     # Orange
    }

    # 1. UMAP visualization with affinity graphs
    ax_umap = fig.add_subplot(gs[0, :])
    embedder = UMAP(n_components=2, random_state=0)
    X_umap = embedder.fit_transform(X_reduced)
    
    # Plot base UMAP with enhanced scatter
    scatter = ax_umap.scatter(X_umap[:, 0], X_umap[:, 1], 
                            c=np.arange(X_umap.shape[0]), 
                            cmap='viridis', 
                            s=2, alpha=0.7)
    ax_umap.set_title('UMAP Visualization with Different Affinity Graphs', 
                     fontsize=14, pad=20)
    
    # Add affinity connections with enhanced visibility
    methods = {
        'GAT': W_gat,
        'kNN': W_knn,
        'k-means': W_kmeans,
        'Monocle3': W_monocle,
        'Slingshot': W_slingshot,
        'PAGA': W_paga
    }
    
    for method_name, W in methods.items():
        rows, cols = W.nonzero()
        # Intelligent subsampling based on matrix density
        n_samples = min(300, len(rows))
        mask = np.random.choice(len(rows), size=n_samples, replace=False)
        
        for i, j in zip(rows[mask], cols[mask]):
            ax_umap.plot([X_umap[i, 0], X_umap[j, 0]], 
                        [X_umap[i, 1], X_umap[j, 1]], 
                        c=method_colors[method_name], 
                        linewidth=0.3, 
                        alpha=0.4, 
                        label=method_name)
    
    # Enhanced legend
    handles = [plt.Line2D([0], [0], color=color, label=method, linewidth=2) 
              for method, color in method_colors.items()]
    leg = ax_umap.legend(handles=handles, 
                        loc='center left', 
                        bbox_to_anchor=(1.02, 0.5),
                        frameon=True,
                        fontsize=10)
    leg.get_frame().set_alpha(0.9)

    # 2. Feature ranking correlation heatmap
    ax_corr = fig.add_subplot(gs[1, :2])
    
    all_features = pd.DataFrame({
        'kNN': features_knn['DELVE'],
        'k-means': features_kmeans['DELVE'],
        'Monocle3': features_monocle['DELVE'],
        'Slingshot': features_slingshot['DELVE'],
        'PAGA': features_paga['DELVE']
    })
    
    corr_matrix = all_features.corr()
    
    # Enhanced heatmap
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='RdYlBu_r', 
                vmin=-1, 
                vmax=1, 
                ax=ax_corr,
                annot_kws={'size': 10},
                square=True,
                fmt='.2f')
    
    ax_corr.set_title('Feature Ranking Correlation Between Methods', 
                      fontsize=14, pad=20)
    ax_corr.set_xticklabels(ax_corr.get_xticklabels(), rotation=45, ha='right')
    ax_corr.set_yticklabels(ax_corr.get_yticklabels(), rotation=0)

    # 3. Top features comparison with colored shared genes
    ax_top = fig.add_subplot(gs[1, 2])
    
    # Get top 10 features from each method
    top_features = pd.DataFrame({
        'kNN': features_knn.nlargest(10, 'DELVE').index,
        'k-means': features_kmeans.nlargest(10, 'DELVE').index,
        'Monocle3': features_monocle.nlargest(10, 'DELVE').index,
        'Slingshot': features_slingshot.nlargest(10, 'DELVE').index,
        'PAGA': features_paga.nlargest(10, 'DELVE').index
    })
    
    # Create color mapping for shared genes
    all_genes = set()
    for col in top_features.columns:
        all_genes.update(top_features[col])
    
    # Create a colormap for shared genes
    n_unique_genes = len(all_genes)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_unique_genes))
    gene_to_color = dict(zip(all_genes, colors))
    
    # Create cell colors matrix
    cell_colors = np.zeros((top_features.shape[0], top_features.shape[1], 4))
    for i in range(top_features.shape[0]):
        for j in range(top_features.shape[1]):
            gene = top_features.iloc[i, j]
            cell_colors[i, j] = gene_to_color[gene]
    
    # Enhanced table with colored cells
    ax_top.axis('tight')
    ax_top.axis('off')
    table = ax_top.table(cellText=top_features.values,
                        colLabels=top_features.columns,
                        loc='center',
                        cellLoc='center',
                        cellColours=cell_colors)
    
    # Enhance table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.8)
    
    # Style the header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#E6E6E6')
        cell.set_edgecolor('#FFFFFF')
    
    ax_top.set_title('Top 10 Features by Method\n(Same colors indicate shared genes)', 
                     fontsize=14, pad=20)

    # Adjust layout
    plt.tight_layout()
    
    # Save high-resolution figure
    plt.savefig('affinity_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create the new comparison visualization
    compare_graph_metrics(methods)

    # In your main function, after creating all matrices and feature rankings:
    features_dict = {
        'kNN': features_knn,
        'k-means': features_kmeans,
        'Monocle3': features_monocle,
        'Slingshot': features_slingshot,
        'PAGA': features_paga
    }

    compare_methods_simple(methods, features_dict, X_umap)

if __name__ == '__main__':
    mp.freeze_support()
    main()


