import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['OMP_NUM_THREADS']='1'
os.environ['LOMP_NUM_THREADS']='1'
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
from scipy.sparse import csr_matrix, eye
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph
from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix
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
    try:
        with mp.Pool(n_jobs) as p:
            results = list(tqdm(p.imap(partial(_run_cluster, delta_mean, feature_names, 
                                             n_clusters, null_iterations), 
                                     random_state_arr),
                              total=n_random_state, 
                              desc='clustering features and performing feature-wise permutation testing'))
        p.close()
        p.join()
        
        # Process results
        for result in results:
            if result is not None:
                mapping_df = pd.concat([mapping_df, result[0]], axis=1)
                pval_df = pd.concat([pval_df, result[1]], axis=1)
                dyn_feats.append(result[2])
                random_state_idx.append(result[3])
                
    except Exception as e:
        print(f"Error in multiprocessing: {e}")
        raise

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
    """Computes the Laplacian score: https://papers.nips.cc/paper/2005/file/b5b03f06271f8917685d14cea7c6c50a-Paper.pdf
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
    
    #compute degree matrix
    D = np.array(W.sum(axis = 1))
    D = scipy.sparse.diags(np.transpose(D), [0])

    #compute graph laplacian
    L = D - W.toarray()

    #ones vector: 1 = [1,···,1]'
    ones = np.ones((n_samples,n_features))

    #feature vector: fr = [fr1,...,frm]'
    fr = X.copy()

    #construct fr_t = fr - (fr' D 1/ 1' D 1) 1
    numerator = np.matmul(np.matmul(np.transpose(fr), D.toarray()), ones)
    denomerator = np.matmul(np.matmul(np.transpose(ones), D.toarray()), ones)
    ratio = numerator / denomerator
    ratio = ratio[:, 0]
    ratio = np.tile(ratio, (n_samples, 1))
    fr_t = fr - ratio

    #compute laplacian score Lr = fr_t' L fr_t / fr_t' D fr_t
    l_score = np.matmul(np.matmul(np.transpose(fr_t), L), fr_t) / np.matmul(np.dot(np.transpose(fr_t), D.toarray()), fr_t)
    l_score = np.diag(l_score)

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
    X, X_reduced, feature_names = extract_affinity(adata = adata, k = 10, num_subsamples = 1000, n_clusters = 5, random_state = 0, n_jobs = -1)

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

### GNN affinity graph 


def construct_knn_graph(features, n_neighbors:int =10):
    """
    Constructs a k-Nearest Neighbors (kNN) adjacency matrix from feature data.

    Parameters:
    - features: ndarray, feature matrix (e.g., X_reduced).
    - n_neighbors: int, number of neighbors for kNN graph.

    Returns:
    - csr_matrix representing the kNN adjacency matrix.
    """
    # Build kNN graph as sparse adjacency matrix
    adjacency_matrix = kneighbors_graph(features, n_neighbors=n_neighbors, mode='connectivity', include_self=True)
    return adjacency_matrix

def markov_affinity(data, num_steps:int =50):
    """
    Generates an affinity matrix using a Markov Chain approach.
    
    Parameters:
    - data: csr_matrix, adjacency matrix of the initial graph (e.g., from kNN).
    - num_steps: int, number of steps in the Markov chain to reach steady state.
    
    Returns:
    - csr_matrix representing the Markov chain-based affinity matrix.
    """
    # Ensure the adjacency matrix is normalized
    transition_matrix = normalize(data, norm='l1', axis=1)  # Row-normalize to get transition probabilities
    
    # Power iteration to compute steady-state probabilities
    markov_matrix = transition_matrix
    for _ in range(num_steps):
        markov_matrix = markov_matrix.dot(transition_matrix)
    
    # Convert final steady-state probabilities to affinity matrix (sparse)
    affinity_matrix = markov_matrix + markov_matrix.T  # Ensures symmetry
    affinity_matrix = normalize(affinity_matrix, norm='l1', axis=1)  # Final normalization
    
    return affinity_matrix


### GMM

def gmm_affinity(X, n_components:int =10):
    """
    Generates an affinity matrix based on Gaussian Mixture Model (GMM) responsibilities.
    
    Parameters:
    - X: ndarray, feature matrix (e.g., PCA-reduced data).
    - n_components: int, number of Gaussian components in the GMM.
    
    Returns:
    - csr_matrix representing the GMM-based affinity matrix.
    """
    # Fit GMM to data
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    
    # Compute the responsibility matrix (soft assignments to each component)
    responsibilities = gmm.predict_proba(X)
    
    # Compute affinity matrix based on shared responsibilities
    # The affinity between points i and j is the dot product of their responsibility vectors
    affinity_matrix = responsibilities @ responsibilities.T
    
    # Normalize affinities and ensure it's sparse
    affinity_matrix = normalize(affinity_matrix, norm='l1', axis=1)  # Normalize rows for consistency
    affinity_sparse = csr_matrix(affinity_matrix)  # Convert to sparse matrix
    
    return affinity_sparse


### cosine affinity 


def cosine_affinity(X):
    similarity_matrix = cosine_similarity(X)
    affinity_sparse = csr_matrix(similarity_matrix)  # Convert to sparse matrix
    return affinity_sparse





### Similarity measures

def compute_cosine_similarity(W1, W2):
    """Compute cosine similarity between two affinity matrices W1 and W2."""
    W1_flat = W1.todense().A.flatten()
    W2_flat = W2.todense().A.flatten()

    return cosine_similarity(W1_flat.reshape(1, -1), W2_flat.reshape(1, -1))[0][0]

def compute_tau_scores(features_original, features_model, n_features):
    """
    Compute Tau score between original and model rankings for top n features
    """
    original_ranks = features_original.iloc[:n_features].index.tolist()
    model_ranks = features_model.iloc[:n_features].index.tolist()
    
    # Calculate intersection
    common_elements = set(original_ranks) & set(model_ranks)
    tau_score = len(common_elements) / n_features * 100
    
    return tau_score

def plot_tau_comparisons(features_knn, features_kmeans, features_rbf, features_gmm,
                        features_knn_weighted, features_kmeans_weighted, 
                        features_rbf_weighted, features_gmm_weighted):
    """
    Create comparison plots of Tau scores before and after attention
    """
    # Calculate Tau scores for different numbers of features
    feature_ranges = range(10, 210, 10)
    
    # Before attention
    tau_kmeans = [compute_tau_scores(features_knn, features_kmeans, n) for n in feature_ranges]
    tau_rbf = [compute_tau_scores(features_knn, features_rbf, n) for n in feature_ranges]
    tau_gmm = [compute_tau_scores(features_knn, features_gmm, n) for n in feature_ranges]
    
    # After attention
    tau_kmeans_weighted = [compute_tau_scores(features_knn_weighted, features_kmeans_weighted, n) for n in feature_ranges]
    tau_rbf_weighted = [compute_tau_scores(features_knn_weighted, features_rbf_weighted, n) for n in feature_ranges]
    tau_gmm_weighted = [compute_tau_scores(features_knn_weighted, features_gmm_weighted, n) for n in feature_ranges]
    
    # Create the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Before attention
    ax1.plot(feature_ranges, tau_kmeans, label='k-means', marker='o')
    ax1.plot(feature_ranges, tau_rbf, label='RBF', marker='s')
    ax1.plot(feature_ranges, tau_gmm, label='GMM', marker='^')
    
    ax1.set_xlabel('Number of features')
    ax1.set_ylabel('Tau score (%)')
    ax1.set_title('Feature Ranking Comparison\nBefore Attention')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: After attention
    ax2.plot(feature_ranges, tau_kmeans_weighted, label='k-means', marker='o')
    ax2.plot(feature_ranges, tau_rbf_weighted, label='RBF', marker='s')
    ax2.plot(feature_ranges, tau_gmm_weighted, label='GMM', marker='^')
    
    ax2.set_xlabel('Number of features')
    ax2.set_ylabel('Tau score (%)')
    ax2.set_title('Feature Ranking Comparison\nAfter Attention')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def prepare_data_for_gnn(X, W):
    edge_index, edge_weight = from_scipy_sparse_matrix(W)
    x = torch.tensor(X, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    return data

def main():
    adata = anndata.read_h5ad('/Users/georgeweale/delve/data/adata_RPE.h5ad')

    #selected_features = rest_of_step2(adata)
    #print(selected_features)

    X, X_reduced, feature_names = extract_affinity(adata = adata, k = 10, num_subsamples = 1000, n_clusters = 5, random_state = 0, n_jobs = -1)


    data_tensor = torch.tensor(X_reduced)

    

    class AttentionMechanism(nn.Module):
        def __init__(self, input_dim):
            super(AttentionMechanism, self).__init__()
            self.attention_weights = nn.Parameter(torch.randn(input_dim))
            
        def forward(self, x):
            attention_scores = torch.softmax(self.attention_weights, dim=0)
            weighted_features = x * attention_scores
            return weighted_features, attention_scores

    class AttentionAutoencoder(nn.Module):
        def __init__(self, input_dim, bottleneck_dim):
            super(AttentionAutoencoder, self).__init__()
            self.attention = AttentionMechanism(input_dim)
            self.encoder = nn.Linear(input_dim, bottleneck_dim)
            self.decoder = nn.Linear(bottleneck_dim, input_dim)

        def forward(self, x):
            weighted_features, attention_scores = self.attention(x)
            bottleneck = self.encoder(weighted_features)
            reconstructed = self.decoder(bottleneck)
            
            return reconstructed, attention_scores


    input_dim = X_reduced.shape[1]
    bottleneck_dim = 5  
    autoencoder = AttentionAutoencoder(input_dim=input_dim, bottleneck_dim=bottleneck_dim)
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)

    num_epochs = 600
    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        reconstructed, attention_scores = autoencoder(data_tensor)
        loss = torch.nn.functional.mse_loss(reconstructed, data_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


    learned_attention_scores = attention_scores.detach().numpy()

    plt.plot(range(num_epochs), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Reconstruction Loss')
    plt.title('Training Loss')
    plt.show()

    sns.barplot(x=np.arange(len(learned_attention_scores)), y=learned_attention_scores)
    plt.xlabel('Feature Index')
    plt.ylabel('Attention Score')
    plt.title('Learned Attention Scores for Features')
    plt.show()

    X_reduced_weighted = X_reduced * learned_attention_scores  



    #-----Without attention-------
    W_knn = kNN_affinity(X_reduced, k = 10)
    W_kmeans = kmeans_affinity(X_reduced, n_clusters=20)
    W_rbf = rbf_affinity(X_reduced, gamma=0.03, threshold=1e-2)
    W_gmm = gmm_affinity(X_reduced)

    features_knn = find_selected_features(W_knn, X, feature_names)
    features_kmeans = find_selected_features(W_kmeans, X, feature_names)
    features_rbf = find_selected_features(W_rbf, X, feature_names)
    features_gmm = find_selected_features(W_gmm, X, feature_names)
    
    #-----With attention-------
    W_knn_weighted = kNN_affinity(X_reduced_weighted, k = 10)
    W_kmeans_weighted = kmeans_affinity(X_reduced_weighted, n_clusters=20, radius=2)
    W_rbf_weighted = rbf_affinity(X_reduced_weighted, gamma=0.03, threshold=1e-2)
    W_gmm_weighted = gmm_affinity(X_reduced_weighted)

    features_knn_weighted = find_selected_features(W_knn_weighted, X, feature_names)
    features_kmeans_weighted = find_selected_features(W_kmeans_weighted, X, feature_names)
    features_rbf_weighted = find_selected_features(W_rbf_weighted, X, feature_names)
    features_gmm_weighted = find_selected_features(W_gmm_weighted, X, feature_names)
    
    

    for i in range(len(features_knn)):
        features_knn.iloc[i, 0] = i  
        features_kmeans.iloc[i, 0] = i
        features_rbf.iloc[i, 0] = i  
        features_gmm.iloc[i, 0] = i  

        features_knn_weighted.iloc[i, 0] = i  
        features_kmeans_weighted.iloc[i, 0] = i
        features_rbf_weighted.iloc[i, 0] = i  
        features_gmm_weighted.iloc[i, 0] = i  





    merged_df = features_knn.join(features_kmeans, lsuffix='_knn', rsuffix='_kmeans').join(features_rbf, rsuffix='_rbf').join(features_gmm, rsuffix='_gmm')
    merged_df.columns = ['knn', 'kmeans', 'rbf', 'gmm']

    merged_df_weighted = features_knn_weighted.join(features_kmeans_weighted, lsuffix='_knn', rsuffix='_kmeans').join(features_rbf_weighted, rsuffix='_rbf').join(features_gmm_weighted, rsuffix='_gmm')
    merged_df_weighted.columns = ['knn', 'kmeans', 'rbf', 'gmm']


    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) 

    sns.heatmap(merged_df.iloc[:20,:], cmap='viridis', annot=True, fmt=".3f", ax=axes[0])
    axes[0].set_ylabel('Ranking of features')
    axes[0].set_xlabel('Method to build affinity graph')
    axes[0].set_title('Comparison of features ranking')
    axes[0].tick_params(axis='y', left=False)  

    sns.heatmap(merged_df_weighted.iloc[:20,:], cmap='viridis', annot=True, fmt=".3f", ax=axes[1])
    axes[1].set_ylabel('Ranking of features')
    axes[1].set_xlabel('Method to build affinity graph')
    axes[1].set_title('Comparison of features ranking AFTER ATTENTION')
    axes[1].tick_params(axis='y', left=False) 

    plt.tight_layout()
    plt.show()

    adata = anndata.read_h5ad('/Users/georgeweale/delve/data/adata_RPE.h5ad')

    # Extract data and features
    X, X_reduced, feature_names = extract_affinity(adata=adata, k=10, num_subsamples=1000, n_clusters=5, random_state=0, n_jobs=-1)

    # Construct initial affinity graph (e.g., using kNN)
    W_knn = kNN_affinity(X_reduced, k=10)

    # Prepare data for GNN
    data = prepare_data_for_gnn(X_reduced, W_knn)

    # Define GNN model with matching dimensions
    num_features = X_reduced.shape[1]  # This is 12 based on your error
    hidden_dim = num_features  # Make hidden_dim match input dimensions
    gnn_model = GCN(num_features=num_features, hidden_dim=hidden_dim)

    # Train GNN
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
    num_epochs = 600
    gnn_model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        embeddings = gnn_model(data.x, data.edge_index)
        loss = F.mse_loss(embeddings, data.x)  # Now dimensions will match
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Get GNN embeddings and create affinity matrix
    gnn_model.eval()  # Switch to evaluation mode
    with torch.no_grad():
        embeddings = gnn_model(data.x, data.edge_index)
        embeddings = embeddings.detach().numpy()
    
    # Create affinity matrix from GNN embeddings
    W_gnn = construct_affinity(embeddings, k=10)
    features_gnn = find_selected_features(W_gnn, X, feature_names)

    # Add GNN results to the merged dataframes
    merged_df['gnn'] = features_gnn['DELVE']
    merged_df_weighted['gnn'] = features_gnn['DELVE']

    # Update the plotting code to include GNN
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(merged_df.iloc[:20,:], cmap='viridis', annot=True, fmt=".3f", ax=axes[0])
    axes[0].set_ylabel('Ranking of features')
    axes[0].set_xlabel('Method to build affinity graph')
    axes[0].set_title('Comparison of features ranking')
    axes[0].tick_params(axis='y', left=False)

    sns.heatmap(merged_df_weighted.iloc[:20,:], cmap='viridis', annot=True, fmt=".3f", ax=axes[1])
    axes[1].set_ylabel('Ranking of features')
    axes[1].set_xlabel('Method to build affinity graph')
    axes[1].set_title('Comparison of features ranking AFTER ATTENTION')
    axes[1].tick_params(axis='y', left=False)

    plt.tight_layout()
    plt.show()

    # Update tau scores calculation to include GNN
    feature_ranges = range(10, 210, 10)
    tau_kmeans = [compute_tau_scores(features_knn, features_kmeans, n) for n in feature_ranges]
    tau_rbf = [compute_tau_scores(features_knn, features_rbf, n) for n in feature_ranges]
    tau_gmm = [compute_tau_scores(features_knn, features_gmm, n) for n in feature_ranges]
    tau_gnn = [compute_tau_scores(features_knn, features_gnn, n) for n in feature_ranges]
    
    tau_kmeans_weighted = [compute_tau_scores(features_knn_weighted, features_kmeans_weighted, n) for n in feature_ranges]
    tau_rbf_weighted = [compute_tau_scores(features_knn_weighted, features_rbf_weighted, n) for n in feature_ranges]
    tau_gmm_weighted = [compute_tau_scores(features_knn_weighted, features_gmm_weighted, n) for n in feature_ranges]

    # Plot updated tau scores including GNN with coordinated colors
    plt.figure(figsize=(10, 6))
    
    # Define colors for each method pair
    kmeans_color = '#1f77b4'  # blue
    rbf_color = '#2ca02c'     # green
    gmm_color = '#ff7f0e'     # orange
    gnn_color = '#ff0000'     # red

    # Plot unweighted methods with dashed lines
    plt.plot(feature_ranges, tau_kmeans, label='k-means', linestyle='--', 
             color=kmeans_color, alpha=0.7)
    plt.plot(feature_ranges, tau_rbf, label='RBF', linestyle='--', 
             color=rbf_color, alpha=0.7)
    plt.plot(feature_ranges, tau_gmm, label='GMM', linestyle='--', 
             color=gmm_color, alpha=0.7)
    
    # Plot weighted methods with solid lines
    plt.plot(feature_ranges, tau_kmeans_weighted, label='k-means (weighted)', 
             color=kmeans_color)
    plt.plot(feature_ranges, tau_rbf_weighted, label='RBF (weighted)', 
             color=rbf_color)
    plt.plot(feature_ranges, tau_gmm_weighted, label='GMM (weighted)', 
             color=gmm_color)
    
    # Plot GNN with thick red line
    plt.plot(feature_ranges, tau_gnn, label='GNN', color=gnn_color, 
             linewidth=3, zorder=10)  # zorder ensures GNN line is on top
    
    plt.xlabel('Number of features')
    plt.ylabel('Tau score (%)')
    plt.title('Feature Ranking Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Update the feature count analysis to include GNN
    subset_up = merged_df.iloc[:20,:]
    result = []
    for id in subset_up.iloc[:,0]:
        count = 0
        for col_id in range(subset_up.shape[1]):  # Now includes GNN column
            if id in subset_up.iloc[:,col_id].tolist():
                count += 1
        result.append([feature_names[int(id)], count])
    
    result_df = pd.DataFrame(result)
    result_df = result_df.sort_values(by=1, ascending=False).reset_index(drop=True)

    plt.figure(figsize=(9, 3))
    plt.barh(result_df[0], result_df[1], color='red')
    plt.xlabel('Count')
    plt.xticks(range(subset_up.shape[1] + 1))  # Updated to include GNN
    plt.title('Top 20 ranked features count over methods')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()