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
import os
from scipy.stats import kendalltau

os.environ['OMP_NUM_THREADS'] = '1'


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
#The reduced matrix can then be used as the input the different functions used to build affinity graphs

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

def construct_affinity_kmeans(X=None, n_clusters: int =10, radius: int =2, n_pcs=None):
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
        cluster_indices = np.where(labels == cluster)[0]  
        if len(cluster_indices) > 1: 
            cluster_distances = euclidean_distances(X_[cluster_indices])  
            affinity_matrix = heat_kernel(cluster_distances, radius=radius) 
            
            # Store non-zero affinities
            for i in range(len(cluster_indices)):
                for j in range(i+1, len(cluster_indices)):  
                    rows.append(cluster_indices[i])
                    cols.append(cluster_indices[j])
                    data.append(affinity_matrix[i, j])
                    
                    rows.append(cluster_indices[j])
                    cols.append(cluster_indices[i])
                    data.append(affinity_matrix[i, j])


    W = csr_matrix((data, (rows, cols)), shape=(X.shape[0], X.shape[0]))

    return W

def kmeans_affinity(X_reduced, n_clusters:int =10, n_pcs = None, radius:int =2):

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

    affinity_matrix = rbf_kernel(X, gamma=gamma)
    affinity_matrix[affinity_matrix < threshold] = 0
    
    W_rbf = csr_matrix(affinity_matrix)
    
    return W_rbf



### GMM affinity graph

def gmm_affinity(X, n_components:int =10):
    """
    Generates an affinity matrix based on Gaussian Mixture Model (GMM) responsibilities.
    
    Parameters:
    - X: ndarray, feature matrix (e.g., PCA-reduced data).
    - n_components: int, number of Gaussian components in the GMM.
    
    Returns:
    - csr_matrix representing the GMM-based affinity matrix.
    """

    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    
    responsibilities = gmm.predict_proba(X)
    
    affinity_matrix = responsibilities @ responsibilities.T
    affinity_matrix = normalize(affinity_matrix, norm='l1', axis=1)  
    affinity_sparse = csr_matrix(affinity_matrix)  
    
    return affinity_sparse





def main():
    adata = anndata.read_h5ad('adata_RPE.h5ad')

    X, X_reduced, feature_names = extract_affinity(adata = adata, k = 10, num_subsamples = 1000, n_clusters = 5, random_state = 0, n_jobs = -1)

    data_tensor = torch.tensor(X_reduced)

    # Creation of the model by combining an attention mechanism with an autoencoder to ensure unsupervised learning of attention weights
    class AttentionMechanism(nn.Module):
            def __init__(self, input_dim):
                super(AttentionMechanism, self).__init__()
                self.query = nn.Linear(input_dim, input_dim) 
                self.key = nn.Linear(input_dim, input_dim)   
                self.value = nn.Linear(input_dim, input_dim) 
                self.scale = np.sqrt(input_dim)  

            def forward(self, x):
                Q = self.query(x)  
                K = self.key(x)   
                V = self.value(x) 

                attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale        
                attention_probs = torch.softmax(attention_scores, dim=-1) 
                
                feature_attention_scores = attention_probs.mean(dim=0) 
                feature_attention_scores = feature_attention_scores[:x.size(1)] 
                feature_attention_scores = feature_attention_scores.unsqueeze(0)  

                weighted_features = x * feature_attention_scores 

                return weighted_features, feature_attention_scores


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

    # The weights values are averaged over the learning of 20 different models
    score_list = []
    for i in range(10):

        input_dim = X_reduced.shape[1]
        bottleneck_dim = 5  
        autoencoder = AttentionAutoencoder(input_dim=input_dim, bottleneck_dim=bottleneck_dim)
        optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)

        num_epochs = 25
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
        learned_attention_scores = learned_attention_scores.squeeze() 
        learned_attention_scores = learned_attention_scores.flatten()

        score_list.append(np.array(learned_attention_scores))

    score_list = np.array(score_list)
    means = score_list.mean(axis=0)
    std_devs = score_list.std(axis=0)
    percentile_25 = np.percentile(score_list, 25, axis=0)
    percentile_75 = np.percentile(score_list, 75, axis=0)
    yerr = np.vstack((means - percentile_25, percentile_75 - means))

    learned_attention_scores = score_list.mean(axis=0)


    #----Reconstruction loss plot------
    plt.figure()
    plt.plot(range(num_epochs), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Reconstruction Loss')
    plt.title('Training Loss')
    #plt.show()

    #----Weights values plot-------
    plt.figure()
    sns.barplot(x=np.arange(len(learned_attention_scores)), y=learned_attention_scores)
    plt.xlabel('Feature Index')
    plt.ylabel('Attention Score')
    plt.title('Learned Attention Scores for Features')
    #plt.show()

    #----Weights values plot with interval-------
    plt.figure()
    x = np.arange(len(means))
    plt.bar(x, means, yerr=std_devs, capsize=5, color='blue', alpha=0.3, edgecolor='black')
    plt.xlabel('Feature index')
    plt.ylabel('Value')
    plt.title('Features learned weights')
    plt.xticks(x)
    plt.tight_layout()
    #plt.show()

    # plt.figure()
    # x = np.arange(len(means))
    # plt.bar(x, means, yerr=yerr, capsize=5, color='blue', alpha=0.3, edgecolor='black')
    # plt.xlabel('Feature index')
    # plt.ylabel('Value')
    # plt.title('Features learned weights')
    # plt.xticks(x)
    # plt.tight_layout()
    plt.show()

    X_reduced_weighted = X_reduced * learned_attention_scores  



    # Creation of affinity graphs and ranking of features




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

    #-----Matching of the rankings without attention learning--------
    merged_df = features_knn.join(features_kmeans, lsuffix='_knn', rsuffix='_kmeans').join(features_rbf, rsuffix='_rbf').join(features_gmm, rsuffix='_gmm')
    merged_df.columns = ['knn', 'kmeans', 'rbf', 'gmm']

    #-----Matching of the rankings with attention learning
    merged_df_weighted = features_knn_weighted.join(features_kmeans_weighted, lsuffix='_knn', rsuffix='_kmeans').join(features_rbf_weighted, rsuffix='_rbf').join(features_gmm_weighted, rsuffix='_gmm')
    merged_df_weighted.columns = ['knn', 'kmeans', 'rbf', 'gmm']


    #-----Full ranking comparison plot------
    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) 

    sns.heatmap(merged_df, cmap='viridis', annot=False, fmt=".3f", ax=axes[0])
    axes[0].set_ylabel('Ranking of features')
    axes[0].set_xlabel('Method to build affinity graph')
    axes[0].set_title('Features ranking')
    #axes[0].set_yticks([])
    axes[0].tick_params(axis='y', left=False)  

    sns.heatmap(merged_df_weighted, cmap='viridis', annot=False, fmt=".3f", ax=axes[1])
    axes[1].set_ylabel('Ranking of features')
    axes[1].set_xlabel('Method to build affinity graph')
    axes[1].set_title('Features ranking after attention learning')
    axes[1].tick_params(axis='y', left=False) 
    #axes[1].set_yticks([])

    plt.tight_layout()
    #plt.show()


    #-----Top 20 ranking comparison plot------
    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) 

    sns.heatmap(merged_df.iloc[:20,:], cmap='viridis', annot=True, fmt=".3f", ax=axes[0], vmax=20)
    axes[0].set_ylabel('Ranking of features')
    axes[0].set_xlabel('Method to build affinity graph')
    axes[0].set_title('Top 20 features ranking')
    axes[0].tick_params(axis='y', left=False)  
    axes[0].set_yticks([])

    sns.heatmap(merged_df_weighted.iloc[:20,:], cmap='viridis', annot=True, fmt=".3f", ax=axes[1], vmax=20)
    axes[1].set_ylabel('Ranking of features')
    axes[1].set_xlabel('Method to build affinity graph')
    axes[1].set_title('Top 20 features ranking after attention learning')
    axes[1].tick_params(axis='y', left=False) 
    axes[1].set_yticks([])

    plt.tight_layout()
    #plt.show()




    #----Tau score of the rankings compared to the original DELVE ranking without attnetion learning
    percentages_col1 = []
    percentages_col2 = []
    percentages_col3 = []

    for i in range(1, len(merged_df) + 1):
        subset = merged_df.iloc[:i]  
        
        tau, p_value = kendalltau(subset.iloc[:,0], subset.iloc[:,1])
        percentages_col1.append(tau)
        tau, p_value = kendalltau(subset.iloc[:,0], subset.iloc[:,2])
        percentages_col2.append(tau)
        tau, p_value = kendalltau(subset.iloc[:,0], subset.iloc[:,3])
        percentages_col3.append(tau)

    plt.figure()
    plt.plot(range(1, len(merged_df) + 1), percentages_col1, label='original vs. kmeans')
    plt.plot(range(1, len(merged_df) + 1), percentages_col2, label='original vs. rbf')
    plt.plot(range(1, len(merged_df) + 1), percentages_col3, label='original vs. gmm')
    plt.xlabel('Number of features Considered')
    plt.ylabel('Percentage')
    plt.title('Tau')
    plt.legend()
    #plt.show()


    #----Tau score of the rankings compared to the original DELVE ranking without attnetion learning
    percentages_col1 = []
    percentages_col2 = []
    percentages_col3 = []

    for i in range(1, len(merged_df_weighted) + 1):
        subset = merged_df_weighted.iloc[:i]  
        
        tau, p_value = kendalltau(subset.iloc[:,0], subset.iloc[:,1])
        percentages_col1.append(tau)
        tau, p_value = kendalltau(subset.iloc[:,0], subset.iloc[:,2])
        percentages_col2.append(tau)
        tau, p_value = kendalltau(subset.iloc[:,0], subset.iloc[:,3])
        percentages_col3.append(tau)

    plt.figure()
    plt.plot(range(1, len(merged_df_weighted) + 1), percentages_col1, label='original vs. kmeans')
    plt.plot(range(1, len(merged_df_weighted) + 1), percentages_col2, label='original vs. rbf')
    plt.plot(range(1, len(merged_df_weighted) + 1), percentages_col3, label='original vs. gmm')
    plt.xlabel('Number of features Considered')
    plt.ylabel('Percentage')
    plt.title('Tau attention')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    mp.freeze_support()  # Only needed if you are going to freeze your application
    main()