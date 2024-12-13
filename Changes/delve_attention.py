import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
import gc


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

def plot_attention_distribution(attention_scores, feature_names):
    plt.figure(figsize=(12, 6))
    
    # Sort attention scores for better visualization
    sorted_idx = np.argsort(attention_scores)
    sorted_scores = attention_scores[sorted_idx]
    sorted_features = feature_names[sorted_idx]
    
    # Plot distribution
    plt.plot(range(len(sorted_scores)), sorted_scores, 'b-', label='Attention Weights')
    plt.fill_between(range(len(sorted_scores)), sorted_scores, alpha=0.3)
    
    # Add top N feature labels
    top_n = 10
    for i in range(top_n):
        idx = -(i+1)
        plt.annotate(sorted_features[idx], 
                    xy=(len(sorted_scores)+idx, sorted_scores[idx]),
                    xytext=(5, 0), textcoords='offset points')
    
    plt.xlabel('Features (sorted by attention weight)')
    plt.ylabel('Attention Weight')
    plt.title('Distribution of Attention Weights Across Features')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

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

    #ones vector: 1 = [1,··,1]'
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





def main():
    adata = anndata.read_h5ad('/Users/georgeweale/delve/data/adata_RPE.h5ad')

    # 1. Convert data to float32 to reduce memory usage
    X, X_reduced, feature_names = extract_affinity(adata = adata, k = 10, num_subsamples = 1000, n_clusters = 5, random_state = 0, n_jobs = -1)
    X_reduced = X_reduced.astype(np.float32)

    # 2. Move data to device and handle tensor conversion more safely
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_tensor = torch.tensor(X_reduced, dtype=torch.float32, device=device)

    class AttentionMechanism(nn.Module):
        def __init__(self, input_dim):
            super(AttentionMechanism, self).__init__()
            # Initialize weights with smaller values
            self.attention_weights = nn.Parameter(torch.randn(input_dim) * 0.01)
        
        def forward(self, x):
            # Add numerical stability
            attention_scores = torch.softmax(self.attention_weights, dim=0)
            weighted_features = x * attention_scores
            return weighted_features, attention_scores

    # 3. Move model to same device as data
    attention_model = AttentionMechanism(input_dim=X_reduced.shape[1]).to(device)

    # 4. Add gradient clipping and better optimization parameters
    optimizer = optim.Adam(attention_model.parameters(), lr=0.001)
    try:
        # 5. Add error handling and batch processing if needed
        for epoch in range(300):
            optimizer.zero_grad()
            weighted_data, attention_scores = attention_model(data_tensor)
            
            loss = -torch.mean(weighted_data)
            loss.backward()
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(attention_model.parameters(), max_norm=1.0)
            
            optimizer.step()

        # 6. Safely move data back to CPU for numpy operations
        learned_attention_scores = attention_scores.detach().cpu().numpy()
        X_reduced_weighted = X_reduced * learned_attention_scores

        # Rest of your code...
        W_knn = kNN_affinity(X_reduced, k = 10)
        W_kmeans = kmeans_affinity(X_reduced, n_clusters=20)
        # ... 

    except RuntimeError as e:
        print(f"Runtime error occurred: {e}")
        return
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        return

def plot_method_comparison(features_dict):
    # Create correlation matrix between different methods
    methods = list(features_dict.keys())
    corr_matrix = np.zeros((len(methods), len(methods)))
    
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            corr = scipy.stats.spearmanr(
                features_dict[method1]['DELVE'],
                features_dict[method2]['DELVE']
            )[0]
            corr_matrix[i,j] = corr
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, 
                xticklabels=methods,
                yticklabels=methods,
                annot=True,
                cmap='RdYlBu_r',
                vmin=-1,
                vmax=1)
    plt.title('Correlation between Different Affinity Graph Methods')
    plt.tight_layout()

def plot_feature_importance_comparison(original_features, attention_features, top_n=20):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Original rankings
    sns.barplot(x=original_features['DELVE'][:top_n], 
                y=original_features.index[:top_n],
                ax=ax1,
                palette='Blues_r')
    ax1.set_title('Top Features Before Attention')
    
    # Attention-weighted rankings
    sns.barplot(x=attention_features['DELVE'][:top_n],
                y=attention_features.index[:top_n],
                ax=ax2,
                palette='Reds_r')
    ax2.set_title('Top Features After Attention')
    
    plt.tight_layout()

def plot_graph_structure(affinity_matrices_dict, X_reduced, n_samples=1000):
    n_methods = len(affinity_matrices_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
    
    # Use UMAP for 2D projection
    import umap
    reducer = umap.UMAP(random_state=42)
    X_2d = reducer.fit_transform(X_reduced)
    
    for ax, (method_name, W) in zip(axes, affinity_matrices_dict.items()):
        # Plot points
        ax.scatter(X_2d[:, 0], X_2d[:, 1], s=1, alpha=0.5)
        
        # Plot edges
        rows, cols = W.nonzero()
        mask = np.random.choice(len(rows), size=min(n_samples, len(rows)), replace=False)
        
        for i, j in zip(rows[mask], cols[mask]):
            ax.plot([X_2d[i, 0], X_2d[j, 0]], 
                   [X_2d[i, 1], X_2d[j, 1]], 
                   'gray', alpha=0.1)
        
        ax.set_title(method_name)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    mp.freeze_support()
    main()