#@title Module Loading

# Check if on compute cluster
import platform
import re
if re.search("^gra\d+$", platform.node()):
  GRAHAM = True
else:
  GRAHAM = False

# Check if GPU is available
from tensorflow.python.client import device_lib
runtimes = device_lib.list_local_devices()

# Data modules
import numpy as np
if len(runtimes) > 1:
  import cupy as cp
  print("Running a GPU!")
else:
  cp = None
import pandas as pd
import nibabel as nib
from sklearn.linear_model import LinearRegression

# Plotting modules
import matplotlib.pyplot as plt
import seaborn as sns

# Graph theory modules
import networkx as nx

# Stats modules
import scipy as sc
import statsmodels.api as sm


# python modules
import functools as func
import operator as op
import os
from collections import defaultdict
from pathlib import Path
import multiprocessing as mult


#@title Run to load basic parameters
# The download cells will store the data in nested directories starting here:
if GRAHAM:
  HCP_DIR = "./.localdata/hcp"
else:
  HCP_DIR = "./hcp"
if not os.path.isdir(HCP_DIR):
  os.mkdir(HCP_DIR)

# The data shared for NMA projects is a subset of the full HCP dataset
N_SUBJECTS = 339
# Flag for keeping tack if subjects have been adjusted
subj_adjust = False

# The data have already been aggregated into ROIs from the Glasesr parcellation
N_PARCELS = 360

# The acquisition parameters for all tasks were identical
TR = 0.72  # Time resolution, in sec

# The parcels are matched across hemispheres with the same order
HEMIS = ["Right", "Left"]

# Each experiment was repeated multiple times in each subject
N_RUNS_REST = 4
N_RUNS_TASK = 2

#NBACK number of frames per run
FRAMES_PER_RUN = 405

# Time series data are organized by experiment, with each experiment
# having an LR and RL (phase-encode direction) acquistion
BOLD_NAMES = [
  "rfMRI_REST1_LR", "rfMRI_REST1_RL",
  "rfMRI_REST2_LR", "rfMRI_REST2_RL",
  "tfMRI_MOTOR_RL", "tfMRI_MOTOR_LR",
  "tfMRI_WM_RL", "tfMRI_WM_LR",
  "tfMRI_EMOTION_RL", "tfMRI_EMOTION_LR",
  "tfMRI_GAMBLING_RL", "tfMRI_GAMBLING_LR",
  "tfMRI_LANGUAGE_RL", "tfMRI_LANGUAGE_LR",
  "tfMRI_RELATIONAL_RL", "tfMRI_RELATIONAL_LR",
  "tfMRI_SOCIAL_RL", "tfMRI_SOCIAL_LR"
]

# You may want to limit the subjects used during code development.
# This will use all subjects:
subjects = range(N_SUBJECTS)

NODES_OF_INTEREST = ["R_p9-46v", "R_IP2", "R_7Pm", "R_AVI", "L_a9-46v", "L_46", "L_AIP", "L_MI"]
NETWORKS_OF_INTEREST = networks = ['Cingulo-Oper', 'Default', 'Dorsal-atten', 'Frontopariet']

regions = np.load(f"{HCP_DIR}/regions.npy").T
region_info = dict(
    name=regions[0].tolist(),
    network=regions[1],
    myelin=regions[2].astype(np.float),)

if not GRAHAM:
  # Load github directory for access to Glasser atlas file
  pass
else:
  atlas_name = 'MMP_in_MNI_corr.nii'

# Load Glasser atlas
glasser_atlas = nib.load(atlas_name)
img = glasser_atlas.get_fdata()
# Renumber to be continuous from 1 to 360 (right is 1-180, left is 201-380 in original atlas)
for i in range(201,381):
  img[img == i] = i-20
  
# Initialize storage for list of nodes in each network
unique_networks = np.unique(region_info['network'])
nNet = len(unique_networks)          # number of networks
network_regions = {}                 # initialize
for net in unique_networks:
  network_regions[net] = []

# Populate lists and extract node volume
nodeVol = np.zeros(N_PARCELS)
for i in range(N_PARCELS):
  nodeVol[i] = np.sum(img == (i+1))
  network_regions[region_info['network'][i]].append(region_info['name'][i])

with np.load(f"{HCP_DIR}/atlas.npz") as dobj:
  atlas = dict(**dobj)

#@title GPU Utils
#@markdown Run this to prevent errors in functions that use GPU

def get_array_mod(arr):
  if cp:
    return cp.get_array_module(arr)
  else:
    return np

#@title Checkpointing
#@markdown Use the methods here to help create npy files to make checkpoints during pipelines

class Checkpoint:
  checkpoint_folder_name = ".checkpoints"
  def __init__(self):
    self.folder = Path(self.checkpoint_folder_name)
    if not self.folder.is_dir():
      self.folder.mkdir()

  def _file_path(self, name):
    return (self.folder / name).with_suffix('.npy')
  
  def checkpoint_exists(self, name):
    return self._file_path(name).is_file()

  def save_checkpoint(self, name, data):
    np.save(self._file_path(name), data)

  def load_checkpoint(self, name):
    if self.checkpoint_exists(name):
      return np.load(self._file_path(name), allow_pickle=True)
    else:
      return False

  def remove_checkpoint(self, name):
    p = self._file_path(name)
    if p.is_file():
      p.unlink()

#@title Data Loading
#@markdown Run to get functions related to data loading:
#@markdown
#@markdown ```
#@markdown load_timeseries(subject, name, runs=None, concat=True, remove_mean=True)
#@markdown load_task(name, mod=np)
#@markdown ```
def get_image_ids(name):
  """Get the 1-based image indices for runs in a given experiment.

    Args:
      name (str) : Name of experiment ("rest" or name of task) to load
    Returns:
      run_ids (list of int) : Numeric ID for experiment image files

  """
  run_ids = [
    i for i, code in enumerate(BOLD_NAMES, 1) if name.upper() in code
  ]
  if not run_ids:
    raise ValueError(f"Found no data for '{name}''")
  return run_ids

def load_timeseries(subject, name, runs=None, concat=True, remove_mean=True):
  """Load timeseries data for a single subject.
  
  Args:
    subject (int): 0-based subject ID to load
    name (str) : Name of experiment ("rest" or name of task) to load
    run (None or int or list of ints): 0-based run(s) of the task to load,
      or None to load all runs.
    concat (bool) : If True, concatenate multiple runs in time
    remove_mean (bool) : If True, subtract the parcel-wise mean

  Returns
    ts (n_parcel x n_tp array): Array of BOLD data values

  """
  # Get the list relative 0-based index of runs to use
  if runs is None:
    runs = range(N_RUNS_REST) if name == "rest" else range(N_RUNS_TASK)
  elif isinstance(runs, int):
    runs = [runs]

  # Get the first (1-based) run id for this experiment 
  offset = get_image_ids(name)[0]

  # Load each run's data
  bold_data = [
      load_single_timeseries(subject, offset + run, remove_mean) for run in runs
  ]

  # Optionally concatenate in time
  if concat:
    bold_data = np.concatenate(bold_data, axis=-1)

  return bold_data


def load_single_timeseries(subject, bold_run, remove_mean=True):
  """Load timeseries data for a single subject and single run.
  
  Args:
    subject (int): 0-based subject ID to load
    bold_run (int): 1-based run index, across all tasks
    remove_mean (bool): If True, subtract the parcel-wise mean

  Returns
    ts (n_parcel x n_timepoint array): Array of BOLD data values

  """
  bold_path = f"{HCP_DIR}/subjects/{subject}/timeseries"
  bold_file = f"bold{bold_run}_Atlas_MSMAll_Glasser360Cortical.npy"
  ts = np.load(f"{bold_path}/{bold_file}")
  if remove_mean:
    ts -= ts.mean(axis=1, keepdims=True)
  return ts

def load_evs(subject, name, condition):
  """Load EV (explanatory variable) data for one task condition.

  Args:
    subject (int): 0-based subject ID to load
    name (str) : Name of task
    condition (str) : Name of condition

  Returns
    evs (list of dicts): A dictionary with the onset, duration, and amplitude
      of the condition for each run.

  """
  evs = []
  for id in get_image_ids(name):
    task_key = BOLD_NAMES[id - 1]
    ev_file = f"{HCP_DIR}/subjects/{subject}/EVs/{task_key}/{condition}.txt"
    ev_array = np.loadtxt(ev_file, ndmin=2, unpack=True)
    ev = dict(zip(["onset", "duration", "amplitude"], ev_array))
    evs.append(ev)
  return evs

def load_task(name, mod=np):
  """
  Load all timeseries for a given condition as a 3D numpy array:
    (n_subjects x n_parcels x n_timepoints)
  Possible conditions include "rest", "wm", etc. Can return results as either
  a numpy or cupy array

  Args:
    name (str) : Condition to 
    mod (numpy/cupy module) : Module to use for forming the array
  
  Returns:
    ts (sub x parcels x timepoints) : Array of BOLD datapoints
  """
  assert mod is np or mod is cp, "Use either numpy (np) or cupy (cp) as mod"
  if mod is cp and cp:
    return cp.array([cp.asarray(load_timeseries(subject, name)) for subject in subjects])
  return mod.array([load_timeseries(subject, name) for subject in subjects])


task = "wm"
conds = {'0bk': ["0bk_body", "0bk_faces", "0bk_places", "0bk_tools"],
         '2bk': ["2bk_body", "2bk_faces", "2bk_places", "2bk_tools"]}

N_CONDS = len(conds['2bk'])
BLOCK_FRAMES = int(27.5/TR) # 27.5s is the duration of each block

def load_wm_data(mod=np, drop=[]):
  # intialize
  ch = Checkpoint()
  if ch.checkpoint_exists('wm_data'):
    data = np.reshape(ch.load_checkpoint('wm_data'), (1,))[0]
    if mod is cp:
      for l in data:
        data[l] = cp.asarray(data[l])
    return data
  timeseries_task_wm = {}
  for l in ('0bk','2bk'):
    timeseries_task_wm[l] = []

  for subject in range(N_SUBJECTS):
    if subject in drop:
      continue

    # Load full timeseries
    timeseries = load_timeseries(subject=subject, name=task) # this is a 360, 810 numpy.ndarray
    # Extract 2back timepoints 
    for l in ('0bk','2bk'):
      evs = [load_evs(subject, task, cond) for cond in conds[l]]
      ## Notes: evs is a list of 4 (sub)lists - 1 per condition
      # Each sublist contains 2 dictionaries - 1 per run
      # Each dictionary contains onset, duration, and amplitude keys

      ts_block = np.zeros((N_PARCELS, N_CONDS*N_RUNS_TASK*BLOCK_FRAMES))
      for i in range(N_CONDS):
        for j in range(N_RUNS_TASK):
          onset_frames = int(evs[i][j]["onset"]/TR) + j*FRAMES_PER_RUN
          ts_block[:,(i*N_RUNS_TASK+j)*BLOCK_FRAMES:(i*N_RUNS_TASK+j + 1)*BLOCK_FRAMES] = timeseries[:,onset_frames:onset_frames+BLOCK_FRAMES]

      # Concat new timeseries into timeseries_task_wm_2back

      timeseries_task_wm[l].append(ts_block)

  for l in timeseries_task_wm:
    timeseries_task_wm[l] = np.array(timeseries_task_wm[l])
  ch.save_checkpoint('wm_data', timeseries_task_wm)
  if mod is cp:
    for l in timeseries_task_wm:
      timeseries_task_wm[l] = cp.asarray(timeseries_task_wm[l])
  return timeseries_task_wm
    

#@title Graph Thresholding
#@markdown Run this to get the thresholding functions! Main entrypoint is below, 
#@markdown use `help(graph_threshold)` for full description

#@markdown `graph_threshold(input, numSurr)`
#@markdown
#@markdown Be sure to change the runtime to gpu to speed this up as much as possible! Pass in cupy arrays rather than numpy arrays.
def graph_threshold(input, numSurr):
  """
    Thresholds FC data using fourier transform surrogates. Pass in numpy or cupy 
    array. Array must be at least 2 dimensions: the second to last must be nodes
    (e.g. atlas ROI), and the last must be the timecourse. The first
    dimension(s) may be any sort of indexing (e.g. subjects, conditions, etc). 

    Note that using cupy arrays is more than 10 times as fast as using numpy 
    arrays. Be sure to enable GPU in colab.

    Returns array of same type as input array (numpy vs cupy). Shape will 
    be the same as the input, with the exception of the last two dimensions,
    which will contain the thresholded functional connectivity graphs

    Args:
    x (numpy/cupy array of floats with at least 2 dims): data to be thresholded
    numSurr (scalar): Number of surrogates to calculate

    Returns:
    (numpy/cupy array of floats) : Functional connectivity graphs 
  """
  xp = get_array_mod(input)
  index_dims = input.shape[:-2]
  if len(index_dims) > 1:
    # We put the single number inside a tuple
    index_dims = ( func.reduce(op.mul, input.shape[:-2]) )
  elif len(index_dims) == 0:
    index_dims = ( 1, )
  # Here, if index_dims is an empty tuple, it will not contribute anything to
  # the shape (i.e. working_shape will be a 2d tuple)
  shape = (*index_dims, *input.shape[-2:])
  working_input = xp.reshape(input, shape)
  results = xp.empty((*index_dims, *corrcoef_shape(input)))
  for i, sample in enumerate(working_input):
    results[i] = surrogate_threshold(sample, numSurr)
  
  return xp.reshape(results, (*input.shape[:-2], *corrcoef_shape(input)))


def surrogate_threshold(input, numSurr, get_reject=False):
  xp = get_array_mod(input)
  assert(len(input.shape) == 2)
  fc_surr = xp.empty((numSurr, *corrcoef_shape(input)))
  for i in range(numSurr):
    surr = phase_randomize(input)
    fc_surr[i] = xp.corrcoef(surr)

  fc = xp.corrcoef(input)
  cp.cuda.Device().synchronize()

  if xp is cp:
    fc = cp.asnumpy(fc)
    fc_surr = cp.asnumpy(fc_surr)
  pvals = sc.stats.mstats.ttest_1samp(fc_surr, fc, axis=0).pvalue
  reject = two_d_multipletest(pvals)
  if get_reject:
    return reject
  dropped = fc.copy()
  dropped[reject] = 0
  fc[~reject] = 0
  if xp is cp:
    fc = cp.asarray(fc)
  return fc


def two_d_multipletest(pvals, method='bonferroni'):
  reject = sm.stats.multipletests(np.reshape(pvals, -1), method=method)[0]
  return np.reshape(reject, pvals.shape)


def phase_randomize(input):
  xp = get_array_mod(input)
  f_len = input.shape[-1]
  # 1. Calculate the Fourier transform of the original signal.
  f_transform = xp.fft.fft(input, f_len, axis=-1)
  amplitudes = xp.abs(f_transform)
  # 2. Generate a vector of random phases (i.e. a random sequence of values in
  #    the range [0, 2pi]) , with length L/2 , where L is the length of the time
  #    series.
  #    In this implementation, we make phases the same length as the transform,
  #    then symmetrize the phases by setting the front half as equal to the 
  #    negative of the back half.
  phases = xp.random.uniform(-xp.pi, xp.pi, input.shape)
  phases[..., f_len//2:] = -phases[..., f_len//2:0:-1]
  phases[..., 0] = 0

  # 3. As the Fourier transform is symmetrical, to create the new phase 
  #    randomized vector, multiply the first half of F (i.e. the half
  #    corresponding to the positive frequencies) by the phases to create the
  #    first half of F_r. The remainder of F_r is then the horizontally flipped
  #    complex conjugate of the first half. 

  phases_added = amplitudes * xp.exp(1j * phases)


  # 4. Finally, the inverse Fourier transform F_r of gives the FT surrogate. 
  #    Specifying time_len (the length of our original matrix) automatically
  #    pads the input array with 2 0s in each trial (as a result of the above
  #    operations, the array has 2 fewer timepoints than our input)
  return xp.real(xp.fft.ifft(phases_added, f_len))

def corrcoef_shape(a):
  assert(len(a.shape) > 1)
  return (a.shape[-2], a.shape[-2])

def fc_filtering(matrix):
  assert type(matrix) is np.ndarray
  b, a = sc.signal.bessel(4, [0.01*2*TR, 0.1*2*TR], btype='bandpass')
  return sc.signal.filtfilt(b,a, matrix, axis=-1)


#@title Functional Connectivity
#@markdown Defines the class `FC_graph` for graph theory analysis
class FC_graph:
  """
    Class to produce functional connectivity graphs and derive metrics. Takes
    functional covariance matrix as input, along with the region names. 
    Absolutizes the connectivity strength before creating the graph.
  """
  # Nodal properties
  WEIGHT = "weight"
  STRENGTH = "strength"
  STRENGTHNORM = "strengthnorm"
  DISTANCE = "distance"
  CLOSENESS = "closeness"
  BETWEENNESS = "bw"
  PATHLENGTH = "pathlength"
  CLUSTERING = "clustering"
  
  def __init__(self, np_matrix, names):
    """
      Args:
        np_matrix (array[n,n]) : Functional covariance matrix
        names (list[n]) : Names of each of the nodes in the covariance matrix
    """
    self.data = abs(np_matrix)
    self.G = self._prepare_graph(np_matrix, region_info["name"])
    self.names = names
    self._compute_distance(self.G)

  def __len__(self):
    """
      Returns number of nodes in graph
    """
    return nx.number_of_nodes(self.G)

  def _prepare_graph(self, np_matrix, names):
    G = nx.from_numpy_matrix(np_matrix)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    return nx.relabel_nodes(G, dict(enumerate(names)))

  def _compute_distance(self, G):
    # The function accepts a argument 'distance' that, in correlation-based 
    # networks, must be seen as the inverse of the weight value. Thus, a high
    # correlation value (e.g., 0.8) means a shorter distance (i.e., 0.2).
    G_distance_dict = {(e1, e2): 1 / abs(weight) for e1, e2, weight in G.edges(data=self.WEIGHT)}

    # Then add them as attributes to the graph edges
    nx.set_edge_attributes(self.G, G_distance_dict, self.DISTANCE)

  def _compute_degree(self, G, node):
    if not nx.get_node_attributes(G, self.STRENGTH):
      strength = G.degree(weight=self.WEIGHT)
      strengths = dict(strength)
      nx.set_node_attributes(G, strengths, self.STRENGTH) # Add as nodal attribute

    if not nx.get_node_attributes(G, self.STRENGTHNORM):
      # Normalized node strength values 1/N-1
      normstrenghts = {node: val * 1/(len(G.nodes)-1) for (node, val) in strength}
      nx.set_node_attributes(G, normstrenghts, self.STRENGTHNORM)
    return nx.get_node_attributes(G, self.STRENGTHNORM)[node]

  def _compute_betweeness_centrality(self, G, node):
    betweenness = nx.get_node_attributes(G, self.BETWEENNESS)
    if betweenness and node in betweenness:
      return betweenness[node]
    betweenness = nx.betweenness_centrality(G, weight='distance', normalized=True) 
    nx.set_node_attributes(G, betweenness, 'bc')
    return betweenness[node]

  def _compute_shortest_path(self, G, node):
    path = nx.get_node_attributes(G, self.PATHLENGTH)
    if path and node in path:
      return path[node]
    path_lengths = nx.shortest_path_length(G, source=node, weight='distance')
    path_dict = { node: np.mean(list(path_lengths.values())) }
    #nx.set_node_attributes(G, path_dict, self.PATHLENGTH)
    return path_dict[node]

  def _compute_clustering(self, G, node):
    clusters = nx.get_node_attributes(G, self.CLUSTERING)
    if clusters and node in clusters:
      return clusters[node]
    clusters = nx.clustering(G, weight=self.DISTANCE, nodes=[node])
    #nx.set_node_attributes(G, clusters, self.CLUSTERING)
    return clusters[node]


  def _node_subnet(self, node):
    return dict(zip(self.names, self.networks))[node]

  def _make_subnet(self, data, names, subnet_names):
    i = np.array([name in subnet_names for name in names]).nonzero()[0]
    i_1 = np.reshape(i, (1, i.shape[0]))
    i_1 = np.repeat(i_1, i_1.shape[1], axis=0)
    i_2 = np.reshape(i, (i.shape[0], 1))
    i_2 = np.repeat(i_2, i_2.shape[0], axis=1)
    
    return FC_graph(data[i_1, i_2], subnet_names)


  # Methods to produce derivative graphs
  def get_subgraphs(self, region_info):
    """
      Returns a dict of new instances of FC_graph, each entry containing the 
      graph for the network of interest.

      Args:
        region_info (dict) : the region info dict derived from previous code blocks

      Returns:
        dict(network: FC_graph) : the subgraph containing the network of interest
    """
    network_members = defaultdict(list)
    names = region_info["name"]
    for net, name in zip(region_info["network"], names):
      network_members[net].append(name)

    return { network: self._make_subnet(self.data, names, members)
                     for network, members in network_members.items() }

  def get_sparser_graph(self, threshold):
    """
      Thresholds the connectivity based on an absolute threshold and returns
      a new graph

      Args:
        threshold (int) : Number in range [0, 1]
      
      Returns:
        FC_graph
    """
    new_matrix = self.data.copy()
    new_matrix[new_matrix<=threshold] = 0
    return FC_graph(new_matrix, self.region_info)


  # Metric methods

  def hubness(self, nodes):
    """
      Calculates four metrics that relate to the hubness of the given node:
        - Degree: The sum of the node's connections strength to all other nodes
        - Path: The average shortest path length between the node and all others
        - Betweenness Centrality: The proportion of shortest paths between every
            other node that pass through the node of interest
        - Clustering cooefficient: The proportion of the node's neighbors that are
            connected, weighted by their connection strength
      
      These metrics are returned as a dict:
      ```
      { node: np.array(degree, path length, betweenness centrality, clustering cooefficient) }
      ```

      Args:
        nodes (string, list-like) : Either a single node or a list of
          nodes to be evaluated

      Returns:
        dict(np.array) : The four hubness values
    """
    if type(nodes) is str:
      nodes = [nodes]
    assert type(nodes) in (tuple, list, np.array), "Nodes must be an iterable container"
    return { node: np.array([
      self._compute_degree(self.G, node),
      self._compute_shortest_path(self.G, node), 
      self._compute_betweeness_centrality(self.G, node), 
      self._compute_clustering(self.G, node)
    ]) for node in nodes }
  
  def small_worldness(self):
    return nx.smallworld.sigma(self.G)
    

  def mean_degree(self):
    strengths = nx.get_node_attributes(self.G, self.STRENGTHNORM).values()
    normstrengthlist = np.array(list(strengths))
    return np.sum(normstrengthlist)/len(self.G.nodes)


  # Plotting Functions

  def plot_connectome(self, use_node_names=False ):
    fc_matrix = pd.DataFrame(self.data)
    if use_node_names:
      fc_matrix.columns = region_info["name"]
      fc_matrix.index = region_info["name"]
      fc_matrix = fc_matrix.sort_index(0).sort_index(0)
    
    plt.imshow(fc_matrix, interpolation="none", cmap="bwr", vmin=-1, vmax=1)
    plt.colorbar()
    plt.show()
    

  def betweenness_centrality(self):
    betweenness = self._compute_betweeness_centrality(self.G)
    sns.distplot(list(betweenness.values()), kde=False)
    plt.xlabel('Centrality Values')
    plt.ylabel('Counts') 


#@title Graph Theory Pipeline
MODULE = cp #cupy# #@param ["cp #cupy#", "np #numpy#"] {type:"raw"}
NUM_SURRUGATES = 500 #@param {type:"number"}
IGNORE_CHECKPOINTS = False #@param {type:"boolean"}
THRESHOLD_CHECKPOINT = "threshold_filtered"
HUBNESS_CHECKPOINT = "graph-stats-filtered"
NUM_CPUS = 32

import time

def get_graph_stats(subject):
  id, subject = subject
  print("Measuring hubness for Subject {}".format(id))
  graph = FC_graph(subject, region_info["name"])
  stats = graph.hubness(NODES_OF_INTEREST)
  print("Measuring small worldness for subject {}".format(id))
  stats["small_world"] = graph.small_worldness()
  return stats
  


def graph_theory_pipeline():
  ch = Checkpoint()
  if IGNORE_CHECKPOINTS:
    ch.remove_checkpoint(THRESHOLD_CHECKPOINT)
  if not ch.checkpoint_exists(THRESHOLD_CHECKPOINT):
    data = load_wm_data(np)['2bk']
    with mult.Pool(NUM_CPUS) as pool:
      filtered = cp.array(list(pool.imap(fc_filtering, data)))
    fc = graph_threshold(filtered, NUM_SURRUGATES)
    if MODULE is cp:
      fc = cp.asnumpy(fc)
    del data
    ch.save_checkpoint(THRESHOLD_CHECKPOINT, fc)
  else:
    fc = ch.load_checkpoint(THRESHOLD_CHECKPOINT)
  if not ch.checkpoint_exists(HUBNESS_CHECKPOINT):
    with mult.Pool(NUM_CPUS) as pool:
      start_time = time.process_time()
      #hubness = get_graph_stats(fc[0])
      hubness = np.array(list(pool.imap(get_graph_stats, enumerate(fc) )))

      print(time.process_time() - start_time)
      ch.save_checkpoint(HUBNESS_CHECKPOINT, hubness)
  else:
    hubness = ch.load_checkpoint(HUBNESS_CHECKPOINT)
  return hubness


if __name__ == "__main__":
    graph_theory_pipeline()