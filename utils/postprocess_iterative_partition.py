import networkx as nx
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple, Set
import numpy as np

def calculate_strange_count(sub_df: pd.DataFrame, sub_G: nx.Graph) -> Tuple[int, float]:
    """Calculate strange count and ratio for a subcommunity."""
    strange_cnt = 0
    for idx, row in sub_df.iterrows():
        if idx == len(sub_df) - 1:
            break
        if not sub_G.has_edge(sub_df.loc[idx, 'index'], sub_df.loc[idx+1, 'index']):
            strange_cnt += 1
    return strange_cnt, strange_cnt/len(sub_df) if len(sub_df) > 0 else 0

def leiden_community_detection(G: nx.Graph) -> List[Set[int]]:
    """
    Apply Leiden community detection algorithm.
    
    Args:
        G: NetworkX graph
        
    Returns:
        List of node sets representing communities
    """
    try:
        import leidenalg
        import igraph as ig
    except ImportError:
        raise ImportError("leidenalg and igraph packages are required for Leiden algorithm. "
                         "Install with: pip install leidenalg python-igraph")
    
    # Convert NetworkX graph to igraph
    # Create edge list with weights if available
    edges = []
    weights = []
    
    node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
    reverse_mapping = {idx: node for node, idx in node_mapping.items()}
    
    for u, v, data in G.edges(data=True):
        edges.append((node_mapping[u], node_mapping[v]))
        weights.append(data.get('weight', 1.0))
    
    # Create igraph Graph
    ig_graph = ig.Graph(n=len(G.nodes()), edges=edges)
    ig_graph.es['weight'] = weights
    
    # Apply Leiden algorithm using modularity
    # partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)
    partition = leidenalg.ModularityVertexPartition(ig_graph, initial_membership=np.random.choice(2, len(G.nodes())))
    opt = leidenalg.Optimiser()
    opt.consider_empty_community = False
    opt.optimise_partition(partition)
    
    # Convert back to NetworkX node IDs
    communities = []
    for community in partition:
        community_nodes = {reverse_mapping[node] for node in community}
        communities.append(community_nodes)
    
    return communities

def partition_community(
    community_id: int,
    new_df: pd.DataFrame,
    G: nx.Graph,
    min_size: int = 50,
    max_strange_count: int = 10,
    max_strange_ratio: float = 0.1,
    algorithm: str = 'leiden'
) -> List[Dict[int, int]]:
    """
    Recursively partition a community until all subcommunities meet the requirements.
    
    Args:
        community_id: ID of the community to partition
        new_df: DataFrame containing the data
        G: NetworkX graph
        min_size: Minimum size of a community to be considered for partitioning
        max_strange_count: Maximum allowed strange count
        max_strange_ratio: Maximum allowed strange ratio
        algorithm: Community detection algorithm to use ('louvain' or 'leiden')
    
    Returns:
        List of membership maps for all subcommunities
    """
    if algorithm not in ['louvain', 'leiden']:
        raise ValueError("algorithm must be either 'louvain' or 'leiden'")
    
    sub_df = new_df[new_df['new_person_id'] == community_id].copy().reset_index(drop=True)
    sub_node_list = [int(i) for i in sub_df['index'] if i in G.nodes()]
    sub_G = G.subgraph(sub_node_list).copy()
    
    # Calculate strange count and ratio
    strange_cnt, strange_ratio = calculate_strange_count(sub_df, sub_G)

    # If community meets requirements or is too small, return as is
    if (strange_cnt <= max_strange_count and strange_ratio <= max_strange_ratio) or len(sub_df) <= min_size:
        return [{node: 0 for node in sub_node_list}]
    # print(community_id, strange_cnt, strange_ratio)
    # Apply community detection based on selected algorithm
    if algorithm == 'louvain':
        communities = nx.community.louvain_communities(sub_G, weight='weight')
    else:  # leiden
        communities = leiden_community_detection(sub_G)
    
    # If only one community is found, return as is
    if len(communities) == 1:
        return [{node: 0 for node in sub_node_list}]
    
    # Create membership map for current level
    membership_map = {}
    for comm_idx, node_set in enumerate(communities):
        for node in node_set:
            membership_map[node] = comm_idx
    
    # Recursively partition each subcommunity
    all_partitions = []
    for comm_idx, node_set in enumerate(communities):
        # Create a new community ID for this subcommunity
        new_comm_id = f"{community_id}_{comm_idx}"
        
        # Get subcommunity data
        comm_nodes = set(node_set)
        comm_df = sub_df[sub_df['index'].isin(comm_nodes)].copy()
        comm_df['new_person_id'] = new_comm_id
        
        # Recursively partition this subcommunity
        sub_partitions = partition_community(
            new_comm_id,
            comm_df,
            G,
            min_size,
            max_strange_count,
            max_strange_ratio,
            algorithm
        )
        
        # Adjust partition indices to be unique across all partitions
        for partition in sub_partitions:
            adjusted_partition = {
                node: idx + len(all_partitions)
                for node, idx in partition.items()
            }
            all_partitions.append(adjusted_partition)
    
    return all_partitions

def iterative_partitioning(
    new_df: pd.DataFrame,
    G: nx.Graph,
    min_size: int = 50,
    max_strange_count: int = 10,
    max_strange_ratio: float = 0.1,
    algorithm: str = 'leiden'
) -> Dict[int, int]:
    """
    Perform iterative partitioning on all communities.
    
    Args:
        new_df: DataFrame containing the data
        G: NetworkX graph
        min_size: Minimum size of a community to be considered for partitioning
        max_strange_count: Maximum allowed strange count
        max_strange_ratio: Maximum allowed strange ratio
        algorithm: Community detection algorithm to use ('louvain' or 'leiden')
    
    Returns:
        Dictionary mapping node IDs to their final community assignments
    """
    if algorithm not in ['louvain', 'leiden']:
        raise ValueError("algorithm must be either 'louvain' or 'leiden'")
    
    final_partition = {}
    community_id = 0
    
    for person_id in tqdm(new_df['new_person_id'].unique()):
        partitions = partition_community(
            person_id,
            new_df,
            G,
            min_size,
            max_strange_count,
            max_strange_ratio,
            algorithm
        )
        
        # Add partitions to final result
        for partition in partitions:
            for node, comm_idx in partition.items():
                final_partition[node] = community_id + comm_idx
            community_id += len(partition)
    
    return final_partition 