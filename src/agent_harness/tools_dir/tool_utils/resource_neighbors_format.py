"""
Manifest Graph Processing for incident analysis.

This module processes resource neighbour graph responses from Manifest API
and formats them into human-readable text for LLM consumption.

The module builds end-to-end ordered paths from the graph data, showing
the complete flow from source (e.g., code repository) to target (e.g., container).
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict

logger = logging.getLogger(__name__)

# Known providers to extract from labels
# Labels are typically: ["Node", "RESOURCE", "KUBERNETES", "CONTAINER_ORCHESTRATOR", "POD", ...]
# Provider is usually one of these identifiable keywords
KNOWN_PROVIDERS = [
    "KUBERNETES",
    "GH",           # GitHub
    "GITHUB",
    "ARGOCD",
    "AZUREDEVOPS",
    "AZURE",
    "AWS",
    "GCP",
    "TERRAFORM",
    "HELM",
    "DOCKER",
    "JENKINS",
    "GITLAB",
    "BITBUCKET",
    "DATADOG",
    "PROMETHEUS",
    "GRAFANA",
    "NEWRELIC",
    "SPLUNK",
    "ELASTICSEARCH",
    "REDIS",
    "MONGODB",
    "POSTGRES",
    "MYSQL",
    "KAFKA",
    "RABBITMQ",
    "NGINX",
    "ISTIO",
    "ENVOY",
    "VAULT",
    "CONSUL",
    "HOST"
]


def extract_provider_from_labels(labels: List[str]) -> str:
    """
    Extract provider name from a list of labels.
    
    Args:
        labels: List of label strings (e.g., ["Node", "RESOURCE", "KUBERNETES", "POD"])
        
    Returns:
        Provider name or "Unknown" if not found
    """
    if not labels:
        return "Unknown"
    
    # Check each label against known providers
    for label in labels:
        label_upper = label.upper()
        if label_upper in KNOWN_PROVIDERS:
            # Return a more readable format
            provider_map = {
                "GH": "GitHub",
                "KUBERNETES": "Kubernetes",
                "ARGOCD": "ArgoCD",
                "AZUREDEVOPS": "Azure DevOps",
                "AZURE": "Azure",
                "AWS": "AWS",
                "GCP": "GCP",
                "TERRAFORM": "Terraform",
                "HELM": "Helm",
                "DOCKER": "Docker",
                "JENKINS": "Jenkins",
                "GITLAB": "GitLab",
                "BITBUCKET": "Bitbucket",
                "HOST": "Host"
            }
            return provider_map.get(label_upper, label)
    
    return "Unknown"


def format_node_details(node: Dict[str, Any]) -> str:
    """
    Format node details into a readable string.
    
    Args:
        node: Node dictionary with keys: labels, key, name, resourceSubType, resourceType
        
    Returns:
        Formatted node string
    """
    name = node.get("name", "Unknown")
    resource_sub_type = node.get("resourceSubType", "Unknown")
    labels = node.get("labels", [])
    provider = extract_provider_from_labels(labels)
    
    # Extract resource ID from key (format: "RESOURCE|84235923")
    key = node.get("key", "")
    resource_id = key.split("|")[1] if "|" in key else key
    
    return f"{name} ({resource_sub_type}, {provider}, ID: {resource_id})"


def get_node_info(node: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract structured node information.
    
    Args:
        node: Node dictionary
        
    Returns:
        Dictionary with extracted node info
    """
    labels = node.get("labels", [])
    key = node.get("key", "")
    resource_id = key.split("|")[1] if "|" in key else key
    
    return {
        "key": key,
        "id": resource_id,
        "name": node.get("name", "Unknown"),
        "resourceSubType": node.get("resourceSubType", "Unknown"),
        "resourceType": node.get("resourceType", "Unknown"),
        "provider": extract_provider_from_labels(labels),
        "labels": labels
    }


def build_graph_from_paths(paths: List[Dict[str, Any]]) -> Tuple[Dict, Dict, Dict, Set, Set]:
    """
    Build adjacency list and node info from paths data.
    
    Args:
        paths: List of path objects with from, to, relationship
        
    Returns:
        Tuple of (adjacency_list, reverse_adjacency, edges, all_nodes, node_info_map)
        - adjacency_list: {from_key: [(to_key, relationship), ...]}
        - reverse_adjacency: {to_key: [(from_key, relationship), ...]}
        - node_info: {key: node_info_dict}
        - all_nodes: set of all node keys
    """
    adjacency = defaultdict(list)  # from_key -> [(to_key, relationship)]
    reverse_adjacency = defaultdict(list)  # to_key -> [(from_key, relationship)]
    node_info = {}
    all_nodes = set()
    
    for path in paths:
        from_node = path.get("from", {})
        to_node = path.get("to", {})
        relationship = path.get("relationship", "CONNECTED_TO")
        
        from_info = get_node_info(from_node)
        to_info = get_node_info(to_node)
        
        from_key = from_info["key"]
        to_key = to_info["key"]
        
        # Store adjacency
        adjacency[from_key].append((to_key, relationship))
        reverse_adjacency[to_key].append((from_key, relationship))
        
        # Store node info
        node_info[from_key] = from_info
        node_info[to_key] = to_info
        
        all_nodes.add(from_key)
        all_nodes.add(to_key)
    
    return adjacency, reverse_adjacency, node_info, all_nodes


def find_root_nodes(adjacency: Dict, reverse_adjacency: Dict, all_nodes: Set) -> List[str]:
    """
    Find root nodes (nodes with no incoming edges).
    These are the starting points of the graph.
    
    Args:
        adjacency: Forward adjacency list
        reverse_adjacency: Reverse adjacency list
        all_nodes: Set of all node keys
        
    Returns:
        List of root node keys
    """
    roots = []
    for node in all_nodes:
        if node not in reverse_adjacency or len(reverse_adjacency[node]) == 0:
            roots.append(node)
    return roots


def find_leaf_nodes(adjacency: Dict, all_nodes: Set) -> List[str]:
    """
    Find leaf nodes (nodes with no outgoing edges).
    These are the end points of the graph.
    
    Args:
        adjacency: Forward adjacency list
        all_nodes: Set of all node keys
        
    Returns:
        List of leaf node keys
    """
    leaves = []
    for node in all_nodes:
        if node not in adjacency or len(adjacency[node]) == 0:
            leaves.append(node)
    return leaves


def build_ordered_chains(
    adjacency: Dict, 
    node_info: Dict, 
    root_nodes: List[str]
) -> List[List[Tuple[str, str, str]]]:
    """
    Build ordered chains from root nodes to leaf nodes using DFS.
    
    Args:
        adjacency: Forward adjacency list
        node_info: Node information dictionary
        root_nodes: List of root node keys to start from
        
    Returns:
        List of chains, where each chain is a list of (from_key, relationship, to_key) tuples
    """
    all_chains = []
    
    def dfs_build_chain(current_key: str, current_chain: List, visited: Set):
        """Recursively build chains via DFS."""
        if current_key not in adjacency or len(adjacency[current_key]) == 0:
            # Leaf node - save the chain if it has edges
            if current_chain:
                all_chains.append(current_chain.copy())
            return
        
        for to_key, relationship in adjacency[current_key]:
            if to_key not in visited:
                visited.add(to_key)
                current_chain.append((current_key, relationship, to_key))
                dfs_build_chain(to_key, current_chain, visited)
                current_chain.pop()
                visited.remove(to_key)
    
    for root in root_nodes:
        visited = {root}
        dfs_build_chain(root, [], visited)
    
    return all_chains


def merge_chains_to_tree(
    chains: List[List[Tuple[str, str, str]]], 
    node_info: Dict
) -> List[Dict]:
    """
    Merge multiple chains into a tree structure for cleaner display.
    
    Args:
        chains: List of chains from build_ordered_chains
        node_info: Node information dictionary
        
    Returns:
        List of tree nodes with children
    """
    if not chains:
        return []
    
    # Build a tree structure
    # Find the longest common prefix among chains
    tree_roots = {}
    
    for chain in chains:
        if not chain:
            continue
        
        # Get the root of this chain
        root_key = chain[0][0]
        
        if root_key not in tree_roots:
            tree_roots[root_key] = {
                "node": node_info.get(root_key, {}),
                "children": {}
            }
        
        # Add chain to tree
        current_level = tree_roots[root_key]["children"]
        for from_key, relationship, to_key in chain:
            edge_key = f"{relationship}:{to_key}"
            if edge_key not in current_level:
                current_level[edge_key] = {
                    "relationship": relationship,
                    "node": node_info.get(to_key, {}),
                    "children": {}
                }
            current_level = current_level[edge_key]["children"]
    
    return list(tree_roots.values())


def format_node_short(node_info: Dict) -> str:
    """Format node info into a short readable string."""
    name = node_info.get("name", "Unknown")
    sub_type = node_info.get("resourceSubType", "Unknown")
    provider = node_info.get("provider", "Unknown")
    return f"{name} ({sub_type}, {provider})"


def build_linear_chains(
    adjacency: Dict, 
    node_info: Dict, 
    root_nodes: List[str],
    all_nodes: Set
) -> List[List[Tuple[Dict, str, Dict]]]:
    """
    Build ALL linear chains from root to leaf nodes.
    
    This explores every possible path from root nodes to leaf nodes,
    capturing the complete flow through the graph.
    
    Returns list of chains where each chain is [(from_info, relationship, to_info), ...]
    """
    all_chains = []
    
    def dfs_all_paths(current_key: str, current_chain: List, visited: Set):
        """Build all paths via DFS - explores every branch."""
        if current_key not in adjacency or len(adjacency[current_key]) == 0:
            # Leaf node - save chain if it has content
            if current_chain:
                all_chains.append(current_chain.copy())
            return
        
        for to_key, relationship in adjacency[current_key]:
            # Only prevent cycles within current path (not globally)
            if to_key not in visited:
                visited.add(to_key)
                
                from_info = node_info.get(current_key, {})
                to_info = node_info.get(to_key, {})
                current_chain.append((from_info, relationship, to_info))
                
                dfs_all_paths(to_key, current_chain, visited)
                
                current_chain.pop()
                visited.remove(to_key)
    
    for root in root_nodes:
        visited = {root}
        dfs_all_paths(root, [], visited)
    
    # Sort chains by length (longest first) to show most complete paths first
    all_chains.sort(key=lambda x: len(x), reverse=True)
    
    return all_chains


def deduplicate_chains(chains: List[List[Tuple[Dict, str, Dict]]]) -> List[List[Tuple[Dict, str, Dict]]]:
    """
    Remove chains that are subsets of longer chains.
    
    If chain A is completely contained within chain B, remove chain A.
    """
    if not chains:
        return []
    
    # Convert chains to comparable format (tuple of node keys)
    def chain_to_keys(chain):
        if not chain:
            return ()
        keys = [chain[0][0].get("key", "")]  # First node
        for _, _, to_info in chain:
            keys.append(to_info.get("key", ""))
        return tuple(keys)
    
    chain_keys = [(chain, chain_to_keys(chain)) for chain in chains]
    
    # Keep chains that are not subsets of any other chain
    result = []
    seen_exact: Set[Tuple[str, ...]] = set()
    for i, (chain, keys) in enumerate(chain_keys):
        # Two edges connecting the exact same nodes via different relationship
        # names (e.g. RUNS/SERVED_BY, FRAGMENT/FRAGMENT_OF) produce chains with
        # identical node-key sequences; keep only the first one seen.
        if keys in seen_exact:
            continue

        is_subset = False
        for j, (other_chain, other_keys) in enumerate(chain_keys):
            if i != j and len(other_keys) > len(keys):
                # Check if keys is a contiguous subset of other_keys
                keys_str = ">".join(keys)
                other_keys_str = ">".join(other_keys)
                if keys_str in other_keys_str:
                    is_subset = True
                    break
        if not is_subset:
            seen_exact.add(keys)
            result.append(chain)
    
    return result


def format_chain_as_flow(chain: List[Tuple[Dict, str, Dict]], chain_num: int) -> List[str]:
    """
    Format a single chain as a simple numbered flow.
    
    Example output:
    Flow 1:
      Step 1: acme-cart (Github-Repository, GitHub)
        | IS_PARENT_OF
        v
      Step 2: acme-cart/gcp (Github-Branch, GitHub)
        | SYNCS
        v
      Step 3: acme-cart (Application, ArgoCD)
    """
    if not chain:
        return []
    
    lines = []
    lines.append(f"  Flow {chain_num}:")
    
    step = 1
    for i, (from_info, relationship, to_info) in enumerate(chain):
        if i == 0:
            # First step - show the source node
            lines.append(f"    Node {step}: {format_node_short(from_info)}")
            step += 1
        
        # Show relationship arrow
        lines.append(f"      | {relationship}")
        lines.append(f"      v")
        
        # Show destination node
        lines.append(f"    Node {step}: {format_node_short(to_info)}")
        step += 1
    
    return lines


def format_chains_as_summary(chains: List[List[Tuple[Dict, str, Dict]]]) -> List[str]:
    """
    Format chains as a compact summary showing end-to-end flows.
    
    Example:
    Complete Flows:
    - Github-Repository → Github-Branch → Application → Deployment → ReplicaSet → Pod → Container
    - Service → Endpoints → Pod → Container
    """
    if not chains:
        return []
    
    lines = []
    lines.append("  End-to-End Flows:")
    
    seen_flows = set()  # Deduplicate identical flows
    
    for chain in chains:
        if not chain:
            continue
        
        # Build flow string
        flow_parts = []
        
        # Add first node
        first_from = chain[0][0]
        flow_parts.append(first_from.get("resourceSubType", "Unknown"))
        
        # Add subsequent nodes
        for from_info, relationship, to_info in chain:
            to_type = to_info.get("resourceSubType", "Unknown")
            flow_parts.append(to_type)
        
        flow_str = " → ".join(flow_parts)
        
        if flow_str not in seen_flows:
            seen_flows.add(flow_str)
            lines.append(f"    • {flow_str}")
    
    return lines


def format_ordered_path_graph(paths: List[Dict[str, Any]], path_type: str) -> str:
    """
    Build and format an ordered end-to-end graph from paths data.
    
    Produces LLM-friendly output showing:
    1. A summary of end-to-end flows (resource types)
    2. Detailed step-by-step flows with resource names
    
    Args:
        paths: List of path objects with from, to, relationship
        path_type: Name of the path type (delivery, network, request)
        
    Returns:
        Formatted graph text showing ordered flow
    """
    if not paths:
        return f"  No {path_type} relationships found.\n"
    
    # Build graph structure
    adjacency, reverse_adjacency, node_info, all_nodes = build_graph_from_paths(paths)
    
    # Find root nodes (no incoming edges)
    root_nodes = find_root_nodes(adjacency, reverse_adjacency, all_nodes)
    
    # If no clear roots, use all nodes as potential roots
    if not root_nodes:
        root_nodes = list(all_nodes)
    
    # Build all linear chains from roots to leaves
    chains = build_linear_chains(adjacency, node_info, root_nodes, all_nodes)
    
    # Remove shorter chains that are subsets of longer chains
    chains = deduplicate_chains(chains)
    
    if not chains:
        return f"  No complete {path_type} flows found.\n"
    
    lines = []
    
    # Add summary of flows (compact view)
    summary_lines = format_chains_as_summary(chains)
    lines.extend(summary_lines)
    lines.append("")
    
    # Add detailed flows
    lines.append("  Detailed Resource Flow:")
    for i, chain in enumerate(chains, 1):
        chain_lines = format_chain_as_flow(chain, i)
        lines.extend(chain_lines)
        if i < len(chains):
            lines.append("")  # Separator between flows
    
    return "\n".join(lines)


def format_resource_graph(
    resource_id: str, 
    resource_data: Dict[str, Any]
) -> str:
    """
    Format the complete graph for a single resource across all path types.
    
    Args:
        resource_id: The resource ID
        resource_data: Dictionary with path types (delivery, network, request)
        
    Returns:
        Formatted graph text for LLM consumption
    """
    lines = []
    lines.append(f"### Resource Graph for Resource ID: {resource_id}")
    lines.append("")
    
    # Define path types and their descriptions
    path_type_descriptions = {
        "delivery": "Delivery Path (CI/CD, source code, deployments)",
        "request": "Request Path (user, services, workloads)",
        "network": "Network Path (routes, security groups, endpoints, firewalls)"
    }
    
    # Process each path type
    for path_type, description in path_type_descriptions.items():
        path_data = resource_data.get(path_type, {})
        paths = path_data.get("paths", [])
        
        if "error" in path_data:
            lines.append(f"**{description}:**")
            lines.append(f"  Error fetching {path_type} graph: {path_data['error']}")
            lines.append("")
            continue
        
        lines.append(f"**{description}:**")
        
        if paths:
            graph_text = format_ordered_path_graph(paths, path_type)
            lines.append(graph_text)
        else:
            lines.append(f"  No {path_type} relationships found.")
        
        lines.append("")
    
    return "\n".join(lines)


def format_all_resource_graphs(
    resource_neighbours: Dict[str, Dict[str, Any]]
) -> str:
    """
    Format graphs for all resources into a single text block for LLM.
    
    Args:
        resource_neighbours: Dictionary mapping resource_id to path type data
            Structure: {
                "84235923": {
                    "delivery": {"paths": [...]},
                    "network": {"paths": [...]},
                    "request": {"paths": [...]}
                }
            }
        
    Returns:
        Formatted text containing all resource graphs
    """
    if not resource_neighbours:
        return "No resource graph information available.\n"
    
    lines = []
    lines.append("## Resource Relationship Graphs")
    lines.append("")

    for resource_id, resource_data in resource_neighbours.items():
        # Skip error entries at the resource level
        if isinstance(resource_data, dict) and "error" in resource_data and len(resource_data) == 1:
            lines.append(f"### Resource ID: {resource_id}")
            lines.append(f"  Error: {resource_data['error']}")
            lines.append("")
            continue
        
        graph_text = format_resource_graph(resource_id, resource_data)
        lines.append(graph_text)
        lines.append("---")
        lines.append("")
    
    return "\n".join(lines)


def extract_graph_summary(resource_neighbours: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract a summary of the graph for quick reference.
    
    Args:
        resource_neighbours: Dictionary mapping resource_id to path type data
        
    Returns:
        Summary dictionary with counts and key resources
    """
    summary = {
        "total_resources": 0,
        "total_relationships": 0,
        "path_type_counts": {
            "delivery": 0,
            "network": 0,
            "request": 0
        },
        "unique_providers": set(),
        "resource_types": set(),
        "key_resources": []
    }
    
    if not resource_neighbours:
        return summary
    
    summary["total_resources"] = len(resource_neighbours)
    
    for resource_id, resource_data in resource_neighbours.items():
        if isinstance(resource_data, dict) and "error" in resource_data and len(resource_data) == 1:
            continue
        
        for path_type in ["delivery", "network", "request"]:
            path_data = resource_data.get(path_type, {})
            paths = path_data.get("paths", [])
            
            summary["path_type_counts"][path_type] += len(paths)
            summary["total_relationships"] += len(paths)
            
            for path in paths:
                for node_key in ["from", "to"]:
                    node = path.get(node_key, {})
                    labels = node.get("labels", [])
                    provider = extract_provider_from_labels(labels)
                    resource_type = node.get("resourceSubType", "")
                    
                    if provider != "Unknown":
                        summary["unique_providers"].add(provider)
                    if resource_type:
                        summary["resource_types"].add(resource_type)
    
    # Convert sets to lists for JSON serialization
    summary["unique_providers"] = list(summary["unique_providers"])
    summary["resource_types"] = list(summary["resource_types"])
    
    return summary


def format_graph_for_llm_prompt(
    resource_neighbours: Dict[str, Dict[str, Any]],
    include_summary: bool = True
) -> str:
    """
    Main function to format resource neighbour graphs for LLM prompt.
    
    This formats the graph data into a human-readable text that provides
    context about resource relationships for incident analysis.
    
    Args:
        resource_neighbours: Dictionary mapping resource_id to path type data
        include_summary: Whether to include a summary section
        
    Returns:
        Formatted text ready to be included in LLM prompt
    """
    if not resource_neighbours:
        return ""
    
    lines = []
    
    # Add summary if requested
    if include_summary:
        summary = extract_graph_summary(resource_neighbours)
        
        # if summary["total_relationships"] > 0:
            # lines.append("## Resource Topology Summary")
            # lines.append("")
            # lines.append(f"- **Resources analyzed:** {summary['total_resources']}")
            # lines.append(f"- **Total relationships:** {summary['total_relationships']}")
            # lines.append(f"  - Delivery paths: {summary['path_type_counts']['delivery']}")
            # lines.append(f"  - Network paths: {summary['path_type_counts']['network']}")
            # lines.append(f"  - Request paths: {summary['path_type_counts']['request']}")
            
            # if summary["unique_providers"]:
            #     lines.append(f"- **Providers involved:** {', '.join(sorted(summary['unique_providers']))}")
            
            # if summary["resource_types"]:
            #     lines.append(f"- **Resource types:** {', '.join(sorted(summary['resource_types']))}")
            
            # lines.append("")
    
    # Add detailed graph
    graph_text = format_all_resource_graphs(resource_neighbours)
    lines.append(graph_text)
    
    return "\n".join(lines)


# ==============================================================================
# LANE-LESS GRAPH FORMATTING
# ==============================================================================
def format_lane_less_resource_graph(resource_id: str, resource_data: Dict[str, Any]) -> str:
    """
    Format a single resource's lane-less graph (a flat "paths" list) using the
    same end-to-end flow rendering as the lane-based graphs.

    Args:
        resource_id: The resource ID the paths were fetched for
        resource_data: Dict with a flat "paths" list (no lane key), e.g.
            {"paths": [{"from": ..., "to": ..., "relationship": ...}, ...]}

    Returns:
        Formatted text for LLM consumption
    """
    lines = [f"### Resource Graph for Resource ID: {resource_id}", ""]

    if not isinstance(resource_data, dict):
        lines.append("  No relationships found.")
        return "\n".join(lines)

    if "error" in resource_data:
        lines.append(f"  Error fetching resource graph: {resource_data['error']}")
        return "\n".join(lines)

    paths = resource_data.get("paths", [])
    if not paths:
        lines.append("  No relationships found.")
        return "\n".join(lines)

    lines.append(format_ordered_path_graph(paths, "resource"))
    return "\n".join(lines)


def format_all_lane_less_resource_graphs(
    resource_neighbours: Dict[str, Dict[str, Any]]
) -> str:
    """
    Format lane-less graphs for all resources into a single text block for
    LLM consumption.

    Args:
        resource_neighbours: Dict mapping resource_id -> {"paths": [...]}

    Returns:
        Formatted text containing all resources' graphs
    """
    if not resource_neighbours:
        return "No resource graph information available.\n"

    lines = ["## Resource Relationship Graphs", ""]

    for resource_id, resource_data in resource_neighbours.items():
        if isinstance(resource_data, dict) and "error" in resource_data and len(resource_data) == 1:
            lines.append(f"### Resource ID: {resource_id}")
            lines.append(f"  Error: {resource_data['error']}")
            lines.append("")
            continue

        lines.append(format_lane_less_resource_graph(resource_id, resource_data))
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def format_lane_less_graph_for_llm_prompt(
    resource_neighbours: Dict[str, Dict[str, Any]],
) -> str:
    """
    Main entry point to format a lane-less resource graph for an LLM prompt.

    Use this instead of `format_graph_for_llm_prompt` for data shaped as a
    flat `{"paths": [...]}` per resource (e.g. what
    `fetch_immediate_resource_neighbours_for_incident` returns), rather than
    the lane-keyed `{"delivery": {...}, "network": {...}, "request": {...}}`
    structure.

    Args:
        resource_neighbours: Dict mapping resource_id -> {"paths": [...]}

    Returns:
        Formatted text ready to be included in LLM prompt
    """
    if not resource_neighbours:
        return ""

    return format_all_lane_less_resource_graphs(resource_neighbours)


# Example usage and testing
if __name__ == "__main__":
    import json
    
    # Load sample data
    sample_file = "resource_neighbours_84235923.json"
    
    try:
        with open(sample_file, "r") as f:
            sample_data = json.load(f)
        
        # Format for LLM
        formatted_output = format_graph_for_llm_prompt(sample_data, include_summary=True)
        print(formatted_output)
        
    except FileNotFoundError:
        print(f"Sample file {sample_file} not found. Creating example...")
        
        # Example data structure
        example_data = {
            "84235923": {
                "deployment": {
                    "paths": [
                        {
                            "from": {
                                "labels": ["Node", "RESOURCE", "KUBERNETES", "POD"],
                                "key": "RESOURCE|84235923",
                                "name": "example-pod",
                                "resourceSubType": "Pod",
                                "resourceType": "Workload"
                            },
                            "to": {
                                "labels": ["Node", "RESOURCE", "KUBERNETES", "CONTAINER"],
                                "key": "RESOURCE|84235924",
                                "name": "example-container",
                                "resourceSubType": "Container",
                                "resourceType": "Workload"
                            },
                            "relationship": "CONTAINS"
                        }
                    ]
                },
                "network": {"paths": []},
                "dependency": {"paths": []}
            }
        }
        
        formatted_output = format_graph_for_llm_prompt(example_data)
        print(formatted_output)
