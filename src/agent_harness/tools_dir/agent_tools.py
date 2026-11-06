"""Agent tools module."""

from __future__ import annotations

import ast
import operator
import logging
from typing import Annotated, Any, Dict, List, Optional

from pydantic import Field

import httpx

from agent_harness.errors import ToolError
from agent_harness.tools import ToolAnnotations, tool
from client.api.manifest_api_integration import execute_manifest_api

# Tool Utils
from agent_harness.tools_dir.tool_utils.resource_metadata_format import format_resource_metadata_from_config
from agent_harness.tools_dir.tool_utils.resource_neighbors_format import format_lane_less_resource_graph

logger = logging.getLogger(__name__)


@tool(annotations=ToolAnnotations(read_only=True, idempotent=True), concurrency_safe=True)
def fetch_resource_metadata(
    resource_ids: List[int],
) -> Dict[str, Any]:
    """
    Fetch resource metadata.
    
    Args:
        resource_ids: List of int resource IDs
        
    Returns:
        Dictionary containing the fetched resource metadata.
    """
    # Build query parameters
    json_data = {
        "resourceIds": resource_ids
    }
    
    logger.info(f"TOOL: Fetching resource metadata for resource IDs: {resource_ids}")
    
    # Define endpoint patterns - both internal and client endpoints available
    client_endpoint =  "client/resource/rca/metadata"
    internal_endpoint = f"org/1/resource/rca/metadata"
    
    # Execute API call using generic executor with internal mode forced
    data, success = execute_manifest_api(
        client_endpoint=client_endpoint,
        internal_endpoint=internal_endpoint,
        org_key="Development",
        org_id="1",
        subscription_id="4",
        method="POST",
        json_data=json_data,
        timeout=30,
        use_client_api=False
    )
    
    # Handle results
    if not success:
        logger.error(f"Failed to fetch resource metadata for RCA: {data.get('error', 'Unknown error')}")
        return {"error": data.get("error", "Unknown error")}
    
    logger.info(f"Successfully fetched resource metadata for resource IDs: {resource_ids}")

    # Format resource metadata
    for resource_id, resource_metadata in data.items():
        data[resource_id] = format_resource_metadata_from_config(resource_metadata=resource_metadata, resource_id=resource_id, indent=" ")

    logger.info(f"Formatted resource metadata for resource IDs: {resource_ids}")
    
    return data

@tool(annotations=ToolAnnotations(read_only=True, idempotent=True), concurrency_safe=True)
def fetch_resource_neighbors(
    resource_id: int,
    in_depth: Optional[int],
    out_depth: Optional[int],
) -> Dict[str, Any]:
    """
    Fetch resource neighbors.
    Specify the depth of incoming and outgoing edges if required until a certain level.
    Do not specify the depth if you want to fetch all neighbors.
    
    Args:
        resource_id: Resource ID for which to fetch neighbors
        in_depth: Depth for incoming edges
        out_depth: Depth for outgoing edges
        
    Returns:
        Dictionary mapping resource_id to its neighbors.
    """ 
    logger.info(f"TOOL: Fetching resource neighbors for resource ID: {resource_id} with in_depth={in_depth} and out_depth={out_depth}")

    # Build query parameters
    params = {
        "mitId": resource_id,
        "mitIdType": "RESOURCE",
        "parent": str(True).lower(),
        "needSimple": str(True),
        "inDepth": str(in_depth),
        "outDepth": str(out_depth)
    }

    # Define endpoint patterns - both internal and client endpoints available
    client_endpoint =  "client/graph/RESOURCE"
    internal_endpoint = f"org/1/resource/graph"
    
    # Execute API call using generic executor with internal mode forced
    data, success = execute_manifest_api(
        client_endpoint=client_endpoint,
        internal_endpoint=internal_endpoint,
        org_key="Development",
        org_id="1",
        subscription_id="4",
        method="GET",
        params=params,
        timeout=30,
        use_client_api=False
    )

    if not success:
        logger.error(f"Error fetching neighbors for resource {resource_id}")
        return {}, False

    logger.info(f"Successfully fetched neighbors for resource ID: {resource_id}")

    # Format resource neighbors 
    if not data:
        return "No resource graph information available.\n"

    lines = ["## Resource Relationship Graphs", ""]

    lines.append(format_lane_less_resource_graph(resource_id, data))
    lines.append("---")
    lines.append("")
        
    return "\n".join(lines), success