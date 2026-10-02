"""
Resource metadata format configuration and formatting utilities.
"""
import json
import re
from typing import Any, Dict, List, Optional, Set


# Resource metadata format configuration map for providers
resource_metadata_provider_config = {
  "kubernetes": {
    "pod": {
      "important_fields": [
        "kind",
        "metadata.name",
        "metadata.namespace",
        "spec.nodeName",
        "status.phase",
        "status.conditions",
        "status.message",
        "status.reason",
        "status.containerStatuses"
      ],
      "list_item_fields": {
        "status.containerStatuses": [
          "name",
          "image",
          "ready",
          "restartCount",
          "state",
          "lastState.terminated.reason",
          "lastState.terminated.exitCode",
          "resources"
        ],
        "status.conditions": [
            "type",
            "reason",
            "message"
        ]
      }
    },
    "replicaset": {
      "important_fields": [
        "kind",
        "metadata.name",
        "metadata.namespace",
        "spec.replicas",
        "spec.template.spec.containers",
        "spec.template.spec.imagePullSecrets",
        "spec.template.spec.nodeSelector",
        "status"
      ],
      "keys_to_exclude": ['spec.template.metadata'],
      "list_item_fields": {
        "spec.template.spec.containers": [
          "name",
          "image",
          "imagePullPolicy",
          "ports",
          "env",
          "resources"
        ]
      }
    },
    "persistentvolume": {
      "important_fields": [
        "apiVersion",
        "kind",
        "metadata.name",
        "metadata.labels",
        "spec.capacity",
        "spec.accessModes",
        "spec.storageClassName",
        "spec.persistentVolumeReclaimPolicy",
        "spec.claimRef",
        "spec.csi.driver",
        "spec.csi.volumeHandle",
        "spec.csi.volumeAttributes",
        "spec.nodeAffinity",
        "status.phase",
      ]
    },
    "configmap": {
      "important_fields": [
        "apiVersion",
        "kind",
        "metadata.name",
        "metadata.namespace",
        "metadata.labels",
        "metadata.annotations",
      ]
    },
    "secret": {
      "important_fields": [
        "apiVersion",
        "kind",
        "metadata.name",
        "metadata.namespace",
        "metadata.labels",
        "metadata.annotations",
        "type",
      ]
    },
    "serviceaccount": {
      "important_fields": [
        "apiVersion",
        "kind",
        "metadata.name",
        "metadata.namespace",
      ]
    },
    "node": {
      "important_fields": [
        "apiVersion",
        "kind",
        "status.conditions"
      ]
    },
    "service": {
      "important_fields": [
        "kind",
        "metadata.name",
        "metadata.namespace",
        "spec.ports",
        "spec.type"
      ]
    }
  },
  "aws": {
      "AWS::EC2::SecurityGroup": {
          "important_fields": [
              "security_group.SecurityGroups",
              "securityGroupRule"
          ]
      }
  },
 "gh": {
     "github-repository": {
         "important_fields": [
             "name",

         ]
     },
     "github-branch": {
            "important_fields": [
                "branch_data.name",
                "branch_data.latest_file_changes",
            ],
            "keys_to_exclude": [
                "branch_data.latest_file_changes.raw_url",
                "branch_data.latest_file_changes.blob_url",
            ]
     },
     "github-workflow-run": {
         "important_fields": [
             "workflow_run.name",
             "workflow_run.event",
             "workflow_run.status",
             "workflow_run.html_url",
             "workflow_run.conclusion",
             "workflow_run.repository.name",
             "workflow_run.run_number",
             "workflow_run.workflow_id",
             "workflow_run.head_branch",
             "workflow_run.head_commit.id",
             "workflow_run.head_commit.message",
             "workflow_run.head_commit.author",
         ]
     },
     "github-workflow": {
         "important_fields": [
             "name",
             "workflow_id",
             "latest_workflow_run.url",
             "latest_workflow_run.state",
             "latest_workflow_run.commit.id",
             "latest_workflow_run.commit.author",
             "latest_workflow_run.commit.message",
             "latest_workflow_run.number",
         ]
     }
  },
  "jenkins" : {
      "jenkins-stage": {
          "important_fields": [
              "id",
              "name",
              "status",
              "stageFlowNodes"
          ],
          "list_item_fields": {
              "stageFlowNodes": [
                  "id",
                  "name",
                  "status"
              ]
          }
      },
      "jenkins-build": {
          "important_fields": [
              "id",
              "url",
              "result",
              "changeSets",
              "changeSets.items"
          ],
          "list_item_fields": {
              "changeSets": [
                  "kind",
                  "name",
                  "status"
              ],
              "changeSets.items": [
                  "commitId",
                  "msg",
                  "paths",
                  "comment",
                  "affectedPaths"
              ]
          }
      },
      "jenkins-pipeline": {
          "important_fields": [
              "job.url",
              "job.name",
              "job.lastBuild",
              "job.firstBuild",
              "job.healthReport",
              "job.lastFailedBuild",
              "job.lastStableBuild",
              "job.lastUnstableBuild",
              "job.lastSuccessfulBuild",
              "job.lastCompletedBuild",
              "job.lastUnsuccessfulBuild"
          ],
          "list_item_fields": {
              "stages": [
                  "id",
                  "name",
                  "status"
              ]
          }
      }
  },
  "argocd": {
      "application": {
          "important_fields": [
              "name",
              "project",
              "labels.cluster",
              "labels.environment",
              "images",
              "source",
              "destination",
              "sync_status",
              "health_status",
              "sync_policy",
              "operation_phase",
              "operation_message",
              "target_revision",
              "deployed_revision",
              "initiated_by_user",
              "resources",
          ],
          "keys_to_exclude": [
              "resources.metadata",
              "resources.spec",
              "resources.pod_status",
              "annotations"
          ]
      }
  },
  "host": {
      "process": {
          "important_fields": [
              "pid",
              "name",
              "ppid",
              "cmdline",
              "app_name",
              "systemd_unit"
          ]
      },
      "service": {
          "important_fields": [
              "name",
              "state",
              "enabled",
              "fragmented_path"
          ]
      },
      "host": {
          "important_fields": [
              "hostname",
              "platform.provider_key",
              "cpu_cores",
              "cpu_model",
              "memory_total_mb",
              "disk_attached_gb",
              "memory_attached_mb"
          ]
      }
  }
}


# Global timestamp keywords for excluding timestamp fields
TIMESTAMP_KEYWORDS = [
    "time", "timestamp", "date", "at", "creationTimestamp", "deletionTimestamp",
    "lastTransitionTime", "lastProbeTime", "lastHeartbeatTime",
    "startedAt", "finishedAt", "createdAt", "updatedAt"
]

# Global fields to always exclude
GLOBAL_EXCLUDE_FIELDS = [
    "managedFields", "selfLink", "uid", "resourceVersion",
    "generation", "finalizers"
]


def is_timestamp_field(key: str, value: Any) -> bool:
    """
    Check if a field is a timestamp to exclude from metadata formatting.
    
    Args:
        key: Field name
        value: Field value
        
    Returns:
        True if field is a timestamp, False otherwise
    """
    key_lower = key.lower()
    
    # Check if key matches timestamp keywords
    for keyword in TIMESTAMP_KEYWORDS:
        keyword_lower = keyword.lower()
        if key_lower == keyword_lower:  # Exact match
            return True
        if key_lower.endswith(keyword_lower):  # Ends with keyword (e.g., createdAt, created_at)
            return True
        if keyword_lower in ['time', 'timestamp', 'date'] and keyword_lower in key_lower:
            # For common timestamp words, allow substring match
            return True
    
    # Check value format (ISO 8601 timestamps)
    if isinstance(value, str):
        # Simple check for ISO format: YYYY-MM-DDTHH:MM:SSZ or similar
        if 'T' in value and ('Z' in value or '+' in value or value.count(':') >= 2):
            return True
    
    return False


def format_key(key: str) -> str:
    """
    Format a metadata key for display (convert camelCase to Title Case).
    
    Args:
        key: Key name
        
    Returns:
        Formatted key name
    """
    # Add space before capital letters
    formatted = re.sub(r'([A-Z])', r' \1', key)
    # Capitalize first letter
    formatted = formatted.strip().title()
    return formatted


def _get_key(current: Any, key: str) -> Any:
    """
    Get `key` off a dict, or map+flatten `key` across a list of dicts.

    This is what lets a dotted path walk through a list-of-objects without an
    explicit index, e.g. "changeSets.items" where `changeSets` is itself a
    list of `{items: [...]}` objects - each object's "items" list is collected
    and flattened into one combined list.
    """
    if isinstance(current, dict):
        return current.get(key)

    if isinstance(current, list):
        collected: List[Any] = []
        for element in current:
            if not isinstance(element, dict) or key not in element:
                continue
            value = element[key]
            if isinstance(value, list):
                collected.extend(value)
            else:
                collected.append(value)
        return collected

    return None


def get_nested_value(data: Dict, path: str) -> Any:
    """
    Get a nested value from a dictionary using dot notation and array indexing.
    Supports paths like "metadata.name", "spec.containers[0].image", etc.
    Also supports paths that cross a list without an index (e.g.
    "changeSets.items") by mapping the remaining key across every element and
    flattening one level - see `_get_key`.
    
    Args:
        data: Source dictionary
        path: Dot-separated path (can include array indices like [0])
        
    Returns:
        Value at the path, or None if not found
    """
    try:
        # Split by dots but preserve array indices
        parts = path.split('.')
        current = data
        
        for part in parts:
            if current is None:
                return None
            
            # Check for array index notation like "containers[0]"
            if '[' in part and ']' in part:
                key = part[:part.index('[')]
                index_str = part[part.index('[') + 1:part.index(']')]
                
                # Get the array (walks through dicts or lists-of-dicts)
                current = _get_key(current, key)
                if current is None:
                    return None
                
                # Get the indexed item
                if isinstance(current, list):
                    index = int(index_str)
                    if 0 <= index < len(current):
                        current = current[index]
                    else:
                        return None
                else:
                    return None
            else:
                # Regular key access (walks through dicts or lists-of-dicts)
                current = _get_key(current, part)
        
        return current
    except Exception:
        return None


def should_exclude_path(path: str, exclude_patterns: List[str]) -> bool:
    """
    Check if a path should be excluded based on exclude patterns.
    Handles array indices by normalizing them (e.g., "status.containerStatuses[0].volumeMounts" 
    matches pattern "status.containerStatuses.volumeMounts").
    
    Args:
        path: Current field path (e.g., "status.containerStatuses[0].volumeMounts")
        exclude_patterns: List of patterns to exclude
        
    Returns:
        True if path should be excluded, False otherwise
    """
    # Normalize path by removing array indices for matching
    normalized_path = re.sub(r'\[\d+\]', '', path)
    
    for pattern in exclude_patterns:
        # Check if normalized path matches the pattern
        if normalized_path == pattern or normalized_path.startswith(pattern + '.'):
            return True
        # Also check exact match with original path (in case pattern includes indices)
        if path == pattern or path.startswith(pattern + '.'):
            return True
    
    return False


def project_item_fields(item: Any, allowed_fields: List[str]) -> Any:
    """Keep only the given dot-paths from a single list item, preserving nesting."""
    if not isinstance(item, dict):
        return item

    projected: Dict[str, Any] = {}
    for field_path in allowed_fields:
        value = get_nested_value(item, field_path)
        if value is None:
            continue

        parts = field_path.split('.')
        cursor = projected
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value

    return projected


def project_list_fields(items: List[Any], allowed_fields: List[str]) -> List[Any]:
    """Apply project_item_fields to every dict item in a list."""
    return [project_item_fields(item, allowed_fields) for item in items]


def format_value_flat(value: Any, indent: str, current_path: str, exclude_patterns: List[str], depth: int = 0, max_depth: int = 10) -> str:
    """
    Format a value in a flat, human-readable format (no JSON).
    Handles nested dicts, arrays, and primitives recursively.
    
    Args:
        value: Value to format
        indent: Current indentation level
        current_path: Current field path for exclusion checking
        exclude_patterns: List of path patterns to exclude
        depth: Current recursion depth
        max_depth: Maximum recursion depth
        
    Returns:
        Formatted string
    """
    if depth > max_depth:
        return ""
    
    result = ""
    
    if value is None or value == "" or (isinstance(value, (list, dict)) and not value):
        return ""
    
    if isinstance(value, dict):
        for key, val in value.items():
            # Skip globally excluded fields
            if key in GLOBAL_EXCLUDE_FIELDS:
                continue
            
            # Skip timestamp fields
            if is_timestamp_field(key, val):
                continue
            
            # Build nested path
            nested_path = f"{current_path}.{key}" if current_path else key
            
            # Check if this path should be excluded
            if should_exclude_path(nested_path, exclude_patterns):
                continue
            
            # Skip empty values
            if val is None or val == "" or (isinstance(val, (list, dict)) and not val):
                continue
            
            formatted_key = format_key(key)
            
            if isinstance(val, dict):
                result += f"{indent}{formatted_key}:\n"
                result += format_value_flat(val, indent + "  ", nested_path, exclude_patterns, depth + 1, max_depth)
            elif isinstance(val, list):
                result += f"{indent}{formatted_key}:\n"
                result += format_value_flat(val, indent + "  ", nested_path, exclude_patterns, depth + 1, max_depth)
            else:
                result += f"{indent}{formatted_key}: {val}\n"
    
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            # Build nested path with array index
            nested_path = f"{current_path}[{idx}]"
            
            if isinstance(item, dict):
                # Check if this is a condition or status item
                item_type = item.get('type', item.get('name', f'Item {idx + 1}'))
                result += f"{indent}- {item_type}:\n"
                result += format_value_flat(item, indent + "  ", nested_path, exclude_patterns, depth + 1, max_depth)
            elif isinstance(item, (list, dict)):
                result += f"{indent}- Item {idx + 1}:\n"
                result += format_value_flat(item, indent + "  ", nested_path, exclude_patterns, depth + 1, max_depth)
            else:
                result += f"{indent}- {item}\n"
    
    else:
        result += f"{indent}{value}\n"
    
    return result


def format_resource_metadata_from_config(
    resource_metadata: Dict,
    resource_id: int,
    indent: str,
) -> str:
    """
    Format resource metadata based on the config map.
    This is a scalable approach that uses the config map to determine which fields to include.
    
    Args:
        resource_metadata: Dictionary containing resource metadata (keyed by resource_id)
        resource_id: The resource ID to format metadata for
        indent: Base indentation string
        
    Returns:
        Formatted string with resource metadata
    """
    if not resource_metadata or str(resource_id) not in resource_metadata:
        return ""

    metadata = resource_metadata[str(resource_id)]
    if not metadata:
        return ""
    
    # Get provider from metadata (always at top level)
    provider = (metadata.get('provider_key') or '').lower()
    
    # Get resource_sub_type from metadata (always at top level)
    resource_sub_type = (metadata.get('resource_sub_type') or '').lower()

    # Check if actual data is nested inside __data (API response structure)
    # If so, use __data for field lookups while keeping provider_key and resource_sub_type from top level
    data_source = metadata.get('__data', metadata)
    
    # Get provider config
    provider_config = resource_metadata_provider_config.get(provider, {})
    
    # Get resource type candidates from metadata
    resource_kind = (data_source.get('kind') or '').lower()
    # Also check resource_sub_type in data_source if not found at top level
    if not resource_sub_type:
        resource_sub_type = (data_source.get('resource_sub_type') or '').lower()

    resource_config = {}
    resource_type = None
    
    # If provider config exists, try to find resource config within it
    if provider_config:
        # Try resource_sub_type first, then kind
        if resource_sub_type and resource_sub_type in provider_config:
            resource_config = provider_config.get(resource_sub_type, {})
            resource_type = resource_sub_type
        elif resource_kind and resource_kind in provider_config:
            resource_config = provider_config.get(resource_kind, {})
            resource_type = resource_kind
        else:
            # Provider is known but doesn't define this subtype/kind, dump everything instead.
            return _format_all_fields(data_source, resource_id, indent)
    
    # Provider unknown - search across all providers using kind or resource_sub_type
    if not resource_config:
        for other_provider, other_provider_config in resource_metadata_provider_config.items():
            if not other_provider_config:
                continue
            # Try resource_sub_type first, then kind
            if resource_sub_type and resource_sub_type in other_provider_config:
                resource_config = other_provider_config.get(resource_sub_type, {})
                resource_type = resource_sub_type
                break
            elif resource_kind and resource_kind in other_provider_config:
                resource_config = other_provider_config.get(resource_kind, {})
                resource_type = resource_kind
                break
    
    # If still no config found, fallback to formatting all fields
    if not resource_config:
        return _format_all_fields(data_source, resource_id, indent)
    
    important_fields = resource_config.get('important_fields', [])
    keys_to_exclude = resource_config.get('keys_to_exclude', [])
    list_item_fields = resource_config.get('list_item_fields', {})
    
    if not important_fields:
        return ""
    
    # Build formatted output
    content = ""
    
    # Process each important field
    for field_path in important_fields:
        # Get the value at this path from data_source (handles __data nesting)
        value = get_nested_value(data_source, field_path)
        
        if value is None:
            continue
        
        # Skip empty values
        if value == "" or (isinstance(value, (list, dict)) and not value):
            continue
        
        # Project each item down to an allowlist of fields, if configured
        if isinstance(value, list) and field_path in list_item_fields:
            value = project_list_fields(value, list_item_fields[field_path])
        
        # Get the display name (last part of the path)
        display_name = field_path.split('.')[-1].replace('[0]', '')
        formatted_name = format_key(display_name)
        
        # Format the value
        if isinstance(value, (dict, list)):
            content += f"{indent}  {formatted_name}:\n"
            content += format_value_flat(value, indent + "    ", field_path, keys_to_exclude, depth=0)
        else:
            # Skip timestamp fields
            if not is_timestamp_field(display_name, value):
                content += f"{indent}  {formatted_name}: {value}\n"
    
    return content


def _format_all_fields(metadata: Dict, resource_id: int, indent: str) -> str:
    """
    Fallback function to format all fields when no specific config exists.
    
    Args:
        metadata: Resource metadata dictionary
        resource_id: Resource ID
        indent: Base indentation
        
    Returns:
        Formatted string
    """
    content = f"{indent}Resource Metadata for Resource ID {resource_id}:\n"
    content += format_value_flat(metadata, indent + "  ", "", [], depth=0)
    return content


if __name__ == "__main__":
    resource_id = 154765
    with open("metadata.json", "r") as f:
        data = json.load(f)

    f = format_resource_metadata_from_config(
        resource_metadata=data,
        resource_id=resource_id,
        indent=""
    )
    print(f)