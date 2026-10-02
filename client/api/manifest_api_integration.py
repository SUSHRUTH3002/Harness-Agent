"""
Manifest API Integration for service request stamping.

This module provides functionality to fetch data from Manifest API via REST/curl .
"""

import logging
import requests
import time
from typing import Any, Dict, Optional, Tuple, Literal
from datetime import datetime, timedelta
import threading

from client.config import (
    MANIFEST_API_CONFIG
)

logger = logging.getLogger(__name__)

# Token cache with thread safety
_token_cache = {
    "access_token": None,
    "expires_at": None,
    "lock": threading.Lock()
}

# HTTP methods safe to blindly replay on timeout without risking duplicate side effects
IDEMPOTENT_HTTP_METHODS = {"GET", "PUT", "DELETE"}


def generate_access_token(force_refresh: bool = False) -> Optional[str]:
    """
    Generate or retrieve cached access token for Manifest API.
    
    Args:
        force_refresh: Force token regeneration even if cached token is valid
        
    Returns:
        Access token string or None if generation fails
    """
    with _token_cache["lock"]:
        # Check if we have a valid cached token
        if not force_refresh and _token_cache["access_token"] and _token_cache["expires_at"]:
            if datetime.now() < _token_cache["expires_at"]:
                logger.debug("Using cached access token")
                return _token_cache["access_token"]
        
        # Generate new token
        try:
            auth_issuer = MANIFEST_API_CONFIG.get("auth_issuer")
            auth_url = f"{auth_issuer}/protocol/openid-connect/token"
            
            client_id = MANIFEST_API_CONFIG.get("client_id")
            client_secret = MANIFEST_API_CONFIG.get("client_secret")
            grant_type = MANIFEST_API_CONFIG.get("grant_type", "client_credentials")
            username = MANIFEST_API_CONFIG.get("username")
            password = MANIFEST_API_CONFIG.get("password")
            
            if not client_id or not client_secret:
                logger.error("Missing client_id or client_secret in configuration")
                return None
            
            # Prepare form data
            data = {
                "client_id": client_id,
                "grant_type": grant_type,
                "client_secret": client_secret,
                # "username": username,
                # "password": password
            }
            
            headers = {
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            logger.info(f"Generating new access token from {auth_url}")
            
            response = requests.post(
                auth_url,
                headers=headers,
                data=data,
                timeout=MANIFEST_API_CONFIG.get("timeout", 30)
            )
            
            response.raise_for_status()
            token_data = response.json()
            
            access_token = token_data.get("access_token")
            expires_in = token_data.get("expires_in", 300)  # Default 5 minutes
            
            if not access_token:
                logger.error("No access_token in response")
                return None
            
            # Cache token with expiration (subtract 60 seconds for safety margin)
            _token_cache["access_token"] = access_token
            _token_cache["expires_at"] = datetime.now() + timedelta(seconds=expires_in - 60)
            
            logger.info(f"Access token generated successfully, expires in {expires_in} seconds")
            return access_token
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to generate access token: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error generating access token: {e}")
            return None
        

def get_client_api_headers(api_key: str, org_key: str, org_id: str) -> Dict[str, str]:
    """
    Build headers for Manifest Client API requests (static API key mode).
    
    Args:
        api_key: Static API key for client API
        org_key: Organization key (e.g., 'dev')
        org_id: Organization ID
        
    Returns:
        Dictionary of headers for requests
    """
    headers = {
        "Mit-Api-Key": api_key,
        "Mit-Org-Key": org_key,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if org_id:
        headers["Mit-Org-ID"] = str(org_id)
    return headers


def get_internal_api_headers(token: str, org_key: str, org_id: str, subscription_id: str) -> Dict[str, str]:
    """
    Build headers for Manifest Internal API requests (Bearer token mode).
    
    Args:
        token: Access token (JWT Bearer token)
        org_key: Organization key (e.g., 'dev')
        org_id: Organization ID
        subscription_id: Subscription ID
        
    Returns:
        Dictionary of headers for requests
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Mit-Org-Key": org_key,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if org_id:
        headers["Mit-Org-ID"] = str(org_id)
    else:
        headers["Mit-Org-ID"] = str(MANIFEST_API_CONFIG.get("org_id", 1))

    if subscription_id:
        headers["Mit-Subscription-ID"] = str(subscription_id)
    else:
        headers["Mit-Subscription-ID"] = str(MANIFEST_API_CONFIG.get("subscription_id", 4))

    return headers


def execute_manifest_api(
    client_endpoint: str,
    internal_endpoint: str,
    org_key: str,
    org_id: str,
    subscription_id: str,
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "GET",
    params: Optional[Dict[str, Any]] = None,
    json_data: Optional[Dict[str, Any]] = None,
    data: Optional[Any] = None,
    timeout: Optional[int] = None,
    retry_on_auth_failure: bool = True,
    retry_on_timeout: Optional[bool] = None,
    use_client_api: Optional[bool] = None
) -> Tuple[Dict[str, Any], bool]:
    """
    Generic function to execute Manifest API calls with support for both client and internal modes.
    
    This function automatically routes to the appropriate API mode based on the feature flag:
    - Client mode: Uses static API key with /client/ endpoints
    - Internal mode: Uses OAuth2 Bearer token with /org/ endpoints
    
    Args:
        client_endpoint: Endpoint path for client API mode (e.g., 'client/resource/search/tags')
                        Set to empty string '' if not available
        internal_endpoint: Endpoint path for internal API mode (e.g., 'org/{org_id}/resource/tags')
        org_key: Organization key (e.g., 'dev')
        org_id: Organization ID (e.g., '1')
        subscription_id: Subscription ID for the API
        method: HTTP method (GET, POST, PUT, DELETE, PATCH)
        params: Optional query parameters
        json_data: Optional JSON body for the request
        data: Optional form data or other body content
        timeout: Request timeout in seconds (uses config default if not provided)
        retry_on_auth_failure: Retry with fresh token if 401 received (internal mode only)
        retry_on_timeout: Whether to replay the request on timeout (internal mode only). Defaults to None, which retries only for idempotent methods (GET/PUT/DELETE).
        use_client_api: Override config flag to force client or internal mode (True for client, False for internal)
        
    Returns:
        Tuple of (response_data, success)
        response_data is the parsed JSON response or error dict
        success is True if request succeeded, False otherwise

    Note:
        Internal mode retries on request timeout (with backoff) via
        MANIFEST_API_CONFIG['timeout_max_retries']/['timeout_retry_backoff_seconds']; client mode does not.
    """
    timeout = timeout or MANIFEST_API_CONFIG.get("timeout", 30)
    base_url = MANIFEST_API_CONFIG.get("base_url", "")

    if retry_on_timeout is None:
        retry_on_timeout = method.upper() in IDEMPOTENT_HTTP_METHODS

    if use_client_api is None:
        use_client_api = MANIFEST_API_CONFIG.get("use_client_api", True)
    
    # Determine which endpoint and auth mode to use
    if use_client_api:
        # CLIENT MODE: Use static API key
        if not client_endpoint:
            logger.error("Client endpoint not available for this API call")
            return {"error": "Client API endpoint not configured for this operation"}, False
        
        endpoint = client_endpoint
        api_key = MANIFEST_API_CONFIG.get("api_key")
        
        if not api_key:
            logger.error("Missing API key for client mode")
            return {"error": "Missing API key in configuration"}, False
        
        # Build headers for client mode
        headers = get_client_api_headers(api_key, org_key, org_id)

        # Add subscription ID if provided
        if subscription_id:
            headers["Mit-Subscription-ID"] = str(subscription_id)
        
        # Construct full URL
        if not endpoint.startswith("http"):
            endpoint = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        logger.info(f"[CLIENT MODE] Calling Manifest API: {method} {client_endpoint}")

        try:
            # Make API request
            response = requests.request(
                method=method,
                url=endpoint,
                headers=headers,
                params=params,
                json=json_data,
                data=data,
                timeout=timeout
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse JSON response
            result_data = response.json()
            logger.info(f"[CLIENT MODE] API call successful: {method} {endpoint}")
            
            return result_data, True
            
        except requests.exceptions.Timeout:
            logger.error(f"[CLIENT MODE] Timeout calling Manifest API: {endpoint}")
            return {"error": "Request timeout"}, False
        except requests.exceptions.HTTPError as e:
            logger.error(f"[CLIENT MODE] HTTP error calling Manifest API: {e}")
            status_code = e.response.status_code if e.response else "Unknown"
            error_detail = e.response.text if e.response else str(e)
            return {"error": f"HTTP {status_code}: {error_detail}"}, False
        except requests.exceptions.RequestException as e:
            logger.error(f"[CLIENT MODE] Request error calling Manifest API: {e}")
            return {"error": str(e)}, False
        except Exception as e:
            logger.error(f"[CLIENT MODE] Unexpected error calling Manifest API: {e}")
            return {"error": str(e)}, False
    
    else:
        # INTERNAL MODE: Use OAuth2 Bearer token
        endpoint = internal_endpoint
        
        # Construct full URL
        if not endpoint.startswith("http"):
            endpoint = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        # Get access token
        token = generate_access_token()
        if not token:
            logger.error("[INTERNAL MODE] Failed to generate access token")
            return {"error": "Failed to generate access token"}, False
        
        # Build headers for internal mode
        headers = get_internal_api_headers(token, org_key, org_id, subscription_id)
        
        logger.info(f"[INTERNAL MODE] Calling Manifest API: {method} {internal_endpoint}")

        timeout_max_retries = MANIFEST_API_CONFIG.get("timeout_max_retries", 2) if retry_on_timeout else 0
        timeout_retry_backoff_seconds = MANIFEST_API_CONFIG.get("timeout_retry_backoff_seconds", 1)
        timeout_attempt = 0

        while True:
            try:
                # Make API request
                response = requests.request(
                    method=method,
                    url=endpoint,
                    headers=headers,
                    params=params,
                    json=json_data,
                    data=data,
                    timeout=timeout
                )

                # Handle 401 Unauthorized - token might be expired
                if response.status_code == 401 and retry_on_auth_failure:
                    logger.warning("[INTERNAL MODE] Received 401 Unauthorized, refreshing token and retrying...")
                    
                    # Force token refresh
                    token = generate_access_token(force_refresh=True)
                    if not token:
                        logger.error("[INTERNAL MODE] Failed to refresh access token")
                        return {"error": "Failed to refresh access token"}, False
                    
                    # Rebuild headers with new token
                    headers = get_internal_api_headers(token, org_key, org_id, subscription_id)
                    
                    # Retry request
                    response = requests.request(
                        method=method,
                        url=endpoint,
                        headers=headers,
                        params=params,
                        json=json_data,
                        data=data,
                        timeout=timeout
                    )
                
                # Check for HTTP errors
                response.raise_for_status()
                
                # Parse JSON response
                result_data = response.json()
                logger.info(f"[INTERNAL MODE] API call successful: {method} {internal_endpoint}")
                
                return result_data, True
                
            except requests.exceptions.Timeout:
                timeout_attempt += 1
                if timeout_attempt > timeout_max_retries:
                    logger.error(f"[INTERNAL MODE] Timeout calling Manifest API after {timeout_attempt} attempt(s): {internal_endpoint}")
                    return {"error": "Request timeout"}, False
                backoff = timeout_retry_backoff_seconds * (2 ** (timeout_attempt - 1))
                logger.warning(
                    f"[INTERNAL MODE] Timeout calling Manifest API (attempt {timeout_attempt}/{timeout_max_retries}), "
                    f"retrying in {backoff}s: {internal_endpoint}"
                )
                time.sleep(backoff)
                continue
            except requests.exceptions.HTTPError as e:
                logger.error(f"[INTERNAL MODE] HTTP error calling Manifest API: {e}")
                status_code = e.response.status_code if e.response else "Unknown"
                error_detail = e.response.text if e.response else str(e)
                return {"error": f"HTTP {status_code}: {error_detail}"}, False
            except requests.exceptions.RequestException as e:
                logger.error(f"[INTERNAL MODE] Request error calling Manifest API: {e}")
                return {"error": str(e)}, False
            except Exception as e:
                logger.error(f"[INTERNAL MODE] Unexpected error calling Manifest API: {e}")
                return {"error": str(e)}, False