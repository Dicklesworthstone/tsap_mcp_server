# examples/api_client.py
import httpx
import asyncio
import os
from typing import Dict, Any, Optional

from rich import print as rich_print
from rich.panel import Panel

# --- Configuration ---
BASE_URL = os.environ.get("TSAP_BASE_URL", "http://127.0.0.1:8021")
API_KEY = os.environ.get("TSAP_API_KEY", "your-default-api-key") # Replace with your actual API key or use env var

# --- API Client Class ---
class TSAPClient:
    """Asynchronous client for interacting with the TSAP MCP Server API."""

    def __init__(self, base_url: str = BASE_URL, api_key: str = API_KEY):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(base_url=self.base_url, headers=self.headers, timeout=120.0) # Increased timeout

    async def close(self):
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _request(
        self,
        method: str,
        endpoint: str,
        json_payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Makes an HTTP request to the TSAP API."""
        try:
            response = await self._client.request(
                method, endpoint, json=json_payload, params=params
            )
            response.raise_for_status()  # Raise exception for 4xx/5xx status codes
            return response.json()
        except httpx.HTTPStatusError as e:
            rich_print(f"[bold red]HTTP Error:[/bold red] {e.response.status_code} - {e.response.text}")
            return {"error": {"code": f"HTTP_{e.response.status_code}", "message": e.response.text}}
        except httpx.RequestError as e:
            rich_print(f"[bold red]Request Error:[/bold red] {e}")
            return {"error": {"code": "REQUEST_ERROR", "message": str(e)}}
        except Exception as e:
            rich_print(f"[bold red]Unexpected Error:[/bold red] {e}")
            return {"error": {"code": "CLIENT_ERROR", "message": str(e)}}

    async def post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send a POST request."""
        return await self._request("POST", endpoint, json_payload=payload)

    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a GET request."""
        return await self._request("GET", endpoint, params=params)

    async def health_check(self) -> Dict[str, Any]:
        """Check the server health."""
        return await self.get("/health")

    async def get_version(self) -> Dict[str, Any]:
        """Get server version information."""
        return await self.get("/version")

    async def check_job_status(self, job_url: str) -> Dict[str, Any]:
        """Check the status of an asynchronous job."""
        # Extract the relative path from the full URL if necessary
        if job_url.startswith(self.base_url):
            job_endpoint = job_url[len(self.base_url):]
        else:
            job_endpoint = job_url
        return await self.get(job_endpoint)

    async def get_job_result(self, result_url: str) -> Dict[str, Any]:
        """Get the result of a completed asynchronous job."""
        if result_url.startswith(self.base_url):
            result_endpoint = result_url[len(self.base_url):]
        else:
            result_endpoint = result_url
        return await self.get(result_endpoint)

    async def wait_for_job(self, job_info: Dict[str, Any], poll_interval: float = 2.0) -> Dict[str, Any]:
        """Poll a job status URL until it completes, then fetch the result."""
        status_url = job_info.get("status_url")
        if not status_url:
            raise ValueError("Job info does not contain a status_url")

        rich_print(f"Job submitted (ID: {job_info.get('job_id')}). Waiting for completion...")
        start_time = asyncio.get_event_loop().time()  # noqa: F841

        while True:
            status_info = await self.check_job_status(status_url)
            status = status_info.get("status", "unknown")
            progress = status_info.get("progress")

            if progress is not None:
                rich_print(f"  Status: {status}, Progress: {progress:.1f}%")
            else:
                rich_print(f"  Status: {status}")

            if status == "completed":
                result_url = status_info.get("result_url")
                if not result_url:
                    raise ValueError("Completed job status does not contain a result_url")
                rich_print("[green]Job completed. Fetching result...[/green]")
                return await self.get_job_result(result_url)
            elif status in ["failed", "cancelled", "timeout"]:
                error_msg = status_info.get("error", "Job failed without specific error message")
                raise RuntimeError(f"Job failed with status '{status}': {error_msg}")
            elif status == "unknown":
                 raise RuntimeError(f"Job status returned 'unknown': {status_info}")

            # Wait before polling again
            await asyncio.sleep(poll_interval)

            # Optional: Add a timeout for waiting
            # if asyncio.get_event_loop().time() - start_time > MAX_WAIT_TIME:
            #     raise TimeoutError("Timeout waiting for job completion")

# --- Example Usage ---
async def main():
    """Example usage of the TSAPClient."""
    async with TSAPClient() as client:
        rich_print(Panel("[bold blue]Checking TSAP Server Health and Version...[/bold blue]", expand=False))

        # Health Check
        health = await client.health_check()
        rich_print("[cyan]Health Check:[/cyan]", health)

        # Version Info
        version = await client.get_version()
        rich_print("[cyan]Version Info:[/cyan]", version)

        # Example: Ripgrep Search (demonstrates a core tool API call)
        # Assumes `tsap_example_data/code/main.py` exists
        rich_print(Panel("[bold blue]Running Ripgrep Search Example...[/bold blue]", expand=False))
        ripgrep_payload = {
            "params": {
                "pattern": "API_KEY",
                "paths": ["tsap_example_data/code/"],
                "case_sensitive": False,
                "file_patterns": ["*.py"],
                "context_lines": 1,
            }
            # Add "async_execution": True here to test async jobs
        }
        rich_print("[cyan]Sending Ripgrep Request:[/cyan]", ripgrep_payload)
        response = await client.post("/api/core/ripgrep", payload=ripgrep_payload)

        if response and "job_id" in response:
            # Handle async job
            try:
                final_result = await client.wait_for_job(response)
                rich_print("[bold green]Async Ripgrep Result:[/bold green]")
                rich_print(final_result)
            except Exception as e:
                 rich_print(f"[bold red]Error waiting for job:[/bold red] {e}")
        elif response and response.get("result"):
            # Handle sync result
            rich_print("[bold green]Sync Ripgrep Result:[/bold green]")
            rich_print(response)
        else:
            rich_print("[bold red]Ripgrep request failed or returned unexpected response.[/bold red]", response)

if __name__ == "__main__":
    # Check for API key
    if API_KEY == "your-default-api-key":
        rich_print("[bold yellow]Warning:[/bold yellow] Using default API key. Set the TSAP_API_KEY environment variable.")

    asyncio.run(main())