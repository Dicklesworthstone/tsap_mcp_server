"""
Resource Allocator for the TSAP MCP Server.

This module implements a tool for intelligently allocating computational
resources based on the complexity of tasks, available system resources,
and configured performance modes.
"""

import asyncio
import psutil
import multiprocessing
from typing import Dict, Any, Optional

from tsap.utils.logging import logger
from tsap.analysis.base import BaseAnalysisTool, register_analysis_tool, AnalysisContext
from tsap.performance_mode import get_performance_mode


@register_analysis_tool("resource_allocator")
class ResourceAllocator(BaseAnalysisTool):
    """
    Manages the allocation of computational resources for TSAP operations.
    
    The ResourceAllocator analyzes tasks, estimates their resource requirements,
    and allocates appropriate resources based on system availability and
    performance mode settings.
    """
    
    def __init__(self, name: str = "resource_allocator"):
        """Initialize the resource allocator."""
        super().__init__(name)
        self._system_resources = self._get_system_resources()
    
    def _get_system_resources(self) -> Dict[str, Any]:
        """
        Get available system resources.
        
        Returns:
            Dictionary of system resource information
        """
        try:
            cpu_count = multiprocessing.cpu_count()
            available_memory = psutil.virtual_memory().available
            total_memory = psutil.virtual_memory().total
            disk_usage = psutil.disk_usage('/').percent
            
            return {
                "cpu_count": cpu_count,
                "available_memory": available_memory,
                "total_memory": total_memory,
                "memory_percent": (total_memory - available_memory) / total_memory * 100,
                "disk_usage": disk_usage
            }
        except Exception as e:
            logger.warning(
                f"Error getting system resources: {str(e)}. Using defaults.",
                component="analysis",
                operation="resource_allocator"
            )
            
            # Return conservative defaults
            return {
                "cpu_count": 2,
                "available_memory": 2 * 1024 * 1024 * 1024,  # 2 GB
                "total_memory": 4 * 1024 * 1024 * 1024,      # 4 GB
                "memory_percent": 50.0,
                "disk_usage": 70.0
            }
    
    def _update_system_resources(self) -> None:
        """Update the cached system resource information."""
        self._system_resources = self._get_system_resources()
        
        logger.debug(
            f"System resources updated: {self._system_resources['cpu_count']} CPUs, "
            f"{self._system_resources['available_memory'] / (1024**3):.2f} GB available memory",
            component="analysis",
            operation="resource_allocator"
        )
    
    def _estimate_task_requirements(
        self, 
        task_type: str,
        task_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Estimate resource requirements for a task.
        
        Args:
            task_type: Type of task (e.g., 'search', 'analyze', 'transform')
            task_params: Parameters for the task
            
        Returns:
            Dictionary with estimated resource requirements
        """
        # Default conservative estimates
        estimates = {
            "cpu_threads": 1,
            "memory_mb": 100,
            "io_intensity": "low",  # low, medium, high
            "execution_time_s": 5,
            "priority": "normal"    # low, normal, high
        }
        
        # Adjust based on task type
        if task_type == "search":
            # Ripgrep searches can be CPU and I/O intensive
            file_count = len(task_params.get("paths", []))
            is_recursive = task_params.get("recursive", False)
            
            estimates["cpu_threads"] = min(2, self._system_resources["cpu_count"] // 2)
            estimates["io_intensity"] = "high" if file_count > 100 or is_recursive else "medium"
            estimates["execution_time_s"] = 10 if file_count > 1000 else 5
        
        elif task_type == "analyze_code":
            # Code analysis can be memory intensive
            file_count = len(task_params.get("paths", []))
            
            estimates["cpu_threads"] = min(4, self._system_resources["cpu_count"] // 2)
            estimates["memory_mb"] = 500 if file_count > 1000 else 200
            estimates["execution_time_s"] = 30 if file_count > 1000 else 15
        
        elif task_type == "pdf_extract":
            # PDF extraction can be CPU and memory intensive
            estimates["cpu_threads"] = 1  # Usually single-threaded
            estimates["memory_mb"] = 300
            estimates["execution_time_s"] = 10
        
        elif task_type == "parallel_search":
            # Parallel searches can use multiple cores efficiently
            pattern_count = len(task_params.get("patterns", []))
            
            estimates["cpu_threads"] = min(pattern_count, self._system_resources["cpu_count"] - 1)
            estimates["memory_mb"] = 100 * pattern_count
            estimates["execution_time_s"] = 15
        
        # Adjust based on performance mode
        performance_mode = get_performance_mode()
        
        if performance_mode == "fast":
            # Fast mode: reduce resource usage to finish quickly
            estimates["cpu_threads"] = max(1, estimates["cpu_threads"] // 2)
            estimates["memory_mb"] = max(50, estimates["memory_mb"] // 2)
        
        elif performance_mode == "deep":
            # Deep mode: allocate more resources for thorough analysis
            estimates["cpu_threads"] = min(estimates["cpu_threads"] * 2, self._system_resources["cpu_count"] - 1)
            estimates["memory_mb"] = estimates["memory_mb"] * 2
            estimates["execution_time_s"] = estimates["execution_time_s"] * 1.5
        
        return estimates
    
    def _calculate_allocation(
        self,
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate actual resource allocation based on requirements and availability.
        
        Args:
            requirements: Estimated resource requirements
            
        Returns:
            Dictionary with allocated resources
        """
        # Update system resources
        self._update_system_resources()
        
        # Get current system load
        system_load = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": self._system_resources["memory_percent"],
            "disk_usage": self._system_resources["disk_usage"]
        }
        
        # Calculate available resources
        available_threads = max(1, self._system_resources["cpu_count"] - 1)
        available_memory_mb = self._system_resources["available_memory"] / (1024 * 1024)
        
        # Adjust based on system load
        if system_load["cpu_percent"] > 80:
            available_threads = max(1, available_threads // 2)
        
        if system_load["memory_percent"] > 80:
            available_memory_mb = available_memory_mb * 0.7  # Be more conservative
        
        # Calculate actual allocation
        allocated_threads = min(requirements["cpu_threads"], available_threads)
        allocated_memory_mb = min(requirements["memory_mb"], available_memory_mb * 0.8)  # Leave some buffer
        
        # Configure environment variables for subprocess control
        env_vars = {
            "OMP_NUM_THREADS": str(allocated_threads),
            "RAYON_NUM_THREADS": str(allocated_threads),
            "PARALLEL_THREADS": str(allocated_threads),
            "MAX_MEMORY_MB": str(int(allocated_memory_mb))
        }
        
        # Log allocation
        logger.debug(
            f"Resource allocation: {allocated_threads} threads, {allocated_memory_mb:.2f} MB memory",
            component="analysis",
            operation="resource_allocator"
        )
        
        return {
            "cpu_threads": allocated_threads,
            "memory_mb": allocated_memory_mb,
            "env_vars": env_vars,
            "throttle": system_load["cpu_percent"] > 90,  # Throttle if system is overloaded
            "system_load": system_load
        }
    
    async def allocate_resources(
        self,
        task_type: str,
        task_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Allocate resources for a task.
        
        Args:
            task_type: Type of task
            task_params: Parameters for the task
            
        Returns:
            Dictionary with allocated resources
        """
        # Estimate resource requirements
        requirements = self._estimate_task_requirements(task_type, task_params)
        
        # Calculate allocation
        allocation = self._calculate_allocation(requirements)
        
        # Apply throttling if needed
        if allocation["throttle"]:
            logger.warning(
                "System is overloaded, throttling task execution",
                component="analysis",
                operation="resource_allocator"
            )
            
            # Add a small delay to reduce system load
            await asyncio.sleep(1.0)
        
        return allocation
    
    async def analyze(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze and allocate resources.
        
        Args:
            params: Dictionary with task parameters including:
                   - task_type: Type of task
                   - task_params: Parameters for the task
            
        Returns:
            Dictionary with allocated resources and system information
        """
        context = AnalysisContext()  # noqa: F841
        
        try:
            async with self._measure_execution_time():
                task_type = params.get("task_type", "generic")
                task_params = params.get("task_params", {})
                
                # Allocate resources
                allocation = await self.allocate_resources(task_type, task_params)
                
                # Add system resource information
                result = {
                    "allocation": allocation,
                    "system_resources": self._system_resources,
                    "performance_mode": get_performance_mode(),
                    "execution_stats": self.get_statistics()
                }
                
                return result
        except Exception as e:
            logger.error(
                f"Error during resource allocation: {str(e)}",
                component="analysis",
                operation="resource_allocator"
            )
            raise


# Global instance for convenience
_resource_allocator: Optional[ResourceAllocator] = None


def get_resource_allocator() -> ResourceAllocator:
    """
    Get the global resource allocator instance.
    
    Returns:
        ResourceAllocator instance
    """
    global _resource_allocator
    
    if _resource_allocator is None:
        _resource_allocator = ResourceAllocator()
    
    return _resource_allocator


async def allocate_resources(task_type: str, task_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to allocate resources for a task.
    
    Args:
        task_type: Type of task
        task_params: Parameters for the task
        
    Returns:
        Dictionary with allocated resources
    """
    allocator = get_resource_allocator()
    return await allocator.allocate_resources(task_type, task_params)