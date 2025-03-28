"""
Template registry system for managing and accessing templates.

This module provides a centralized registry for template management,
supporting registration, lookup, validation, and execution of templates.
"""

import importlib
import time
import pkgutil
from typing import Dict, List, Any, Optional, Type

from tsap.utils.logging import logger
from tsap.templates.base import Template, TemplateResult, TemplateError


class TemplateRegistry:
    """
    Global registry for templates.
    
    Provides a centralized registry for managing templates, allowing registration,
    lookup, and execution of templates by name or ID.
    """
    _templates: Dict[str, Type[Template]] = {}
    _instances: Dict[str, Template] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}
    _categories: Dict[str, List[str]] = {}
    
    @classmethod
    def register(cls, template_id: str, template_class: Type[Template]) -> None:
        """
        Register a template class with the registry.
        
        Args:
            template_id: Unique identifier for the template
            template_class: Template class to register
        """
        cls._templates[template_id] = template_class
        
        # Extract metadata
        metadata = template_class.get_metadata()
        cls._metadata[template_id] = metadata
        
        # Categorize template
        category = metadata.get("category", "general")
        if category not in cls._categories:
            cls._categories[category] = []
        cls._categories[category].append(template_id)
        
        logger.debug(f"Registered template: {template_id}")
    
    @classmethod
    def unregister(cls, template_id: str) -> bool:
        """
        Unregister a template from the registry.
        
        Args:
            template_id: Unique identifier for the template
            
        Returns:
            True if the template was unregistered, False if it wasn't registered
        """
        if template_id not in cls._templates:
            return False
        
        # Remove from templates
        template_class = cls._templates.pop(template_id)  # noqa: F841
        
        # Remove instance if it exists
        if template_id in cls._instances:
            del cls._instances[template_id]
        
        # Remove from metadata
        metadata = cls._metadata.pop(template_id, {})
        
        # Remove from categories
        category = metadata.get("category", "general")
        if category in cls._categories and template_id in cls._categories[category]:
            cls._categories[category].remove(template_id)
            
            # Clean up empty categories
            if not cls._categories[category]:
                del cls._categories[category]
        
        logger.debug(f"Unregistered template: {template_id}")
        return True
    
    @classmethod
    def get_template_class(cls, template_id: str) -> Optional[Type[Template]]:
        """
        Get a template class by ID.
        
        Args:
            template_id: Unique identifier for the template
            
        Returns:
            The template class, or None if not found
        """
        return cls._templates.get(template_id)
    
    @classmethod
    def get_template(cls, template_id: str) -> Optional[Template]:
        """
        Get or create a template instance by ID.
        
        Args:
            template_id: Unique identifier for the template
            
        Returns:
            An instance of the template, or None if not found
        """
        # Return existing instance if available
        if template_id in cls._instances:
            return cls._instances[template_id]
        
        # Create a new instance if the class is registered
        template_class = cls.get_template_class(template_id)
        if template_class:
            try:
                instance = template_class()
                cls._instances[template_id] = instance
                return instance
            except Exception as e:
                logger.error(f"Error instantiating template {template_id}: {str(e)}")
                return None
        
        return None
    
    @classmethod
    def get_template_by_name(cls, name: str) -> Optional[Template]:
        """
        Get a template instance by name.
        
        Args:
            name: Name of the template
            
        Returns:
            An instance of the template, or None if not found
        """
        # Find template ID by name
        for template_id, metadata in cls._metadata.items():
            if metadata.get("name") == name:
                return cls.get_template(template_id)
        
        return None
    
    @classmethod
    def list_templates(cls) -> List[Dict[str, Any]]:
        """
        List all registered templates.
        
        Returns:
            List of template metadata dictionaries
        """
        return [
            {
                "id": template_id,
                **metadata
            }
            for template_id, metadata in cls._metadata.items()
        ]
    
    @classmethod
    def list_categories(cls) -> List[str]:
        """
        List all template categories.
        
        Returns:
            List of category names
        """
        return list(cls._categories.keys())
    
    @classmethod
    def list_templates_by_category(cls, category: str) -> List[Dict[str, Any]]:
        """
        List templates in a specific category.
        
        Args:
            category: Category name
            
        Returns:
            List of template metadata dictionaries
        """
        if category not in cls._categories:
            return []
        
        return [
            {
                "id": template_id,
                **cls._metadata[template_id]
            }
            for template_id in cls._categories[category]
        ]
    
    @classmethod
    def find_templates(cls, query: str) -> List[Dict[str, Any]]:
        """
        Find templates matching a search query.
        
        Args:
            query: Search query (case-insensitive)
            
        Returns:
            List of matching template metadata dictionaries
        """
        query = query.lower()
        results = []
        
        for template_id, metadata in cls._metadata.items():
            # Check name
            if query in metadata.get("name", "").lower():
                results.append({"id": template_id, **metadata})
                continue
            
            # Check description
            if query in metadata.get("description", "").lower():
                results.append({"id": template_id, **metadata})
                continue
            
            # Check tags
            for tag in metadata.get("tags", []):
                if query in tag.lower():
                    results.append({"id": template_id, **metadata})
                    break
        
        return results
    
    @classmethod
    async def execute_template(cls, template_id: str, params: Dict[str, Any]) -> TemplateResult:
        """
        Execute a template by ID.
        
        Args:
            template_id: Unique identifier for the template
            params: Parameters for the template
            
        Returns:
            Result of the template execution
            
        Raises:
            TemplateError: If the template was not found or execution failed
        """
        # Get template instance
        template = cls.get_template(template_id)
        if not template:
            raise TemplateError(f"Template not found: {template_id}", template_id=template_id)
        
        try:
            # Validate parameters
            valid, errors = await template.validate_parameters(params)
            if not valid:
                error_list = "\n".join([f"- {error}" for error in errors])
                raise TemplateError(
                    f"Invalid parameters for template {template_id}:\n{error_list}",
                    template_id=template_id,
                    template_name=cls._metadata[template_id].get("name")
                )
            
            # Execute template
            start_time = time.time()
            result = await template.execute(params)
            execution_time = time.time() - start_time
            
            # Add execution time to result
            if hasattr(result, "metadata") and isinstance(result.metadata, dict):
                result.metadata["execution_time"] = execution_time
            
            return result
            
        except TemplateError:
            raise  # Re-raise template errors
        except Exception as e:
            raise TemplateError(
                f"Error executing template {template_id}: {str(e)}",
                template_id=template_id,
                template_name=cls._metadata[template_id].get("name"),
                details={"error": str(e)}
            ) from e
    
    @classmethod
    async def execute_template_by_name(cls, name: str, params: Dict[str, Any]) -> TemplateResult:
        """
        Execute a template by name.
        
        Args:
            name: Name of the template
            params: Parameters for the template
            
        Returns:
            Result of the template execution
            
        Raises:
            TemplateError: If the template was not found or execution failed
        """
        # Find template ID by name
        template_id = None
        for tid, metadata in cls._metadata.items():
            if metadata.get("name") == name:
                template_id = tid
                break
        
        if not template_id:
            raise TemplateError(f"Template not found: {name}")
        
        return await cls.execute_template(template_id, params)
    
    @classmethod
    def discover_templates(cls, package_path: str = "tsap.templates") -> List[str]:
        """
        Discover templates in a package.
        
        Args:
            package_path: Path to the package to search for templates
            
        Returns:
            List of discovered template IDs
        """
        try:
            # Import the package
            package = importlib.import_module(package_path)
            
            # Keep track of discovered templates
            discovered = []
            
            # Iterate through modules in the package
            if hasattr(package, "__path__"):
                package_name = package.__name__
                package_path = package.__path__
                
                for module_finder, name, _ in pkgutil.iter_modules(package_path):
                    # Skip __init__.py
                    if name == "__init__":
                        continue
                    
                    # Import the module
                    module_path = f"{package_name}.{name}"
                    try:
                        module = importlib.import_module(module_path)
                        
                        # Look for template classes
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            
                            # Check if it's a template class
                            if (isinstance(attr, type) and 
                                issubclass(attr, Template) and 
                                attr != Template and
                                hasattr(attr, "_template_id")):
                                
                                # Get template ID
                                template_id = getattr(attr, "_template_id")
                                
                                # Register the template if not already registered
                                if template_id not in cls._templates:
                                    cls.register(template_id, attr)
                                    discovered.append(template_id)
                        
                    except Exception as e:
                        logger.error(f"Error discovering templates in module {module_path}: {str(e)}")
            
            return discovered
            
        except Exception as e:
            logger.error(f"Error discovering templates in package {package_path}: {str(e)}")
            return []

def get_template_registry() -> TemplateRegistry:
    """
    Get the template registry.
    
    Returns:
        The template registry
    """
    return TemplateRegistry


def register_template(template_id: str, template_class: Type[Template]) -> None:
    """
    Register a template with the registry.
    
    Args:
        template_id: Unique identifier for the template
        template_class: Template class to register
    """
    TemplateRegistry.register(template_id, template_class)


def get_template(template_id: str) -> Optional[Template]:
    """
    Get a template instance by ID.
    
    Args:
        template_id: Unique identifier for the template
        
    Returns:
        An instance of the template, or None if not found
    """
    return TemplateRegistry.get_template(template_id)


def list_templates() -> List[Dict[str, Any]]:
    """
    List all registered templates.
    
    Returns:
        List of template metadata dictionaries
    """
    return TemplateRegistry.list_templates()


async def execute_template(template_id: str, params: Dict[str, Any]) -> TemplateResult:
    """
    Execute a template by ID.
    
    Args:
        template_id: Unique identifier for the template
        params: Parameters for the template
        
    Returns:
        Result of the template execution
        
    Raises:
        TemplateError: If the template was not found or execution failed
    """
    return await TemplateRegistry.execute_template(template_id, params)


# Discover templates upon import
TemplateRegistry.discover_templates()