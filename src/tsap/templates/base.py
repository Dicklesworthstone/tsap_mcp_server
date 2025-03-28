"""
TSAP Template Base Classes.

This module defines the base classes for TSAP task templates, which provide
pre-configured workflows for common tasks.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, TypeVar, Generic, ClassVar, Tuple, Type
from dataclasses import dataclass, field

from tsap.utils.logging import logger
from tsap.utils.helpers import generate_id
from tsap.utils.errors import TSAPError


# Type variable for template parameters
P = TypeVar('P')
R = TypeVar('R')


class TemplateError(TSAPError):
    """Error raised when a template execution fails."""
    
    def __init__(
        self,
        message: str,
        template_id: Optional[str] = None,
        template_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a template error.
        
        Args:
            message: Error message
            template_id: Template ID
            template_name: Template name
            details: Additional error details
        """
        error_details = details or {}
        if template_id:
            error_details["template_id"] = template_id
        if template_name:
            error_details["template_name"] = template_name
        
        super().__init__(message, "TEMPLATE_ERROR", error_details)


@dataclass
class TemplateParameter:
    """Template parameter definition."""
    
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    options: Optional[List[Any]] = None
    validation: Optional[Callable[[Any], bool]] = None
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate a parameter value.
        
        Args:
            value: Parameter value
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if value is required
        if self.required and value is None:
            return False, f"Parameter '{self.name}' is required"
        
        # If value is None and not required, it's valid
        if value is None and not self.required:
            return True, None
        
        # Check type
        if self.type == "str" and not isinstance(value, str):
            return False, f"Parameter '{self.name}' must be a string"
        elif self.type == "int" and not isinstance(value, int):
            return False, f"Parameter '{self.name}' must be an integer"
        elif self.type == "float" and not isinstance(value, (int, float)):
            return False, f"Parameter '{self.name}' must be a number"
        elif self.type == "bool" and not isinstance(value, bool):
            return False, f"Parameter '{self.name}' must be a boolean"
        elif self.type == "list" and not isinstance(value, list):
            return False, f"Parameter '{self.name}' must be a list"
        elif self.type == "dict" and not isinstance(value, dict):
            return False, f"Parameter '{self.name}' must be a dictionary"
        
        # Check options
        if self.options is not None and value not in self.options:
            return False, f"Parameter '{self.name}' must be one of: {', '.join(str(o) for o in self.options)}"
        
        # Check custom validation
        if self.validation is not None:
            try:
                if not self.validation(value):
                    return False, f"Parameter '{self.name}' failed validation"
            except Exception as e:
                return False, f"Parameter '{self.name}' validation error: {str(e)}"
        
        return True, None


@dataclass
class TemplateResult:
    """Base class for template execution results."""
    
    template_id: str
    template_name: str
    status: str = "success"
    message: Optional[str] = None
    execution_time: Optional[float] = None
    timestamp: Optional[float] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary.
        
        Returns:
            Dictionary representation of the result
        """
        result = {
            "template_id": self.template_id,
            "template_name": self.template_name,
            "status": self.status,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
            "parameters": self.parameters,
            "results": self.results,
        }
        
        if self.message:
            result["message"] = self.message
            
        if self.error:
            result["error"] = str(self.error)
            
        return result


class Template(Generic[P, R], ABC):
    """Base class for TSAP task templates."""
    
    # Class attributes
    id: ClassVar[str]
    name: ClassVar[str]
    description: ClassVar[str]
    version: ClassVar[str]
    author: ClassVar[str]
    parameters: ClassVar[List[TemplateParameter]]
    
    def __init__(self):
        """Initialize the template."""
        # Generate ID if not defined
        if not hasattr(self, "id") or not self.id:
            self.id = generate_id(f"{self.__class__.__name__.lower()}-")
    
    @abstractmethod
    async def execute(self, params: P) -> R:
        """Execute the template.
        
        Args:
            params: Template parameters
            
        Returns:
            Template result
            
        Raises:
            TemplateError: If template execution fails
        """
        pass
    
    async def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate template parameters.
        
        Args:
            params: Parameters to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        if not hasattr(self, "parameters"):
            return True, []
        
        errors: List[str] = []
        
        # Check for unknown parameters
        param_names = {p.name for p in self.parameters}
        for name in params.keys():
            if name not in param_names:
                errors.append(f"Unknown parameter: {name}")
        
        # Validate each parameter
        for param in self.parameters:
            value = params.get(param.name)
            
            # Use default if not provided
            if value is None and param.default is not None:
                value = param.default
                params[param.name] = value
            
            # Validate parameter
            valid, error = param.validate(value)
            if not valid:
                errors.append(error if error else f"Invalid parameter: {param.name}")
        
        return len(errors) == 0, errors
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get template metadata.
        
        Returns:
            Dictionary with template metadata
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                    "options": p.options,
                }
                for p in self.parameters
            ] if hasattr(self, "parameters") else [],
        }


class TemplateRunner:
    """Runs TSAP templates with error handling and logging."""
    
    def __init__(self):
        """Initialize the template runner."""
        # Dictionary of registered templates
        self.templates: Dict[str, Template] = {}
    
    def register_template(self, template: Template) -> None:
        """Register a template.
        
        Args:
            template: Template to register
        """
        self.templates[template.id] = template
        logger.debug(f"Registered template: {template.name} ({template.id})")
    
    def unregister_template(self, template_id: str) -> None:
        """Unregister a template.
        
        Args:
            template_id: Template ID
        """
        if template_id in self.templates:
            template = self.templates[template_id]
            del self.templates[template_id]
            logger.debug(f"Unregistered template: {template.name} ({template.id})")
    
    def get_template(self, template_id: str) -> Optional[Template]:
        """Get a template by ID.
        
        Args:
            template_id: Template ID
            
        Returns:
            Template or None if not found
        """
        return self.templates.get(template_id)
    
    def get_template_by_name(self, name: str) -> Optional[Template]:
        """Get a template by name.
        
        Args:
            name: Template name
            
        Returns:
            Template or None if not found
        """
        for template in self.templates.values():
            if template.name == name:
                return template
        return None
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """Get a list of all registered templates.
        
        Returns:
            List of template metadata
        """
        return [template.get_metadata() for template in self.templates.values()]
    
    async def run_template(self, template_id: str, params: Dict[str, Any]) -> TemplateResult:
        """Run a template by ID.
        
        Args:
            template_id: Template ID
            params: Template parameters
            
        Returns:
            Template result
            
        Raises:
            TemplateError: If template not found or execution fails
        """
        # Check if template exists
        template = self.get_template(template_id)
        if not template:
            raise TemplateError(f"Template not found: {template_id}", template_id=template_id)
        
        # Run the template
        return await self._run_template(template, params)
    
    async def run_template_by_name(self, name: str, params: Dict[str, Any]) -> TemplateResult:
        """Run a template by name.
        
        Args:
            name: Template name
            params: Template parameters
            
        Returns:
            Template result
            
        Raises:
            TemplateError: If template not found or execution fails
        """
        # Check if template exists
        template = self.get_template_by_name(name)
        if not template:
            raise TemplateError(f"Template not found: {name}", template_name=name)
        
        # Run the template
        return await self._run_template(template, params)
    
    async def _run_template(self, template: Template, params: Dict[str, Any]) -> TemplateResult:
        """Run a template.
        
        Args:
            template: Template to run
            params: Template parameters
            
        Returns:
            Template result
            
        Raises:
            TemplateError: If parameter validation or execution fails
        """
        import time
        
        # Start time
        start_time = time.time()
        
        try:
            # Validate parameters
            valid, errors = await template.validate_parameters(params)
            if not valid:
                raise TemplateError(
                    f"Parameter validation failed: {', '.join(errors)}",
                    template_id=template.id,
                    template_name=template.name,
                )
            
            # Execute template
            logger.info(f"Running template: {template.name} ({template.id})")
            result = await template.execute(params)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Create template result if not already a TemplateResult
            if not isinstance(result, TemplateResult):
                result = TemplateResult(
                    template_id=template.id,
                    template_name=template.name,
                    status="success",
                    execution_time=execution_time,
                    timestamp=time.time(),
                    parameters=params,
                    results=result if isinstance(result, dict) else {"result": result},
                )
            else:
                # Update execution time if not set
                if not result.execution_time:
                    result.execution_time = execution_time
                if not result.timestamp:
                    result.timestamp = time.time()
            
            logger.info(
                f"Template executed successfully: {template.name} "
                f"({execution_time:.3f}s)"
            )
            
            return result
            
        except Exception as e:
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Log error
            logger.error(
                f"Template execution failed: {template.name} ({template.id}): {str(e)}"
            )
            
            # Create template result for error
            result = TemplateResult(
                template_id=template.id,
                template_name=template.name,
                status="error",
                message=str(e),
                execution_time=execution_time,
                timestamp=time.time(),
                parameters=params,
                error=e,
            )
            
            # Re-raise as TemplateError
            if not isinstance(e, TemplateError):
                raise TemplateError(
                    str(e),
                    template_id=template.id,
                    template_name=template.name,
                    details={"original_error": str(e)},
                ) from e
            
            raise
            
        finally:
            # Log completion
            logger.debug(
                f"Template run completed: {template.name} ({template.id}) "
                f"in {time.time() - start_time:.3f}s"
            )


# Global template runner instance
_template_runner = TemplateRunner()


def get_template_runner() -> TemplateRunner:
    """Get the global template runner instance.
    
    Returns:
        Global template runner
    """
    return _template_runner


def register_template(template: Template) -> None:
    """Register a template with the global template runner.
    
    Args:
        template: Template to register
    """
    get_template_runner().register_template(template)


def template_decorator(
    name: str,
    description: str,
    version: str = "1.0.0",
    author: str = "TSAP Team",
) -> Callable[[Type[Template]], Type[Template]]:
    """Class decorator for templates.
    
    Args:
        name: Template name
        description: Template description
        version: Template version
        author: Template author
        
    Returns:
        Decorator function
    """
    def decorator(cls: Type[Template]) -> Type[Template]:
        cls.name = name
        cls.description = description
        cls.version = version
        cls.author = author
        
        # Register template when it's decorated
        def register_on_module_load():
            try:
                # Create an instance of the template
                template = cls()
                
                # Register with global template runner
                register_template(template)
            except Exception as e:
                logger.error(f"Failed to register template {name}: {e}")
        
        # Schedule registration to happen after module loading is complete
        from threading import Timer
        Timer(0, register_on_module_load).start()
        
        return cls
    
    return decorator


def parameter(
    name: str,
    type: str,
    description: str,
    required: bool = True,
    default: Any = None,
    options: Optional[List[Any]] = None,
    validation: Optional[Callable[[Any], bool]] = None,
) -> TemplateParameter:
    """Create a template parameter definition.
    
    Args:
        name: Parameter name
        type: Parameter type
        description: Parameter description
        required: Whether the parameter is required
        default: Default value for the parameter
        options: Valid options for the parameter
        validation: Custom validation function
        
    Returns:
        Template parameter definition
    """
    return TemplateParameter(
        name=name,
        type=type,
        description=description,
        required=required,
        default=default,
        options=options,
        validation=validation,
    )