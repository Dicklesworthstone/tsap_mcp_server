"""
TSAP Templates Package.

This package provides task templates for common workflows, enabling
standardized execution of complex operations with minimal configuration.
"""

from tsap.templates.base import (
    Template,
    TemplateResult,
    TemplateError,
    template_decorator,
    parameter,
    get_template_runner,
    register_template,
)

# Import all templates to make them discoverable
from tsap.templates.security_audit import (
    SecurityAuditTemplate,
    run_security_audit,
)

# Add more templates here as they are implemented

# Convenience function to run a template by name
async def run_template(name: str, params: dict) -> TemplateResult:
    """Run a template by name.
    
    Args:
        name: Template name
        params: Template parameters
        
    Returns:
        Template result
        
    Raises:
        TemplateError: If template not found or execution fails
    """
    return await get_template_runner().run_template_by_name(name, params)


# Convenience function to get a list of available templates
def list_templates() -> list:
    """Get a list of available templates.
    
    Returns:
        List of template metadata
    """
    return get_template_runner().list_templates()


__all__ = [
    # Base classes and decorators
    "Template",
    "TemplateResult",
    "TemplateError",
    "template_decorator",
    "parameter",
    
    # Template runner
    "get_template_runner",
    "register_template",
    
    # Template convenience functions
    "run_template",
    "list_templates",
    
    # Specific templates
    "SecurityAuditTemplate",
    "run_security_audit",
]