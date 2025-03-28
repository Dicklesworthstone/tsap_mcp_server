"""
Custom templates module for TSAP MCP Server.

This module is intended to be extended by users to define their own custom
templates for specific search, analysis, or processing workflows. Templates
registered here will be automatically discovered by the template system.
"""




# Below are examples of how to create custom templates.
# Uncomment and modify as needed.

# @template_decorator(
#     name="custom_code_analysis",
#     description="Custom template for analyzing code with specific patterns",
#     version="0.1.0",
#     tags=["code", "analysis", "custom"]
# )
# class CustomCodeAnalysisTemplate(Template):
#     """
#     Custom template for analyzing code with specific patterns.
#     
#     This template extends the standard code analysis with additional
#     patterns and post-processing specific to your needs.
#     """
#     
#     async def execute(self, params: Dict[str, Any]) -> TemplateResult:
#         """
#         Execute the custom code analysis template.
#         
#         Args:
#             params: Parameters for the template execution
#             
#         Returns:
#             Result of the template execution
#         """
#         # Implement custom code analysis logic here
#         # This would typically combine core and composite operations
#         
#         return TemplateResult(
#             template_id=self.id,
#             template_name=self.name,
#             execution_time=0.0,
#             result={},
#             status="completed"
#         )


# @template_decorator(
#     name="custom_document_analyzer",
#     description="Custom template for analyzing documents in a specific format",
#     version="0.1.0",
#     tags=["documents", "analysis", "custom"]
# )
# class CustomDocumentAnalyzerTemplate(Template):
#     """
#     Custom template for analyzing documents in a specific format.
#     
#     This template is designed to handle documents with a particular structure
#     or content and extract information in a specialized way.
#     """
#     
#     async def execute(self, params: Dict[str, Any]) -> TemplateResult:
#         """
#         Execute the custom document analyzer template.
#         
#         Args:
#             params: Parameters for the template execution
#             
#         Returns:
#             Result of the template execution
#         """
#         # Implement custom document analysis logic here
#         
#         return TemplateResult(
#             template_id=self.id,
#             template_name=self.name,
#             execution_time=0.0,
#             result={},
#             status="completed"
#         )


# Example helper function for use with custom templates

# async def run_custom_code_analysis(params: Dict[str, Any]) -> TemplateResult:
#     """
#     Run the custom code analysis template.
#     
#     Args:
#         params: Parameters for the template execution
#         
#     Returns:
#         Result of the template execution
#     """
#     template = CustomCodeAnalysisTemplate()
#     return await template.execute(params)