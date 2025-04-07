"""
Semantic resources for TSAP MCP Server.

This module provides MCP resource implementations for accessing semantic search
capabilities and embedding models.
"""
import os
import json
from typing import Dict, Any, List, Optional
from mcp.server.fastmcp import FastMCP


def register_semantic_resources(mcp: FastMCP) -> None:
    """Register all semantic-related resources with the MCP server.
    
    Args:
        mcp: FastMCP server instance
    """
    
    @mcp.resource("semantic://corpus/{corpus_id}")
    async def get_corpus_info(corpus_id: str) -> str:
        """Get information about a semantic corpus.
        
        This resource provides metadata about a semantic corpus,
        including vector count, dimensions, and configuration.
        
        Args:
            corpus_id: ID of the corpus
            
        Returns:
            Corpus metadata as JSON string
        """
        # Get corpus info from the original implementation
        from tsap.composite.semantic_search import get_corpus_info as original_get_corpus_info
        
        try:
            corpus_info = await original_get_corpus_info(corpus_id)
            
            # Format as JSON
            if isinstance(corpus_info, dict):
                return json.dumps(corpus_info, indent=2)
            elif hasattr(corpus_info, "dict"):
                return json.dumps(corpus_info.dict(), indent=2)
            else:
                return json.dumps({"error": "Corpus info format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve corpus info: {str(e)}",
                "corpus_id": corpus_id
            }, indent=2)
    
    @mcp.resource("semantic://corpus/{corpus_id}/stats")
    async def get_corpus_stats(corpus_id: str) -> str:
        """Get statistical information about a semantic corpus.
        
        This resource provides detailed statistics about a corpus,
        including vector distribution, clustering information, and performance metrics.
        
        Args:
            corpus_id: ID of the corpus
            
        Returns:
            Corpus statistics as JSON string
        """
        # Get corpus stats from the original implementation
        from tsap.composite.semantic_search import get_corpus_stats as original_get_corpus_stats
        
        try:
            corpus_stats = await original_get_corpus_stats(corpus_id)
            
            # Format as JSON
            if isinstance(corpus_stats, dict):
                return json.dumps(corpus_stats, indent=2)
            elif hasattr(corpus_stats, "dict"):
                return json.dumps(corpus_stats.dict(), indent=2)
            else:
                return json.dumps({"error": "Corpus stats format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve corpus stats: {str(e)}",
                "corpus_id": corpus_id
            }, indent=2)
    
    @mcp.resource("semantic://corpus/{corpus_id}/fields")
    async def get_corpus_fields(corpus_id: str) -> str:
        """Get field information for a semantic corpus.
        
        This resource provides information about the fields/attributes
        available in the corpus documents.
        
        Args:
            corpus_id: ID of the corpus
            
        Returns:
            Corpus fields as JSON string
        """
        # Get corpus fields from the original implementation
        from tsap.composite.semantic_search import get_corpus_fields as original_get_corpus_fields
        
        try:
            corpus_fields = await original_get_corpus_fields(corpus_id)
            
            # Format as JSON
            if isinstance(corpus_fields, dict) or isinstance(corpus_fields, list):
                return json.dumps(corpus_fields, indent=2)
            elif hasattr(corpus_fields, "dict"):
                return json.dumps(corpus_fields.dict(), indent=2)
            else:
                return json.dumps({"error": "Corpus fields format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve corpus fields: {str(e)}",
                "corpus_id": corpus_id
            }, indent=2)
    
    @mcp.resource("semantic://model/{model_id}")
    async def get_model_info(model_id: str) -> str:
        """Get information about a semantic embedding model.
        
        This resource provides metadata about an embedding model,
        including dimensionality, training data, and capabilities.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Model information as JSON string
        """
        # Get model info from the original implementation
        from tsap.composite.semantic_search import get_model_info as original_get_model_info
        
        try:
            model_info = await original_get_model_info(model_id)
            
            # Format as JSON
            if isinstance(model_info, dict):
                return json.dumps(model_info, indent=2)
            elif hasattr(model_info, "dict"):
                return json.dumps(model_info.dict(), indent=2)
            else:
                return json.dumps({"error": "Model info format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve model info: {str(e)}",
                "model_id": model_id
            }, indent=2)
    
    @mcp.resource("semantic://models")
    async def list_models() -> str:
        """List available semantic embedding models.
        
        This resource provides a list of all available embedding models
        that can be used for semantic operations.
        
        Returns:
            List of models as JSON string
        """
        # List models from the original implementation
        from tsap.composite.semantic_search import list_models as original_list_models
        
        try:
            models = await original_list_models()
            
            # Format as JSON
            if isinstance(models, list):
                return json.dumps(models, indent=2)
            elif hasattr(models, "dict"):
                return json.dumps(models.dict(), indent=2)
            else:
                return json.dumps({"error": "Models format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to list models: {str(e)}"
            }, indent=2)
    
    @mcp.resource("semantic://corpora")
    async def list_corpora() -> str:
        """List available semantic corpora.
        
        This resource provides a list of all available semantic corpora
        that can be searched or analyzed.
        
        Returns:
            List of corpora as JSON string
        """
        # List corpora from the original implementation
        from tsap.composite.semantic_search import list_corpora as original_list_corpora
        
        try:
            corpora = await original_list_corpora()
            
            # Format as JSON
            if isinstance(corpora, list):
                return json.dumps(corpora, indent=2)
            elif hasattr(corpora, "dict"):
                return json.dumps(corpora.dict(), indent=2)
            else:
                return json.dumps({"error": "Corpora format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to list corpora: {str(e)}"
            }, indent=2)

    @mcp.resource("semantic://{query}")
    async def semantic_search(query: str) -> str:
        """Perform a semantic search for a query.
        
        This resource performs semantic search using vector embeddings
        to find content related to the query by meaning, not just keywords.
        
        Args:
            query: Search query
            
        Returns:
            Search results as JSON string
        """
        # Import from original implementation
        from tsap.composite.semantic_search import SemanticSearchParams, get_operation
        
        # Create parameters
        params = SemanticSearchParams(
            query=query,
            corpus_path=".",  # Default to current directory
            top_k=5,
            min_score=0.7,
        )
        
        # Get the operation and execute search
        operation = get_operation()
        result = await operation.execute(params)
        
        # Format the results
        formatted_results = []
        if hasattr(result, "results"):
            for item in result.results:
                formatted_results.append({
                    "path": item.path if hasattr(item, "path") else None,
                    "score": item.score if hasattr(item, "score") else None,
                    "text": item.text if hasattr(item, "text") else None,
                    "metadata": item.metadata if hasattr(item, "metadata") else None,
                })
        
        return json.dumps({
            "query": query,
            "results": formatted_results,
        }, indent=2)
    
    @mcp.resource("semantic://{query}/top/{k}")
    async def semantic_search_top_k(query: str, k: str) -> str:
        """Perform a semantic search with a specified number of results.
        
        This resource performs semantic search and returns the top K results.
        
        Args:
            query: Search query
            k: Number of top results to return
            
        Returns:
            Search results as JSON string
        """
        # Import from original implementation
        from tsap.composite.semantic_search import SemanticSearchParams, get_operation
        
        # Parse k to integer
        try:
            top_k = int(k)
        except ValueError:
            top_k = 5  # Default if parsing fails
        
        # Create parameters
        params = SemanticSearchParams(
            query=query,
            corpus_path=".",  # Default to current directory
            top_k=top_k,
            min_score=0.6,  # Lower threshold for more results
        )
        
        # Get the operation and execute search
        operation = get_operation()
        result = await operation.execute(params)
        
        # Format the results
        formatted_results = []
        if hasattr(result, "results"):
            for item in result.results:
                formatted_results.append({
                    "path": item.path if hasattr(item, "path") else None,
                    "score": item.score if hasattr(item, "score") else None,
                    "text": item.text if hasattr(item, "text") else None,
                    "metadata": item.metadata if hasattr(item, "metadata") else None,
                })
        
        return json.dumps({
            "query": query,
            "top_k": top_k,
            "results": formatted_results,
        }, indent=2)
    
    @mcp.resource("semantic://corpus/{corpus_path}/search/{query}")
    async def semantic_search_corpus(corpus_path: str, query: str) -> str:
        """Perform a semantic search on a specific corpus.
        
        This resource performs semantic search within a specific corpus path.
        
        Args:
            corpus_path: Path to the corpus to search
            query: Search query
            
        Returns:
            Search results as JSON string
        """
        # Import from original implementation
        from tsap.composite.semantic_search import SemanticSearchParams, get_operation
        
        # Create parameters
        params = SemanticSearchParams(
            query=query,
            corpus_path=corpus_path,
            top_k=5,
            min_score=0.7,
        )
        
        # Get the operation and execute search
        operation = get_operation()
        result = await operation.execute(params)
        
        # Format the results
        formatted_results = []
        if hasattr(result, "results"):
            for item in result.results:
                formatted_results.append({
                    "path": item.path if hasattr(item, "path") else None,
                    "score": item.score if hasattr(item, "score") else None,
                    "text": item.text if hasattr(item, "text") else None,
                    "metadata": item.metadata if hasattr(item, "metadata") else None,
                })
        
        return json.dumps({
            "query": query,
            "corpus_path": corpus_path,
            "results": formatted_results,
        }, indent=2) 