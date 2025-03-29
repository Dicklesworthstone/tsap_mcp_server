from tsap.core.semantic_search_tool import get_tool as get_semantic_tool
from tsap.core.base import CompositeOperation, register_operation, get_operation
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class SemanticSearchParams(BaseModel):
    """Parameters for semantic search operation"""
    
    texts: List[str] = Field(..., description="List of documents to search")
    query: str = Field(..., description="Query text to search for")
    ids: Optional[List[str]] = Field(None, description="Optional document IDs")
    metadata: Optional[List[Dict[str, Any]]] = Field(None, description="Optional metadata for documents")
    top_k: int = Field(10, description="Number of top results to return")
    embedding_model: str = Field("nomic-ai/nomic-embed-text-v2-moe", description="Embedding model to use")
    use_gpu: bool = Field(True, description="Whether to use GPU if available")
    normalize_vectors: bool = Field(True, description="Whether to normalize vectors")
    
    def model_dump(self) -> Dict[str, Any]:
        """Make the model JSON serializable for cache operations."""
        return {
            "texts": self.texts,
            "query": self.query,
            "ids": self.ids,
            "metadata": self.metadata,
            "top_k": self.top_k,
            "embedding_model": self.embedding_model,
            "use_gpu": self.use_gpu,
            "normalize_vectors": self.normalize_vectors
        }
    
    def __str__(self) -> str:
        """String representation of the parameters."""
        return f"Query: {self.query}, Top-K: {self.top_k}, Model: {self.embedding_model}"


@register_operation("semantic_search")
class SemanticSearchOperation(CompositeOperation[SemanticSearchParams, List[Dict[str, Any]]]):
    """Composite operation for semantic search using vector embeddings."""
    
    async def execute(self, params: SemanticSearchParams) -> List[Dict[str, Any]]:
        """Execute the semantic search operation.
        
        Args:
            params: Semantic search parameters
            
        Returns:
            List of search results
        """
        # Get the semantic search tool
        tool = get_semantic_tool("semantic_search")
        
        # Configure the tool with the parameters
        tool.set_model(params.embedding_model)  # Use set_model instead of direct assignment
        tool.use_gpu = params.use_gpu
        tool.normalize_vectors = params.normalize_vectors
        
        # Index the documents
        tool.index_texts(
            texts=params.texts, 
            ids=params.ids, 
            metadata=params.metadata
        )
        
        # Perform the search
        results = tool.search(params.query, top_k=params.top_k)
        
        # Add backend info to the results
        backend_info = tool.get_backend_info()
        
        # Add additional metadata to each result
        for result in results:
            result["backend_info"] = backend_info
        
        return results


# Re-export the get_operation function to avoid circular imports
# This is needed by other modules that import from this module
from tsap.core.base import get_operation  # noqa: E402, F401, F811