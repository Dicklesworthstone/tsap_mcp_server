from tsap.core.base import BaseCoreTool, register_tool, ToolRegistry
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import traceback
import inspect

# Configure logging
logger = logging.getLogger(__name__)

# Try to import faiss, preferring GPU version but falling back to CPU version
try:
    # First try to import faiss-gpu
    import faiss
    try:
        # Check if it has GPU support
        if hasattr(faiss, "StandardGpuResources"):
            gpu_available = True
            logger.info("Successfully loaded faiss with GPU support")
        else:
            gpu_available = False
            logger.info("Loaded faiss but without GPU support")
    except Exception as e:
        gpu_available = False
        logger.info(f"Error checking faiss GPU support: {str(e)}")
except ImportError:
    # If faiss import fails completely, try to load the CPU version
    try:
        import faiss_cpu as faiss # type: ignore
        gpu_available = False
        logger.info("Loaded faiss-cpu as fallback")
    except ImportError:
        # If both fail, provide a more helpful error message
        logger.error("Failed to import faiss or faiss-cpu. Please install one of them: pip install faiss-cpu")
        raise ImportError("No faiss module found. Install with: pip install faiss-cpu")

def get_tool(name: str):
    """Get a tool instance by name.
    
    Args:
        name: Tool name
        
    Returns:
        Tool instance or None if not found
    """
    return ToolRegistry.get_tool(name)

def is_gpu_available():
    """Check if GPU is available for FAISS.
    
    Returns:
        bool: True if GPU is available, False otherwise
    """
    # First check our global flag from import time
    if not gpu_available:
        return False
    
    # Then do a deeper check to make sure GPU resources can actually be created
    try:
        # Try to create GPU resources
        res = faiss.StandardGpuResources()  # noqa: F841
        logger.info("Successfully created StandardGpuResources")
        
        # Check if we can get GPU count
        try:
            gpu_count = faiss.get_num_gpus()
            if gpu_count > 0:
                logger.info(f"FAISS detected {gpu_count} GPUs")
                return True
            else:
                logger.info("FAISS detected no GPUs - check CUDA configuration")
                return False
        except AttributeError:
            # If get_num_gpus not found but StandardGpuResources works, assume GPU is available
            return True
    except Exception as e:
        logger.info(f"Failed to create StandardGpuResources: {str(e)}")
        return False

@register_tool("semantic_search")
class SemanticSearchTool(BaseCoreTool):
    """Semantic search tool using vector embeddings and FAISS for approximate nearest neighbor search.
    
    This tool supports both CPU and GPU backends, automatically falling back to CPU if GPU is not available.
    It uses the SentenceTransformers library for generating text embeddings locally.
    """
    
    def __init__(self, 
                embedding_model: str = "nomic-ai/nomic-embed-text-v2-moe", 
                vector_dim: int = 768,
                use_gpu: bool = True,
                normalize_vectors: bool = True,
                batch_size: int = 32):
        """Initialize the semantic search tool.
        
        Args:
            embedding_model: Name of the embedding model to use
            vector_dim: Dimension of embedding vectors
            use_gpu: Whether to use GPU if available
            normalize_vectors: Whether to normalize vectors during embedding
            batch_size: Batch size for embedding processing
        """
        super().__init__("semantic_search")
        self.embedding_model_name = embedding_model
        self.vector_dim = vector_dim
        
        # Check if GPU is actually available and user wants to use it
        self.use_gpu = use_gpu and is_gpu_available()
        if use_gpu and not self.use_gpu:
            logger.info("GPU requested but not available - falling back to CPU")
        
        self.normalize_vectors = normalize_vectors
        self.batch_size = batch_size
        
        # Will be initialized when needed
        self.index = None
        self.doc_ids = []
        self.texts = []
        self.metadata = []
        self.embedding_model = None
        
        # Initialize the embedding model
        self._load_embedding_model()
        
        # Try to create the index immediately to fail fast if there are issues
        self._create_index()
    
    def _load_embedding_model(self):
        """Load the embedding model using SentenceTransformers."""
        try:
            # Set device based on GPU availability
            device = "cuda" if self.use_gpu else "cpu"
            
            # Log which model we're loading with detailed info
            logger.info(f"Loading embedding model {self.embedding_model_name} on {device}")
            
            # Detect if this is a Nomic model which might need special handling
            is_nomic_model = "nomic" in self.embedding_model_name.lower()
            if is_nomic_model:
                logger.info(f"Detected Nomic model: {self.embedding_model_name}, using specialized handling if needed")
            
            # Load model
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=device,
                trust_remote_code=True
            )
            
            # Inspect what encode method the model has
            has_encode_method = hasattr(self.embedding_model, 'encode')
            encode_signature = None
            if has_encode_method:
                try:
                    encode_signature = inspect.signature(self.embedding_model.__class__.encode)
                    logger.info(f"Model encode method signature: {encode_signature}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not inspect encode signature: {str(e)}")
            
            logger.info(f"Successfully loaded embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to load embedding model: {str(e)}")
    
    def _create_index(self):
        """Create a FAISS index, attempting to use GPU if available and requested.
        
        Returns:
            FAISS index instance
        """
        try:
            # Create a flat index using inner product (IP) similarity
            cpu_index = faiss.IndexFlatIP(self.vector_dim)
            
            # Only try GPU if specifically requested and available
            if self.use_gpu:
                try:
                    # Create GPU resources and move the index to GPU
                    gpu_resources = faiss.StandardGpuResources()
                    gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, cpu_index)
                    logger.info("Successfully created GPU FAISS index")
                    return gpu_index
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"Failed to create GPU index, falling back to CPU: {error_msg}")
                    # Reset use_gpu flag to avoid further attempts
                    self.use_gpu = False
            
            # If we get here, use CPU index
            logger.info("Using CPU FAISS index")
            return cpu_index
        except Exception as e:
            logger.error(f"Error creating FAISS index: {str(e)}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to create FAISS index: {str(e)}")
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the current backend.
        
        Returns:
            Dictionary with backend information
        """
        # Check if index is a GPU index
        is_gpu = False
        if self.index is not None:
            try:
                # Different ways to detect GPU index depending on FAISS version
                is_gpu = (hasattr(self.index, "getDevice") or 
                         "Gpu" in self.index.__class__.__name__)
            except Exception:
                pass
                
        return {
            "backend": "gpu" if is_gpu else "cpu",
            "gpu_available": is_gpu_available(),
            "embedding_model": self.embedding_model_name,
            "vector_dim": self.vector_dim,
            "normalize_vectors": self.normalize_vectors,
            "index_type": type(self.index).__name__ if self.index else None,
            "index_size": len(self.doc_ids),
        }
    
    def set_model(self, model_name: str):
        """Set and load the embedding model by name."""
        self.embedding_model_name = model_name
        self._load_embedding_model()

    def embed_texts(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Embed a list of texts into vector representations.

        Args:
            texts (List[str]): List of text strings to embed.
            is_query (bool): Whether the texts are queries (True) or documents (False).

        Returns:
            np.ndarray: Array of embedded vectors.
        """
        if not texts:
            return np.array([], dtype=np.float32)

        try:
            if self.embedding_model is None:
                self._load_embedding_model()

            # Log the type and method signature for debugging
            logger.info(f"Type of embedding_model: {type(self.embedding_model)}")
            if not hasattr(self.embedding_model, 'encode'):
                raise AttributeError("embedding_model does not have an 'encode' method")
            encode_method = self.embedding_model.encode
            logger.info(f"Encode method signature: {inspect.signature(encode_method)}")

            # Handle Nomic model specifically
            if "nomic-embed-text-v2-moe" in self.embedding_model_name:
                prefix = "search_query: " if is_query else "search_document: "
                prefixed_texts = [prefix + text for text in texts]
                embeddings = self.embedding_model.encode(prefixed_texts)
                if self.normalize_vectors:
                    from sklearn.preprocessing import normalize
                    embeddings = normalize(embeddings, axis=1)
            else:
                embeddings = self.embedding_model.encode(
                    texts,
                    normalize_embeddings=self.normalize_vectors
                )

            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error embedding texts: {str(e)}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to embed texts: {str(e)}")

    def index_texts(self, texts: List[str], ids: Optional[List[str]] = None, 
                    metadata: Optional[List[Dict[str, Any]]] = None):
        """Index a list of texts for semantic search.
        
        Args:
            texts: List of text strings to index
            ids: Optional list of IDs for the texts
            metadata: Optional list of metadata dictionaries for each text
        """
        if not texts:
            logger.warning("No texts provided for indexing")
            return
            
        # Create index if not already done
        if self.index is None:
            self.index = self._create_index()
        
        # Generate default IDs if not provided
        if ids is None:
            start_idx = len(self.doc_ids)
            ids = [f"doc_{i+start_idx}" for i in range(len(texts))]
        
        # Generate empty metadata if not provided
        if metadata is None:
            metadata = [{} for _ in range(len(texts))]
            
        # Ensure all lists have the same length
        if len(texts) != len(ids) or len(texts) != len(metadata):
            raise ValueError("texts, ids, and metadata must have the same length")
        
        # Embed the texts and add to the index
        vectors = self.embed_texts(texts, is_query=False)
        self.index.add(vectors)
        
        # Store the documents and metadata
        self.doc_ids.extend(ids)
        self.texts.extend(texts)
        self.metadata.extend(metadata)
        
        logger.info(f"Indexed {len(texts)} documents, index now contains {len(self.doc_ids)} documents")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for documents similar to the query.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with search results
        """
        if self.index is None or len(self.doc_ids) == 0:
            logger.warning("No documents indexed yet")
            return []
            
        # Limit top_k to the number of documents
        top_k = min(top_k, len(self.doc_ids))
        if top_k == 0:
            return []
            
        # Embed the query
        q_vector = self.embed_texts([query], is_query=True)
        
        # Search the index
        D, I = self.index.search(q_vector, top_k)  # noqa: E741
        
        # Format the results
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx >= 0 and idx < len(self.doc_ids):
                results.append({
                    "id": self.doc_ids[idx],
                    "text": self.texts[idx],
                    "score": float(score),
                    "metadata": self.metadata[idx] if idx < len(self.metadata) else {},
                })
        
        return results
    
    def reset_index(self):
        """Reset the index and clear all indexed documents."""
        self.index = self._create_index()
        self.doc_ids = []
        self.texts = []
        self.metadata = []
        logger.info("Index reset")
    
    def get_indexed_documents(self) -> Dict[str, Any]:
        """Get information about the indexed documents.
        
        Returns:
            Dictionary with index information
        """
        return {
            "count": len(self.doc_ids),
            "ids": self.doc_ids,
            "backend_info": self.get_backend_info(),
        }
