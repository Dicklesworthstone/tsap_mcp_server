#!/usr/bin/env python
"""
Semantic Search Demo Script

This script demonstrates the semantic search capabilities of TSAP.
It performs text embedding and similarity search using the FAISS backend.
"""

import sys
import os
import time
import argparse
from pprint import pprint
import asyncio
import json

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tsap.core.semantic_search_tool import get_tool as get_semantic_tool
from src.tsap.composite.semantic_search import SemanticSearchParams, SemanticSearchOperation

# Sample text dataset (Shakespeare quotes)
SAMPLE_TEXTS = [
    "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer "
    "The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles "
    "And by opposing end them.",
    
    "All the world's a stage, and all the men and women merely players. "
    "They have their exits and their entrances; And one man in his time plays many parts.",
    
    "The quality of mercy is not strained. It droppeth as the gentle rain from heaven "
    "Upon the place beneath. It is twice blest: It blesseth him that gives and him that takes.",
    
    "Friends, Romans, countrymen, lend me your ears; I come to bury Caesar, not to praise him. "
    "The evil that men do lives after them; The good is oft interred with their bones.",
    
    "Tomorrow, and tomorrow, and tomorrow, Creeps in this petty pace from day to day, "
    "To the last syllable of recorded time; And all our yesterdays have lighted fools "
    "The way to dusty death.",
    
    "What's in a name? That which we call a rose By any other name would smell as sweet.",
    
    "If music be the food of love, play on, Give me excess of it; that surfeiting, "
    "The appetite may sicken, and so die.",
    
    "The fool doth think he is wise, but the wise man knows himself to be a fool.",
    
    "Life's but a walking shadow, a poor player, That struts and frets his hour upon the stage, "
    "And then is heard no more. It is a tale Told by an idiot, full of sound and fury, Signifying nothing.",
    
    "Some are born great, some achieve greatness, and some have greatness thrust upon them."
]

# Sample metadata
SAMPLE_METADATA = [
    {"play": "Hamlet", "character": "Hamlet", "act": 3, "scene": 1},
    {"play": "As You Like It", "character": "Jaques", "act": 2, "scene": 7},
    {"play": "The Merchant of Venice", "character": "Portia", "act": 4, "scene": 1},
    {"play": "Julius Caesar", "character": "Mark Antony", "act": 3, "scene": 2},
    {"play": "Macbeth", "character": "Macbeth", "act": 5, "scene": 5},
    {"play": "Romeo and Juliet", "character": "Juliet", "act": 2, "scene": 2},
    {"play": "Twelfth Night", "character": "Duke Orsino", "act": 1, "scene": 1},
    {"play": "As You Like It", "character": "Touchstone", "act": 5, "scene": 1},
    {"play": "Macbeth", "character": "Macbeth", "act": 5, "scene": 5},
    {"play": "Twelfth Night", "character": "Malvolio", "act": 2, "scene": 5}
]

# Sample queries
SAMPLE_QUERIES = [
    "What is the meaning of life?",
    "How can I be wise?",
    "Tell me about love",
    "What is greatness?",
    "Should I take action against problems?"
]


def print_section(title: str):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


async def direct_tool_demo(use_gpu: bool = True):
    """Demo using the semantic search tool directly."""
    print_section("Direct Tool Usage Demo")
    
    # Get the semantic search tool
    tool = get_semantic_tool("semantic_search")
    tool.use_gpu = use_gpu
    
    # Index the documents
    print("Indexing sample texts...")
    start_time = time.time()
    
    # Generate IDs for the texts
    ids = [f"quote_{i+1}" for i in range(len(SAMPLE_TEXTS))]
    
    # Index the texts with metadata
    tool.index_texts(SAMPLE_TEXTS, ids, SAMPLE_METADATA)
    
    print(f"Indexing completed in {time.time() - start_time:.2f} seconds")
    
    # Print backend info
    backend_info = tool.get_backend_info()
    print("\nBackend Information:")
    pprint(backend_info)
    
    # Run searches for each query
    print("\nPerforming searches:")
    for i, query in enumerate(SAMPLE_QUERIES):
        print(f"\nQuery {i+1}: '{query}'")
        
        start_time = time.time()
        results = tool.search(query, top_k=3)
        search_time = time.time() - start_time
        
        print(f"Found {len(results)} results in {search_time:.4f} seconds:")
        for j, result in enumerate(results):
            print(f"  Result {j+1}: {result['text'][:80]}... (score: {result['score']:.4f})")
            print(f"    Metadata: {result['metadata']}")


async def composite_operation_demo(use_gpu: bool = True):
    """Demo using the composite semantic search operation."""
    print_section("Composite Operation Demo")
    
    # Create the operation object
    op = SemanticSearchOperation("semantic_search")
    
    # Create parameters
    params = SemanticSearchParams(
        texts=SAMPLE_TEXTS,
        query="What is the meaning of existence?",
        ids=[f"quote_{i+1}" for i in range(len(SAMPLE_TEXTS))],
        metadata=SAMPLE_METADATA,
        top_k=5,
        use_gpu=use_gpu,
        embedding_model="nomic-ai/nomic-embed-text-v2-moe"
    )
    
    # Execute the operation
    print("Executing semantic search operation...")
    start_time = time.time()
    results = await op.execute_with_stats(params)
    
    print(f"Operation completed in {time.time() - start_time:.2f} seconds")
    
    # Display the results
    print("\nResults:")
    for i, result in enumerate(results):
        print(f"\nResult {i+1}: (score: {result['score']:.4f})")
        print(f"  Text: {result['text'][:100]}...")
        print(f"  Metadata: {result['metadata']}")
    
    # Display the backend info
    if results and 'backend_info' in results[0]:
        print("\nBackend Information:")
        pprint(results[0]['backend_info'])


async def custom_dataset_demo(dataset_file: str, query: str, use_gpu: bool = True):
    """Demo using a custom dataset from a file."""
    print_section(f"Custom Dataset Demo: {dataset_file}")
    
    # Load the dataset
    try:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            if dataset_file.endswith('.json'):
                # Assuming JSON file with a list of objects with 'text' field
                data = json.load(f)
                if isinstance(data, list):
                    texts = [item.get('text', str(item)) for item in data]
                    metadata = [item for item in data]
                else:
                    texts = [data.get('text', str(data))]
                    metadata = [data]
            else:
                # Assuming text file with one document per line
                texts = [line.strip() for line in f if line.strip()]
                metadata = [{"source": dataset_file, "line": i+1} for i in range(len(texts))]
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    # Get the semantic search tool
    tool = get_semantic_tool("semantic_search")
    tool.use_gpu = use_gpu
    
    # Index the documents
    print(f"Indexing {len(texts)} documents...")
    start_time = time.time()
    
    # Generate IDs for the texts
    ids = [f"doc_{i+1}" for i in range(len(texts))]
    
    # Index the texts with metadata
    tool.index_texts(texts, ids, metadata)
    
    print(f"Indexing completed in {time.time() - start_time:.2f} seconds")
    
    # Print backend info
    backend_info = tool.get_backend_info()
    print("\nBackend Information:")
    pprint(backend_info)
    
    # Run search
    print(f"\nSearching for: '{query}'")
    start_time = time.time()
    results = tool.search(query, top_k=5)
    search_time = time.time() - start_time
    
    print(f"Found {len(results)} results in {search_time:.4f} seconds:")
    for i, result in enumerate(results):
        print(f"\nResult {i+1}: (score: {result['score']:.4f})")
        print(f"  Text: {result['text'][:100]}...")
        if 'metadata' in result:
            print(f"  Metadata: {result['metadata']}")


async def performance_benchmark(use_gpu: bool = True, num_docs: int = 1000, num_queries: int = 10):
    """Performance benchmark for semantic search."""
    print_section(f"Performance Benchmark: {num_docs} documents, {num_queries} queries")
    
    # Generate synthetic dataset
    print(f"Generating {num_docs} synthetic documents...")
    texts = []
    for i in range(num_docs):
        # Generate a synthetic document with some random words
        import random
        words = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit",
                "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore", "et", "dolore",
                "magna", "aliqua", "enim", "ad", "minim", "veniam", "quis", "nostrud", "exercitation"]
        
        # Generate random document of 20-50 words
        doc_len = random.randint(20, 50)
        doc = " ".join(random.choice(words) for _ in range(doc_len))
        texts.append(doc)
    
    # Generate random queries
    queries = []
    for i in range(num_queries):
        # Generate random query of 3-7 words
        query_len = random.randint(3, 7)
        query = " ".join(random.choice(words) for _ in range(query_len))
        queries.append(query)
    
    # Get the semantic search tool
    tool = get_semantic_tool("semantic_search")
    tool.use_gpu = use_gpu
    
    # Index the documents
    print(f"Indexing {len(texts)} documents...")
    index_start_time = time.time()
    
    # Generate IDs for the texts
    ids = [f"doc_{i+1}" for i in range(len(texts))]
    
    # Index the texts
    tool.index_texts(texts, ids)
    
    index_time = time.time() - index_start_time
    print(f"Indexing completed in {index_time:.2f} seconds ({num_docs / index_time:.2f} docs/sec)")
    
    # Print backend info
    backend_info = tool.get_backend_info()
    print("\nBackend Information:")
    pprint(backend_info)
    
    # Run searches for each query
    print("\nPerforming searches:")
    total_search_time = 0
    for i, query in enumerate(queries):
        start_time = time.time()
        results = tool.search(query, top_k=10)
        search_time = time.time() - start_time
        total_search_time += search_time
        
        print(f"Query {i+1}: Found {len(results)} results in {search_time:.4f} seconds")
    
    avg_search_time = total_search_time / num_queries
    print(f"\nAverage search time: {avg_search_time:.4f} seconds ({1/avg_search_time:.2f} queries/sec)")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Semantic Search Demo")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--dataset", type=str, help="Path to custom dataset file")
    parser.add_argument("--query", type=str, default="What is the meaning of life?", 
                       help="Query for custom dataset search")
    parser.add_argument("--demo-type", type=str, choices=["direct", "composite", "all"], 
                       default="all", help="Type of demo to run")
    
    args = parser.parse_args()
    use_gpu = not args.no_gpu
    
    print(f"Starting Semantic Search Demo (GPU {'disabled' if args.no_gpu else 'enabled'})")
    
    if args.benchmark:
        await performance_benchmark(use_gpu)
        return
    
    if args.dataset:
        await custom_dataset_demo(args.dataset, args.query, use_gpu)
        return
    
    if args.demo_type in ["direct", "all"]:
        await direct_tool_demo(use_gpu)
    
    if args.demo_type in ["composite", "all"]:
        await composite_operation_demo(use_gpu)


if __name__ == "__main__":
    asyncio.run(main()) 