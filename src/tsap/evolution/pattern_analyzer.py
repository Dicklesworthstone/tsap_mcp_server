"""
Pattern analyzer for TSAP's evolution system.

This module provides functionality to analyze and optimize search patterns,
learning from previous results to improve future searches.
"""
import re
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
import statistics
import json
import uuid
import asyncio
import httpx
import logging

from tsap.utils.logging import logger
from tsap.core.ripgrep import ripgrep_search
from tsap.mcp.models import RipgrepSearchParams

# Define a simple MCPClient class for internal use
class MCPClient:
    """Simple client for interacting with MCP Servers."""

    def __init__(self, base_url: str = "http://localhost:8013"):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(
            base_url=self.base_url, 
            headers=self.headers, 
            timeout=60.0
        )

    async def close(self):
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def send_request(self, command: str, args: Dict[str, Any], mode: Optional[str] = None) -> Dict[str, Any]:
        """Send an MCP request to the server."""
        # Create an MCP request payload
        request = {
            "request_id": str(uuid.uuid4()),
            "command": command,
            "args": args,
        }
        
        if mode:
            request["mode"] = mode
            
        try:
            response = await self._client.post("/mcp/", json=request)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error in MCP request: {e}", component="MCPClient")
            return {"error": {"code": "CLIENT_ERROR", "message": str(e)}}

# Check if LLM pattern generation is enabled
USE_LLM_PATTERN_GENERATION = os.environ.get("USE_LLM_PATTERN_GENERATION", "").lower() in ("true", "1", "yes")
LLM_MCP_SERVER_URL = os.environ.get("LLM_MCP_SERVER_URL", "http://localhost:8013")


async def generate_patterns_with_llm(
    pattern: str,
    description: str,
    reference_set: Dict[str, List[str]],
    num_variants: int = 5
) -> List[str]:
    """Generate regex pattern variants using an LLM through MCP.
    
    Args:
        pattern: The original regex pattern
        description: Description of what the pattern is intended to match
        reference_set: Dictionary containing 'positive' and 'negative' example strings
        num_variants: Number of pattern variants to generate
        
    Returns:
        List of generated regex pattern variants
    """
    try:
        # Connect to the LLM MCP Server
        async with MCPClient(base_url=LLM_MCP_SERVER_URL) as client:
            positive_examples = reference_set.get("positive", [])
            negative_examples = reference_set.get("negative", [])
            
            # Create a prompt that instructs the LLM to generate regex patterns
            prompt = f"""Generate {num_variants} improved regular expression patterns based on the following information:

Original regex pattern: {pattern}
Description: {description}

The regex should match all positive examples but not match any negative examples if possible.

Positive examples (should match):
{json.dumps(positive_examples, indent=2)}

Negative examples (should NOT match):
{json.dumps(negative_examples, indent=2)}

Provide your response as a JSON array of regex patterns only, with no explanation. For example:
["pattern1", "pattern2", "pattern3"]
"""
            
            # Call the LLM using the semantic_search endpoint
            response = await client.send_request("completions", {
                "prompt": prompt,
                "max_tokens": 1000,
                "temperature": 0.7,
                "stop": ["\n\n"],
                "model": "default"  # Use default model configured for the gateway
            })
            
            # Extract pattern variants from response
            if "data" in response and "text" in response["data"]:
                generated_text = response["data"]["text"].strip()
                
                # Extract JSON array portion
                try:
                    # Try to find a JSON array in the response
                    start_idx = generated_text.find("[")
                    end_idx = generated_text.rfind("]") + 1
                    
                    if start_idx >= 0 and end_idx > start_idx:
                        json_array = generated_text[start_idx:end_idx]
                        pattern_variants = json.loads(json_array)
                        
                        # Validate patterns
                        valid_patterns = []
                        for var_pattern in pattern_variants:
                            try:
                                # Verify it's a valid regex
                                re.compile(var_pattern)
                                valid_patterns.append(var_pattern)
                            except re.error as e:
                                logger.warning(
                                    f"LLM generated invalid regex: {var_pattern}. Error: {e}",
                                    component="PatternAnalyzer"
                                )
                        
                        # If we have valid patterns, return them
                        if valid_patterns:
                            # Make sure we don't have duplicate patterns
                            unique_patterns = list(set(valid_patterns))
                            # Also ensure original pattern isn't included
                            if pattern in unique_patterns:
                                unique_patterns.remove(pattern)
                            return unique_patterns
                    
                    # If JSON parsing fails or no valid patterns, log and continue with rule-based generation
                    logger.warning(
                        f"Failed to extract valid regex patterns from LLM response: {generated_text}",
                        component="PatternAnalyzer"
                    )
                except json.JSONDecodeError:
                    logger.warning(
                        f"Failed to parse JSON from LLM response: {generated_text}",
                        component="PatternAnalyzer"
                    )
    except Exception as e:
        logger.error(
            f"Error using LLM for pattern generation: {e}",
            exc_info=True,
            component="PatternAnalyzer"
        )
    
    # Return empty list if LLM generation fails
    return []

@dataclass
class PatternStats:
    """Statistics for a search pattern."""
    
    pattern: str
    total_matches: int = 0
    files_with_matches: int = 0
    match_locations: List[Tuple[str, int]] = field(default_factory=list)
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 1.0
    recall: float = 1.0
    f1_score: float = 1.0
    avg_context_relevance: float = 1.0
    execution_time: float = 0.0
    
    def calculate_metrics(self, total_relevant: int = 0):
        """Calculate precision, recall, and F1 score.
        
        Args:
            total_relevant: Total number of relevant items
        """
        true_positives = self.total_matches - self.false_positives
        
        # Calculate precision
        if self.total_matches > 0:
            self.precision = true_positives / self.total_matches
        else:
            self.precision = 1.0  # By convention, precision is 1.0 when no matches
        
        # Calculate recall
        if total_relevant > 0:
            self.recall = true_positives / total_relevant
        else:
            self.recall = 1.0  # By convention, recall is 1.0 when no relevant items
        
        # Calculate F1 score
        if self.precision > 0 or self.recall > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        else:
            self.f1_score = 0.0


@dataclass
class PatternVariant:
    """A variant of a search pattern with its performance metrics."""
    
    pattern: str
    description: str
    is_regex: bool = True
    case_sensitive: bool = False
    stats: PatternStats = None
    generation: int = 0
    parent_pattern: Optional[str] = None
    
    def __post_init__(self):
        if not self.stats:
            self.stats = PatternStats(pattern=self.pattern)
    
    @property
    def score(self) -> float:
        """Calculate an overall score for this pattern variant.
        
        Returns:
            Score value (higher is better)
        """
        # Weighted combination of metrics
        # You can adjust weights based on your priorities
        weights = {
            "precision": 0.4,
            "recall": 0.4,
            "f1": 0.2,
        }
        
        return (
            weights["precision"] * self.stats.precision +
            weights["recall"] * self.stats.recall +
            weights["f1"] * self.stats.f1_score
        )


class PatternAnalyzer:
    """Analyzer for search patterns with evolutionary capabilities."""
    
    def __init__(self):
        """Initialize the pattern analyzer."""
        self.pattern_variants = {}  # Maps pattern ID to list of variants
        self.pattern_stats = {}  # Maps pattern to PatternStats
        self.pattern_history = {}  # Maps pattern ID to history of variants
    
    @staticmethod
    def _extract_pattern_components(pattern: str) -> List[str]:
        """Extract components from a regex pattern.
        
        Args:
            pattern: Regex pattern
            
        Returns:
            List of pattern components
        """
        # Simple component extraction for common regex constructs
        components = []
        
        # Extract literals (text not in special regex constructs)
        literal_parts = re.split(r'[\\[\](){}?*+^$.|]', pattern)
        components.extend([p for p in literal_parts if p])
        
        # Extract character classes
        char_classes = re.findall(r'\[[^\]]+\]', pattern)
        components.extend(char_classes)
        
        # Extract groups
        groups = re.findall(r'\([^)]+\)', pattern)
        components.extend(groups)
        
        return components
    
    @staticmethod
    def _generate_simpler_pattern(pattern: str) -> str:
        """Generate a simpler version of a regex pattern.
        
        Args:
            pattern: Original pattern
            
        Returns:
            Simplified pattern
        """
        # Check if pattern is None and return empty string
        if pattern is None:
            return ""
            
        # Replace complex regex constructs with simpler alternatives
        simplified = pattern
        
        # Replace character classes with wildcards
        simplified = re.sub(r'\[[^\]]+\]', '.', simplified)
        
        # Replace non-capturing groups with their content
        simplified = re.sub(r'\(\?:[^)]+\)', '.+', simplified)
        
        # Replace capturing groups with their content
        simplified = re.sub(r'\([^)]+\)', '.+', simplified)
        
        # Replace complex quantifiers
        simplified = re.sub(r'\{\d+,\d+\}', '+', simplified)
        simplified = re.sub(r'\{\d+\}', '+', simplified)
        
        # Replace word boundaries with spaces
        simplified = simplified.replace('\\b', ' ')
        
        # Remove anchors
        simplified = simplified.replace('^', '')
        simplified = simplified.replace('$', '')
        
        return simplified
    
    @staticmethod
    def _generate_more_specific_pattern(pattern: str) -> str:
        """Generate a more specific version of a pattern.
        
        Args:
            pattern: Original pattern
            
        Returns:
            More specific pattern
        """
        specific = pattern
        original_pattern_for_error_return = pattern # Keep original for fallback

        # Attempt to add word boundaries more carefully
        try:
            potential_words = list(re.finditer(r'\w+', specific))
            added_boundaries = 0
            offset = 0
            
            current_specific = list(specific) # Work with list for easier modification

            for match in potential_words:
                word = match.group(0)
                start, end = match.span()
                current_start = start + offset
                current_end = end + offset

                if len(word) > 2:
                    # Check context directly in the current state of the pattern (as a string)
                    temp_specific_str = "".join(current_specific)
                    precedes_boundary = (current_start > 1 and temp_specific_str[current_start-2:current_start] == '\\b')
                    follows_boundary = (current_end < len(temp_specific_str) -1 and temp_specific_str[current_end:current_end+2] == '\\b')
                    
                    inside_brackets = False
                    # Rough check for brackets - might need refinement for nested/escaped cases
                    open_bracket_idx = temp_specific_str.rfind('[', 0, current_start)
                    if open_bracket_idx != -1:
                        # Check if the corresponding closing bracket is after our word end
                        # This simple check might fail for complex nested brackets
                        corresponding_close = temp_specific_str.find(']', open_bracket_idx)
                        if corresponding_close != -1 and corresponding_close >= current_end:
                             # Further check: ensure no closing bracket between open and start
                             if temp_specific_str.find(']', open_bracket_idx, current_start) == -1:
                                 inside_brackets = True 

                    if not precedes_boundary and not follows_boundary and not inside_brackets:
                        replacement = f'\\b{word}\\b'
                        # Modify the list representation
                        current_specific[current_start:current_end] = list(replacement)
                        offset += len(replacement) - len(word)
                        added_boundaries += 1
           
            specific = "".join(current_specific) # Convert back to string
            if added_boundaries > 0:
                 logger.debug(f"Added word boundaries, intermediate pattern: {specific}", component="PatternAnalyzer")

        except re.error as e:
            logger.warning(f"Regex error during word boundary addition for '{pattern}': {e}. Skipping boundary addition.", component="PatternAnalyzer")
            specific = original_pattern_for_error_return # Revert to original pattern if boundary logic failed
        except Exception as e: # Catch potential index errors too
             logger.warning(f"Unexpected error during word boundary addition for '{pattern}': {e}. Skipping boundary addition.", component="PatternAnalyzer")
             specific = original_pattern_for_error_return

        # Replace wildcards (unescaped dots) with more specific character classes
        try:
            # Use re.sub to avoid replacing escaped dots (\\.)
            specific = re.sub(r'(?<!\\)\.', '[a-zA-Z0-9_]', specific)
        except re.error as e:
             logger.warning(f"Regex error during wildcard replacement for '{pattern}': {e}. Skipping wildcard replacement.", component="PatternAnalyzer")
             # Keep the pattern state before this step

        # --- Removed unconditional start/end anchor addition --- 
        
        # Final check: Ensure the generated pattern is still valid regex
        try:
            re.compile(specific)
        except re.error as e:
            logger.warning(f"Generated specific pattern '{specific}' is invalid regex: {e}. Returning original.", component="PatternAnalyzer")
            return original_pattern_for_error_return # Return original if specific version is invalid
            
        # Return the modified pattern only if it's different and valid
        return specific if specific != original_pattern_for_error_return else original_pattern_for_error_return
    
    @staticmethod
    def _generate_pattern_variations(pattern: str, is_regex: bool) -> List[str]:
        """Generate variations of a pattern using simpler/specific versions and mutations.
        
        Args:
            pattern: Original pattern
            is_regex: Whether the pattern is a regex
            
        Returns:
            List of pattern variations
        """
        variations = set() # Use a set to avoid duplicates initially
        
        if is_regex:
            # 1. Generate simpler version
            try:
                simpler = PatternAnalyzer._generate_simpler_pattern(pattern)
                if simpler != pattern:
                    re.compile(simpler) # Check validity
                    variations.add(simpler)
            except re.error: pass

            # 2. Generate more specific version
            try:
                specific = PatternAnalyzer._generate_more_specific_pattern(pattern)
                if specific != pattern:
                    re.compile(specific) # Check validity
                    variations.add(specific)
            except re.error: pass

            # 3. Targeted Mutations
            mutations = []
            #   a) Swap .* and .+
            if '.*' in pattern: mutations.append(pattern.replace('.*', '.+', 1))
            if '.+' in pattern: mutations.append(pattern.replace('.+', '.*', 1))
            
            #   b) Add/Remove specific word boundaries (example: around 'ERROR')
            if '\\bERROR\\b' in pattern: 
                mutations.append(pattern.replace('\\bERROR\\b', 'ERROR', 1))
            elif 'ERROR' in pattern and not pattern.startswith('ERROR'): # Avoid adding boundary if starts with word
                 # Check if ERROR is preceded by non-word char or start of string
                 if re.search(r'(?<!\\w)ERROR', pattern): 
                     mutations.append(pattern.replace('ERROR', '\\bERROR\\b', 1))
                     
            #   c) Change . to \s or \w (if present)
            if '.' in pattern: 
                mutations.append(pattern.replace('.', '\\s', 1)) # Replace first . with whitespace
                mutations.append(pattern.replace('.', '\\w', 1)) # Replace first . with word char

            # Add valid, different mutations to the set
            for mut in mutations:
                if mut != pattern:
                    try:
                        re.compile(mut) # Check validity
                        variations.add(mut)
                    except re.error: pass
        else:
            # For literal patterns (unchanged)
            words = pattern.split()
            for i, word in enumerate(words):
                if len(word) > 3:
                    for j in range(len(word)):
                        varied_word = word[:j] + '.' + word[j+1:]
                        varied_words = words.copy()
                        varied_words[i] = varied_word
                        variations.add(' '.join(varied_words))
        
        variations.discard(pattern)
        return list(variations)
    
    @staticmethod
    def _mutate_pattern(
        pattern: str, 
        is_regex: bool, 
        mutation_strength: float = 0.2
    ) -> str:
        """Mutate a pattern to create a new variant.
        
        Args:
            pattern: Original pattern
            is_regex: Whether the pattern is a regex
            mutation_strength: Strength of mutation (0.0-1.0)
            
        Returns:
            Mutated pattern
        """
        if mutation_strength <= 0:
            return pattern
            
        # Randomly apply mutations based on mutation_strength
        mutated = pattern
        
        # For regex patterns
        if is_regex:
            # Possible mutations
            mutations = [
                # Replace a dot with a character class
                (r'\.', lambda m: '[a-zA-Z0-9_]'),
                # Replace a character class with a dot
                (r'\[[^\]]+\]', lambda m: '.'),
                # Add/remove word boundary
                (r'\\b', lambda m: ''),
                (r'[a-zA-Z0-9_]+', lambda m: f'\\b{m.group(0)}\\b'),
                # Modify quantifiers
                (r'\+', lambda m: '*'),
                (r'\*', lambda m: '+'),
                (r'\{\d+\}', lambda m: '+'),
                # Add/remove start/end anchors
                (r'^\^', lambda m: ''),
                (r'\$$', lambda m: ''),
            ]
            
            # Apply mutations with probability based on mutation_strength
            import random
            for pattern, replacement in mutations:
                if random.random() < mutation_strength:
                    # Find all matches
                    matches = list(re.finditer(pattern, mutated))
                    if matches:
                        # Select a random match
                        match = random.choice(matches)
                        # Apply the replacement
                        mutated = (
                            mutated[:match.start()] + 
                            replacement(match) + 
                            mutated[match.end():]
                        )
        else:
            # For literal patterns, make small changes
            import random
            
            # Possible mutations
            if random.random() < mutation_strength:
                # Change a random character
                if len(mutated) > 0:
                    pos = random.randint(0, len(mutated) - 1)
                    mutated = mutated[:pos] + '.' + mutated[pos+1:]
        
        return mutated
    
    @staticmethod
    def _crossover_patterns(
        pattern1: str, 
        pattern2: str, 
        is_regex: bool
    ) -> str:
        """Create a new pattern by combining two patterns.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            is_regex: Whether the patterns are regexes
            
        Returns:
            Combined pattern
        """
        if is_regex:
            # For regex patterns, try to combine meaningful parts
            components1 = PatternAnalyzer._extract_pattern_components(pattern1)
            components2 = PatternAnalyzer._extract_pattern_components(pattern2)
            
            # Combine components from both patterns
            import random
            
            # Select components from each pattern
            selected = []
            for comp in components1:
                if random.random() < 0.5:
                    selected.append(comp)
            for comp in components2:
                if random.random() < 0.5:
                    selected.append(comp)
                    
            # Ensure we have at least one component
            if not selected and components1 and components2:
                selected = [random.choice(components1), random.choice(components2)]
            elif not selected and components1:
                selected = [random.choice(components1)]
            elif not selected and components2:
                selected = [random.choice(components2)]
                
            # Join components with a reasonable separator
            if all(re.match(r'^[a-zA-Z0-9_]+$', comp) for comp in selected):
                # For word components, join with spaces or pipes
                if random.random() < 0.5:
                    return ' '.join(selected)
                else:
                    return '|'.join(selected)
            else:
                # For regex components, join with a regex alternation
                return '|'.join(selected)
        else:
            # For literal patterns, combine words
            words1 = pattern1.split()
            words2 = pattern2.split()
            
            # Take words from both patterns
            import random
            combined = []
            
            # Select words from each pattern
            for word in words1:
                if random.random() < 0.5:
                    combined.append(word)
            for word in words2:
                if random.random() < 0.5:
                    combined.append(word)
                    
            # Ensure we have at least one word
            if not combined:
                if words1 and words2:
                    combined = [random.choice(words1), random.choice(words2)]
                elif words1:
                    combined = [random.choice(words1)]
                elif words2:
                    combined = [random.choice(words2)]
                    
            # Join the words
            return ' '.join(combined)
    
    async def evaluate_pattern(
        self,
        pattern: str,
        is_regex: bool,
        case_sensitive: bool,
        paths: List[str],
        reference_set: Optional[Dict[str, List[str]]] = None
    ) -> PatternStats:
        """Evaluate a pattern's performance against files or a reference set.
        
        Args:
            pattern: The pattern string to evaluate.
            is_regex: Whether the pattern is a regular expression.
            case_sensitive: Whether the pattern matching should be case-sensitive.
            paths: List of file paths to search (used if reference_set is None or file-based).
            reference_set: An optional dictionary containing 'positive' and 'negative'
                        lists of example strings to evaluate against. If provided,
                        metrics are calculated based on these examples. Otherwise,
                        metrics might be calculated based on file search results
                        (though this part needs refinement).
            
        Returns:
            PatternStats containing performance metrics.
        """
        start_time = time.time()
        
        # Check for None pattern
        if pattern is None:
            logger.error(
                "Error evaluating pattern 'None': first argument must be string or compiled pattern",
                component="patternanalyzer",
                operation="evaluate_pattern"
            )
            # Return default stats for None pattern
            return PatternStats(pattern="None")
            
        stats = PatternStats(pattern=pattern)
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        total_relevant = 0 # Total positive examples
        
        try:
            # Compile regex pattern
            flags = 0 if case_sensitive else re.IGNORECASE
            if not is_regex:
                pattern = re.escape(pattern) # Treat non-regex as literal strings
            
            compiled_pattern = re.compile(pattern, flags)

            if reference_set and isinstance(reference_set, dict):
                positive_examples = reference_set.get("positive", [])
                negative_examples = reference_set.get("negative", [])
                total_relevant = len(positive_examples)

                # Evaluate against positive examples
                logger.debug(f"Evaluating pattern against positive examples: '{pattern}'", component="PatternAnalyzer", operation="evaluate_pattern")
                for i, example in enumerate(positive_examples):
                    match = compiled_pattern.search(example)
                    if match:
                        true_positives += 1
                        logger.debug(f"  Pos[{i}] MATCH: '{example}'", component="PatternAnalyzer", operation="evaluate_pattern")
                    else:
                        false_negatives += 1
                        logger.debug(f"  Pos[{i}] NO MATCH: '{example}'", component="PatternAnalyzer", operation="evaluate_pattern")
                
                # Evaluate against negative examples
                logger.debug(f"Evaluating pattern against negative examples: '{pattern}'", component="PatternAnalyzer", operation="evaluate_pattern")
                for i, example in enumerate(negative_examples):
                    match = compiled_pattern.search(example)
                    if match:
                        false_positives += 1
                        logger.debug(f"  Neg[{i}] MATCH (FP): '{example}'", component="PatternAnalyzer", operation="evaluate_pattern")
                    else:
                        logger.debug(f"  Neg[{i}] NO MATCH: '{example}'", component="PatternAnalyzer", operation="evaluate_pattern")
                        
                # Update stats based on example evaluation
                stats.total_matches = true_positives + false_positives # Total times the pattern matched anything in the reference set
                stats.false_positives = false_positives
                # Note: stats.false_negatives is not directly used by calculate_metrics, 
                # it uses total_relevant and true_positives instead.
                
                # We don't have file/line locations when evaluating examples directly
                stats.files_with_matches = 0 
                stats.match_locations = []

            else:
                # --- Original file-based evaluation (kept for potential future use, but needs review) ---
                # This part assumes reference_set is List[Tuple[str, int]] if not the dict type.
                # This logic might need significant rework if file-based evaluation against 
                # a ground-truth reference set is required alongside string-based evaluation.
                logger.warning(
                    "File-based evaluation with ground truth locations is not fully implemented "
                    "alongside string-based reference sets. Metrics might be inaccurate.",
                    component="PatternAnalyzer",
                    operation="evaluate_pattern"
                )

                # Example: Use ripgrep if paths are provided (needs reference_set adaptation)
                if paths:
                    search_params = {
                        "pattern": pattern,
                        "paths": paths,
                        "regex": is_regex, # Use the original is_regex flag
                        "case_sensitive": case_sensitive,
                        "stats": True, # Request stats from ripgrep
                        "json_output": True, # Need JSON for parsing
                    }
                    rg_result = await ripgrep_search(**search_params)
                    
                    if rg_result and "data" in rg_result:
                        rg_matches = rg_result["data"].get("matches", [])
                        rg_stats = rg_result["data"].get("stats", {})
                        
                        stats.total_matches = rg_stats.get("total_matches", len(rg_matches))
                        stats.files_with_matches = rg_stats.get("files_with_matches", 0)
                        stats.match_locations = [
                            (match.get("path", "N/A"), match.get("line_number", 0)) 
                            for match in rg_matches
                        ]
                        
                        # Placeholder: False positive/negative calculation for file search needs ground truth
                        # If reference_set was meant to be file locations:
                        # reference_locations = set(reference_set) if reference_set else set()
                        # for match in rg_matches:
                        #     match_loc = (match.get("path"), match.get("line_number"))
                        #     if reference_locations and match_loc not in reference_locations:
                        #         stats.false_positives += 1
                        # total_relevant = len(reference_locations) if reference_locations else 0
                        # current_true_positives = stats.total_matches - stats.false_positives
                        # false_negatives = total_relevant - current_true_positives

                # Fallback if no reference set and no paths or ripgrep fails
                else:
                    stats.total_matches = 0
                    stats.files_with_matches = 0
                    stats.false_positives = 0
                    total_relevant = 0 # Cannot determine relevance without ground truth
                    
                    # Assign default metrics if no evaluation possible
                    stats.precision = 0.0
                    stats.recall = 0.0
                    stats.f1_score = 0.0
                    stats.execution_time = time.time() - start_time
                    return stats # Return early as metrics can't be calculated

            # Calculate metrics using the dedicated method
            stats.calculate_metrics(total_relevant=total_relevant)

        except re.error as e:
            logger.error(
                f"Invalid regex pattern: {pattern}. Error: {e}",
                component="PatternAnalyzer",
                operation="evaluate_pattern",
                exception=e
            )
            # Set stats to indicate failure
            stats.precision = 0.0
            stats.recall = 0.0
            stats.f1_score = 0.0
            stats.total_matches = 0
            stats.false_positives = 0
        except Exception as e:
            logger.error(
                f"Error evaluating pattern '{pattern}': {e}",
                exception=e,
                component="PatternAnalyzer",
                operation="evaluate_pattern"
            )
            stats.precision = 0.0
            stats.recall = 0.0
            stats.f1_score = 0.0

        stats.execution_time = time.time() - start_time
        logger.debug(
            f"Evaluated pattern '{pattern}'. "
            f"TP:{true_positives}, FP:{false_positives}, FN:{false_negatives}, TotalRelevant:{total_relevant}. "
            f"Precision: {stats.precision:.4f}, Recall: {stats.recall:.4f}, F1: {stats.f1_score:.4f}. "
            f"Time: {stats.execution_time:.3f}s",
            component="PatternAnalyzer",
            operation="evaluate_pattern"
        )
        return stats
    
    async def analyze_pattern(
        self,
        pattern: str,
        description: str,
        is_regex: bool,
        case_sensitive: bool,
        paths: List[str],
        reference_set: Optional[Dict[str, List[str]]] = None,
        generate_variants: bool = True,
        num_variants: int = 3,
    ) -> Dict[str, Any]:
        """Analyze a pattern and generate variations.
        
        Args:
            pattern: Pattern to analyze
            description: Description of the pattern
            is_regex: Whether the pattern is a regex
            case_sensitive: Whether the pattern is case-sensitive
            paths: Paths to search
            reference_set: Optional dictionary with 'positive' and 'negative' example strings.
            generate_variants: Whether to generate variants
            num_variants: Number of variants to generate
            
        Returns:
            Analysis results
        """
        operation_start_time = time.time()
        logger.info(
            f"Analyzing pattern: {pattern}",
            component="PatternAnalyzer",
            context={
                "is_regex": is_regex,
                "case_sensitive": case_sensitive,
                "paths_count": len(paths) if paths else 0,
            }
        )
        
        # Initialize variant_patterns list and best_pattern_in_variants
        variant_patterns = []
        best_pattern_in_variants = pattern
        
        # Validate pattern input
        if pattern is None or pattern.strip() == "":
            logger.warning("Empty or None pattern provided to analyze_pattern", component="PatternAnalyzer")
            return {
                "error": "Pattern is empty or None",
                "status": "error",
                "pattern": pattern,
                "variations": [],
                "stats": None,
                "execution_time": 0.0
            }
            
        # Create a pattern ID if we want to track history
        pattern_id = str(uuid.uuid4())
        
        # If LLM pattern generation is enabled and we have reference sets, use LLM
        llm_used = False
        rule_based_variations = []
        llm_variations = []
        
        # First evaluate the original pattern
        original_stats = await self.evaluate_pattern(
            pattern=pattern,
            is_regex=is_regex,
            case_sensitive=case_sensitive,
            paths=paths,
            reference_set=reference_set,
        )
        
        # Store the stats
        self.pattern_stats[pattern] = original_stats
        
        # Log the operation
        logger.info(
            f"Analyzing pattern: {pattern}",
            component="evolution",
            operation="analyze_pattern",
            context={
                "pattern_id": pattern_id,
                "pattern": pattern,
                "is_regex": is_regex,
                "generate_variants": generate_variants,
                "llm_enabled": USE_LLM_PATTERN_GENERATION,
                "reference_set_type": type(reference_set).__name__ if reference_set else None
            }
        )
        
        # Initialize variant list if needed
        if pattern_id not in self.pattern_variants:
            self.pattern_variants[pattern_id] = []
            self.pattern_history[pattern_id] = []
        
        # Create the base variant
        base_variant = PatternVariant(
            pattern=pattern,
            description=description,
            is_regex=is_regex,
            case_sensitive=case_sensitive,
            stats=original_stats,
        )
        
        # Add to variants and history
        self.pattern_variants[pattern_id].append(base_variant)
        self.pattern_history[pattern_id].append(base_variant)
        
        # Generate variants if requested
        if generate_variants:
            variant_patterns = []
            llm_used = False

            # Use LLM for generation if we have a reference set and description
            if reference_set and description and hasattr(self, "_generate_llm_variations"):
                try:
                    llm_used = True
                    llm_variations = await generate_patterns_with_llm(pattern, description, reference_set, num_variants)
                    logger.info(
                        f"LLM generation produced {len(llm_variations)} pattern variants",
                        component="PatternAnalyzer",
                        operation="analyze_pattern"
                    )
                    variant_patterns.extend(llm_variations)
                except Exception as e:
                    logger.error(
                        f"Error during LLM pattern generation: {str(e)}",
                        component="PatternAnalyzer",
                        operation="analyze_pattern",
                        exception=e
                    )
                    # Fall back to rule-based generation
                    llm_used = False
                    
            # Use rule-based generation as a fallback or if LLM is disabled
            if not llm_used:
                # Handle the case where pattern is None gracefully
                if pattern is None:
                    logger.warning(
                        "Cannot generate variations for None pattern",
                        component="PatternAnalyzer",
                        operation="analyze_pattern"
                    )
                    rule_based_variations = []
                else:
                    try:
                        rule_based_variations = self._generate_pattern_variations(pattern, is_regex)
                        logger.info(
                            f"Rule-based generation produced {len(rule_based_variations)} pattern variants",
                            component="PatternAnalyzer",
                            operation="analyze_pattern"
                        )
                        variant_patterns.extend(rule_based_variations)
                    except Exception as e:
                        logger.error(
                            f"Error during rule-based pattern generation: {str(e)}",
                            component="PatternAnalyzer",
                            operation="analyze_pattern",
                            exception=e
                        )
                        # Continue with an empty list of variations
                        rule_based_variations = []
            
            # Combine variations (LLM takes precedence if available)
            variations = llm_variations + rule_based_variations
            
            # Evaluate each variation
            for i, var_pattern in enumerate(variations[:num_variants]):
                # Skip exact duplicates
                if var_pattern == pattern:
                    continue
                    
                # Generate a description for the variant
                source = "LLM" if var_pattern in llm_variations else "Rule-based"
                var_description = f"Variant {i+1} of '{description}' ({source})"
                
                # Evaluate the variant
                var_stats = await self.evaluate_pattern(
                    pattern=var_pattern,
                    is_regex=is_regex,
                    case_sensitive=case_sensitive,
                    paths=paths,
                    reference_set=reference_set,
                )
                
                # Create variant object
                variant = PatternVariant(
                    pattern=var_pattern,
                    description=var_description,
                    is_regex=is_regex,
                    case_sensitive=case_sensitive,
                    stats=var_stats,
                    parent_pattern=pattern,
                )
                
                # Add to variants and history
                self.pattern_variants[pattern_id].append(variant)
                self.pattern_history[pattern_id].append(variant)
                
                # Add to results
                variant_patterns.append({
                    "pattern": var_pattern,
                    "description": var_description,
                    "is_regex": is_regex,
                    "case_sensitive": case_sensitive,
                    "stats": {
                        "total_matches": var_stats.total_matches,
                        "files_with_matches": var_stats.files_with_matches,
                        "precision": var_stats.precision,
                        "recall": var_stats.recall,
                        "f1_score": var_stats.f1_score,
                        "execution_time": var_stats.execution_time,
                    },
                    "score": variant.score,
                })
            
            # Update best pattern if we have variants
            if variant_patterns:
                # Sort variants by score
                sorted_variants = sorted(variant_patterns, key=lambda x: x["score"], reverse=True)
                best_pattern_in_variants = sorted_variants[0]["pattern"]
            
        operation_execution_time = time.time() - operation_start_time # Total time for the operation
        
        # Create result
        result = {
            "pattern_id": pattern_id,
            "original_pattern": pattern,
            "description": description,
            "is_regex": is_regex,
            "case_sensitive": case_sensitive,
            "stats": {
                "total_matches": original_stats.total_matches,
                "files_with_matches": original_stats.files_with_matches,
                "precision": original_stats.precision,
                "recall": original_stats.recall,
                "f1_score": original_stats.f1_score,
                "execution_time": original_stats.execution_time,
            },
            "variants": variant_patterns,
            "best_pattern": best_pattern_in_variants,
            "execution_time": operation_execution_time,
        }
        
        # Log completion
        logger.success(
            f"Pattern analysis completed: {pattern}",
            component="evolution",
            operation="analyze_pattern",
            context={
                "pattern_id": pattern_id,
                "matches": original_stats.total_matches,
                "variant_count": len(variant_patterns),
                "llm_used": llm_used
            }
        )
        
        return result
    
    async def evolve_pattern(
        self,
        pattern_id: str,
        paths: List[str],
        reference_set: Optional[List[Tuple[str, int]]] = None,
        generations: int = 3,
        population_size: int = 5,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.3,
    ) -> Dict[str, Any]:
        """Evolve a pattern through multiple generations.
        
        Args:
            pattern_id: ID of the pattern to evolve
            paths: Paths to search
            reference_set: Optional reference set of (file, line) locations
            generations: Number of generations to evolve
            population_size: Size of each generation
            mutation_rate: Rate of mutation (0.0-1.0)
            crossover_rate: Rate of crossover (0.0-1.0)
            
        Returns:
            Evolution results
        """
        # Check if pattern exists
        if pattern_id not in self.pattern_variants:
            raise ValueError(f"Pattern ID not found: {pattern_id}")
            
        # Get the current variants
        variants = self.pattern_variants[pattern_id]
        
        if not variants:
            raise ValueError(f"No variants found for pattern ID: {pattern_id}")
            
        # Log the operation
        logger.info(
            f"Evolving pattern: {variants[0].pattern}",
            component="evolution",
            operation="evolve_pattern",
            context={
                "pattern_id": pattern_id,
                "generations": generations,
                "population_size": population_size,
            }
        )
        
        # Track evolution history
        evolution_history = []
        
        # Initial population is the current variants
        population = variants.copy()
        
        # Ensure we have enough variants for the initial population
        while len(population) < population_size:
            # Clone random variants with mutations
            import random
            parent = random.choice(variants)
            
            # Create a mutated variant
            mutated_pattern = self._mutate_pattern(
                pattern=parent.pattern,
                is_regex=parent.is_regex,
                mutation_strength=mutation_rate,
            )
            
            # Skip exact duplicates
            if any(v.pattern == mutated_pattern for v in population):
                continue
                
            # Evaluate the new variant
            stats = await self.evaluate_pattern(
                pattern=mutated_pattern,
                is_regex=parent.is_regex,
                case_sensitive=parent.case_sensitive,
                paths=paths,
                reference_set=reference_set,
            )
            
            # Create variant object
            variant = PatternVariant(
                pattern=mutated_pattern,
                description=f"Mutation of '{parent.description}'",
                is_regex=parent.is_regex,
                case_sensitive=parent.case_sensitive,
                stats=stats,
                generation=1,
                parent_pattern=parent.pattern,
            )
            
            # Add to population
            population.append(variant)
            
            # Add to history
            self.pattern_history[pattern_id].append(variant)
        
        # Evolve through generations
        for gen in range(1, generations + 1):
            # Record the current generation
            gen_stats = {
                "generation": gen,
                "population": [
                    {
                        "pattern": v.pattern,
                        "description": v.description,
                        "score": v.score,
                        "matches": v.stats.total_matches,
                        "precision": v.stats.precision,
                        "recall": v.stats.recall,
                        "f1_score": v.stats.f1_score,
                    }
                    for v in population
                ],
                "best_pattern": max(population, key=lambda v: v.score).pattern,
                "best_score": max(population, key=lambda v: v.score).score,
                "avg_score": statistics.mean(v.score for v in population),
            }
            evolution_history.append(gen_stats)
            
            # Log generation progress
            logger.info(
                f"Evolution generation {gen}: {len(population)} variants",
                component="evolution",
                operation="evolve_pattern",
                context={
                    "generation": gen,
                    "best_score": gen_stats["best_score"],
                    "avg_score": gen_stats["avg_score"],
                }
            )
            
            # Generate new population through selection, crossover, and mutation
            new_population = []
            
            # Keep the best variants (elitism)
            elite_count = max(1, population_size // 5)
            elites = sorted(population, key=lambda v: v.score, reverse=True)[:elite_count]
            new_population.extend(elites)
            
            # Fill the rest of the population
            while len(new_population) < population_size:
                # Select parents through tournament selection
                import random
                
                tournament_size = min(3, len(population))
                tournament = random.sample(population, tournament_size)
                parent1 = max(tournament, key=lambda v: v.score)
                
                tournament = random.sample(population, tournament_size)
                parent2 = max(tournament, key=lambda v: v.score)
                
                # Determine if we do crossover
                if random.random() < crossover_rate and parent1.pattern != parent2.pattern:
                    # Create a crossover variant
                    child_pattern = self._crossover_patterns(
                        pattern1=parent1.pattern,
                        pattern2=parent2.pattern,
                        is_regex=parent1.is_regex,
                    )
                    
                    # Determine if we also do mutation
                    if random.random() < mutation_rate:
                        child_pattern = self._mutate_pattern(
                            pattern=child_pattern,
                            is_regex=parent1.is_regex,
                            mutation_strength=mutation_rate,
                        )
                    
                    # Skip exact duplicates
                    if any(v.pattern == child_pattern for v in new_population + population):
                        continue
                        
                    # Evaluate the new variant
                    stats = await self.evaluate_pattern(
                        pattern=child_pattern,
                        is_regex=parent1.is_regex,
                        case_sensitive=parent1.case_sensitive,
                        paths=paths,
                        reference_set=reference_set,
                    )
                    
                    # Create variant object
                    variant = PatternVariant(
                        pattern=child_pattern,
                        description=f"Crossover of '{parent1.description}' and '{parent2.description}'",
                        is_regex=parent1.is_regex,
                        case_sensitive=parent1.case_sensitive,
                        stats=stats,
                        generation=gen + 1,
                        parent_pattern=f"{parent1.pattern} + {parent2.pattern}",
                    )
                    
                    # Add to new population
                    new_population.append(variant)
                    
                    # Add to history
                    self.pattern_history[pattern_id].append(variant)
                else:
                    # Just do mutation
                    parent = parent1
                    
                    # Create a mutated variant
                    mutated_pattern = self._mutate_pattern(
                        pattern=parent.pattern,
                        is_regex=parent.is_regex,
                        mutation_strength=mutation_rate,
                    )
                    
                    # Skip exact duplicates
                    if any(v.pattern == mutated_pattern for v in new_population + population):
                        continue
                        
                    # Evaluate the new variant
                    stats = await self.evaluate_pattern(
                        pattern=mutated_pattern,
                        is_regex=parent.is_regex,
                        case_sensitive=parent.case_sensitive,
                        paths=paths,
                        reference_set=reference_set,
                    )
                    
                    # Create variant object
                    variant = PatternVariant(
                        pattern=mutated_pattern,
                        description=f"Mutation of '{parent.description}'",
                        is_regex=parent.is_regex,
                        case_sensitive=parent.case_sensitive,
                        stats=stats,
                        generation=gen + 1,
                        parent_pattern=parent.pattern,
                    )
                    
                    # Add to new population
                    new_population.append(variant)
                    
                    # Add to history
                    self.pattern_history[pattern_id].append(variant)
            
            # Set the new population
            population = new_population
        
        # Final generation stats
        final_gen_stats = {
            "generation": generations,
            "population": [
                {
                    "pattern": v.pattern,
                    "description": v.description,
                    "score": v.score,
                    "matches": v.stats.total_matches,
                    "precision": v.stats.precision,
                    "recall": v.stats.recall,
                    "f1_score": v.stats.f1_score,
                }
                for v in population
            ],
            "best_pattern": max(population, key=lambda v: v.score).pattern,
            "best_score": max(population, key=lambda v: v.score).score,
            "avg_score": statistics.mean(v.score for v in population),
        }
        evolution_history.append(final_gen_stats)
        
        # Find the best overall pattern
        best_variant = max(population, key=lambda v: v.score)
        
        # Update the pattern variants
        self.pattern_variants[pattern_id] = sorted(
            population, key=lambda v: v.score, reverse=True
        )
        
        # Calculate improvement
        original_variant = variants[0]
        improvement = (best_variant.score - original_variant.score) / original_variant.score
        improvement_percentage = improvement * 100
        
        # Log completion
        logger.success(
            f"Pattern evolution completed: {best_variant.pattern}",
            component="evolution",
            operation="evolve_pattern",
            context={
                "pattern_id": pattern_id,
                "generations": generations,
                "best_score": best_variant.score,
                "improvement": f"{improvement_percentage:.2f}%",
            }
        )
        
        # Create result
        result = {
            "pattern_id": pattern_id,
            "original_pattern": original_variant.pattern,
            "best_pattern": best_variant.pattern,
            "improvement_percentage": improvement_percentage,
            "generations": generations,
            "population_size": population_size,
            "history": evolution_history,
            "final_population": [
                {
                    "pattern": v.pattern,
                    "description": v.description,
                    "score": v.score,
                    "stats": {
                        "total_matches": v.stats.total_matches,
                        "files_with_matches": v.stats.files_with_matches,
                        "precision": v.stats.precision,
                        "recall": v.stats.recall,
                        "f1_score": v.stats.f1_score,
                    },
                }
                for v in self.pattern_variants[pattern_id]
            ],
        }
        
        return result
    
    def get_pattern_history(self, pattern_id: str) -> List[Dict[str, Any]]:
        """Get the history of a pattern's evolution.
        
        Args:
            pattern_id: Pattern ID
            
        Returns:
            Pattern history
        """
        if pattern_id not in self.pattern_history:
            return []
            
        return [
            {
                "pattern": v.pattern,
                "description": v.description,
                "score": v.score,
                "generation": v.generation,
                "parent_pattern": v.parent_pattern,
                "stats": {
                    "total_matches": v.stats.total_matches,
                    "files_with_matches": v.stats.files_with_matches,
                    "precision": v.stats.precision,
                    "recall": v.stats.recall,
                    "f1_score": v.stats.f1_score,
                },
            }
            for v in self.pattern_history[pattern_id]
        ]


# Create a singleton instance
_pattern_analyzer = None


def get_pattern_analyzer() -> PatternAnalyzer:
    """Get the singleton PatternAnalyzer instance.
    
    Returns:
        PatternAnalyzer instance
    """
    global _pattern_analyzer
    
    if _pattern_analyzer is None:
        _pattern_analyzer = PatternAnalyzer()
        
    return _pattern_analyzer


async def analyze_pattern(
    pattern: str,
    description: str,
    is_regex: bool,
    case_sensitive: bool,
    paths: List[str],
    reference_set: Optional[Dict[str, List[str]]] = None,
    generate_variants: bool = True,
    num_variants: int = 3,
) -> Dict[str, Any]:
    """Analyze a pattern and optionally generate variations.
    
    This is a convenience function that uses the singleton PatternAnalyzer.
    
    Args:
        pattern: Pattern to analyze
        description: Description of the pattern
        is_regex: Whether the pattern is a regex
        case_sensitive: Whether the pattern is case-sensitive
        paths: Paths to search (used if reference_set doesn't provide examples)
        reference_set: Optional dictionary with 'positive' and 'negative' example strings.
        generate_variants: Whether to generate variants
        num_variants: Number of variants to generate
        
    Returns:
        Analysis results
    """
    # Validate pattern input
    if pattern is None or pattern.strip() == "":
        logger.warning("Empty or None pattern provided to analyze_pattern", component="PatternAnalyzer")
        return {
            "error": "Pattern is empty or None",
            "status": "error",
            "pattern": pattern,
            "variations": [],
            "stats": None,
            "execution_time": 0.0
        }
        
    analyzer = get_pattern_analyzer()
    
    return await analyzer.analyze_pattern(
        pattern=pattern,
        description=description,
        is_regex=is_regex,
        case_sensitive=case_sensitive,
        paths=paths,
        reference_set=reference_set,
        generate_variants=generate_variants,
        num_variants=num_variants,
    )


async def evolve_pattern(
    pattern_id: str,
    paths: List[str],
    reference_set: Optional[List[Tuple[str, int]]] = None,
    generations: int = 3,
    population_size: int = 5,
    mutation_rate: float = 0.2,
    crossover_rate: float = 0.3,
) -> Dict[str, Any]:
    """Evolve a pattern through multiple generations.
    
    This is a convenience function that uses the singleton PatternAnalyzer.
    
    Args:
        pattern_id: ID of the pattern to evolve
        paths: Paths to search
        reference_set: Optional reference set of (file, line) locations
        generations: Number of generations to evolve
        population_size: Size of each generation
        mutation_rate: Rate of mutation (0.0-1.0)
        crossover_rate: Rate of crossover (0.0-1.0)
        
    Returns:
        Evolution results
    """
    analyzer = get_pattern_analyzer()
    
    return await analyzer.evolve_pattern(
        pattern_id=pattern_id,
        paths=paths,
        reference_set=reference_set,
        generations=generations,
        population_size=population_size,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
    )