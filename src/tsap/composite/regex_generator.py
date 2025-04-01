"""
Regex Generator for the TSAP MCP Server.

This module implements tools for automatically generating
regular expressions based on examples or patterns.
"""

import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
from collections import Counter

from tsap.utils.logging import logger
from tsap.mcp.models import RegexGeneratorParams, RegexGeneratorResult


@dataclass
class RegexCandidate:
    """
    A candidate regular expression.
    """
    pattern: str
    confidence: float
    description: str
    examples_matched: List[str] = field(default_factory=list)
    examples_missed: List[str] = field(default_factory=list)
    precision: float = 1.0
    recall: float = 1.0
    
    @property
    def f1_score(self) -> float:
        """Calculate F1 score based on precision and recall."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)


def _find_common_prefix(strings: List[str]) -> str:
    """
    Find the longest common prefix among a list of strings.
    
    Args:
        strings: List of strings
        
    Returns:
        Common prefix
    """
    if not strings:
        return ""
    
    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    
    return prefix


def _find_common_suffix(strings: List[str]) -> str:
    """
    Find the longest common suffix among a list of strings.
    
    Args:
        strings: List of strings
        
    Returns:
        Common suffix
    """
    if not strings:
        return ""
    
    # Reverse strings to find common prefix, then reverse back
    reversed_strings = [s[::-1] for s in strings]
    common_prefix = _find_common_prefix(reversed_strings)
    return common_prefix[::-1]


def _escape_for_regex(s: str) -> str:
    """
    Escape special characters for use in a regular expression.
    
    Args:
        s: String to escape
        
    Returns:
        Escaped string
    """
    special_chars = r".*+?^$()[]{}|\\"
    return "".join("\\" + c if c in special_chars else c for c in s)


def _classify_character(c: str) -> str:
    """
    Classify a character for pattern generalization.
    
    Args:
        c: Character to classify
        
    Returns:
        Character class
    """
    if c.isdigit():
        return "d"
    elif c.isalpha():
        if c.isupper():
            return "U"
        else:
            return "l"
    elif c.isspace():
        return "s"
    else:
        return "o"  # Other


def _identify_patterns(examples: List[str]) -> List[Tuple[str, float]]:
    """
    Identify common patterns in examples.
    
    Args:
        examples: List of example strings
        
    Returns:
        List of (pattern, confidence) tuples
    """
    if not examples:
        return []
    
    # Detect character patterns
    patterns = []
    
    # Check for fixed-length patterns
    lengths = [len(ex) for ex in examples]
    common_length = max(set(lengths), key=lengths.count) if lengths else 0
    length_confidence = lengths.count(common_length) / len(examples)
    
    if length_confidence > 0.8:
        patterns.append((f"^.{{{common_length}}}$", length_confidence))
    
    # Check for alphanumeric, numeric, and alphabetic patterns
    counts = {
        "alphanumeric": sum(1 for ex in examples if ex.isalnum()),
        "alphabetic": sum(1 for ex in examples if ex.isalpha()),
        "lowercase": sum(1 for ex in examples if ex.islower()),
        "uppercase": sum(1 for ex in examples if ex.isupper()),
        "numeric": sum(1 for ex in examples if ex.isdigit()),
        "alphanumeric_with_symbols": sum(1 for ex in examples if re.match(r'^[\w\s\-_]+$', ex))
    }
    
    for pattern_type, count in counts.items():
        confidence = count / len(examples)
        if confidence > 0.8:
            regex = ""
            if pattern_type == "alphanumeric":
                regex = r"^\w+$"
            elif pattern_type == "alphabetic":
                regex = r"^[a-zA-Z]+$"
            elif pattern_type == "lowercase":
                regex = r"^[a-z]+$"
            elif pattern_type == "uppercase":
                regex = r"^[A-Z]+$"
            elif pattern_type == "numeric":
                regex = r"^\d+$"
            elif pattern_type == "alphanumeric_with_symbols":
                regex = r"^[\w\s\-_]+$"
            
            patterns.append((regex, confidence))
    
    # Check for common structures
    # Email pattern
    email_count = sum(1 for ex in examples if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', ex))
    if email_count / len(examples) > 0.8:
        patterns.append((r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email_count / len(examples)))
    
    # URL pattern
    url_count = sum(1 for ex in examples if re.match(r'^(https?|ftp)://[^\s/$.?#].[^\s]*$', ex))
    if url_count / len(examples) > 0.8:
        patterns.append((r"^(https?|ftp)://[^\s/$.?#].[^\s]*$", url_count / len(examples)))
    
    # IP address pattern
    ip_count = sum(1 for ex in examples if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', ex))
    if ip_count / len(examples) > 0.8:
        patterns.append((r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", ip_count / len(examples)))
    
    # Date patterns
    date_formats = [
        (r'^\d{4}-\d{2}-\d{2}$', "ISO date"),
        (r'^\d{2}/\d{2}/\d{4}$', "MM/DD/YYYY"),
        (r'^\d{2}-\d{2}-\d{4}$', "MM-DD-YYYY"),
        (r'^\d{4}/\d{2}/\d{2}$', "YYYY/MM/DD")
    ]
    
    for date_regex, desc in date_formats:
        date_count = sum(1 for ex in examples if re.match(date_regex, ex))
        if date_count / len(examples) > 0.8:
            patterns.append((date_regex, date_count / len(examples)))
    
    return patterns


def _generalize_pattern(examples: List[str]) -> Tuple[str, float]:
    """
    Generate a generalized pattern from examples.
    
    Args:
        examples: List of example strings
        
    Returns:
        Tuple of (regex pattern, confidence)
    """
    if not examples:
        return ("", 0.0)
    
    # Find common prefix and suffix
    common_prefix = _find_common_prefix(examples)
    common_suffix = _find_common_suffix(examples)
    
    # Remove prefix and suffix from examples to analyze the variable part
    middle_parts = [
        ex[len(common_prefix):len(ex)-len(common_suffix)]
        for ex in examples
    ]
    
    # Analyze middle parts
    if not any(middle_parts):
        # Fixed string, no variable part
        return (_escape_for_regex(common_prefix + common_suffix), 1.0)
    
    # Check if middle parts are all digits
    all_digits = all(part.isdigit() for part in middle_parts if part)
    if all_digits:
        digit_lengths = [len(part) for part in middle_parts if part]
        if len(set(digit_lengths)) == 1:
            # Fixed number of digits
            digit_count = digit_lengths[0]
            return (f"{_escape_for_regex(common_prefix)}\\d{{{digit_count}}}{_escape_for_regex(common_suffix)}", 0.9)
        else:
            # Variable number of digits
            min_len = min(digit_lengths)
            max_len = max(digit_lengths)
            return (f"{_escape_for_regex(common_prefix)}\\d{{{min_len},{max_len}}}{_escape_for_regex(common_suffix)}", 0.8)
    
    # Check if middle parts are all alphabetic
    all_alpha = all(part.isalpha() for part in middle_parts if part)
    if all_alpha:
        alpha_lengths = [len(part) for part in middle_parts if part]
        if len(set(alpha_lengths)) == 1:
            # Fixed number of letters
            alpha_count = alpha_lengths[0]
            return (f"{_escape_for_regex(common_prefix)}[a-zA-Z]{{{alpha_count}}}{_escape_for_regex(common_suffix)}", 0.9)
        else:
            # Variable number of letters
            min_len = min(alpha_lengths)
            max_len = max(alpha_lengths)
            return (f"{_escape_for_regex(common_prefix)}[a-zA-Z]{{{min_len},{max_len}}}{_escape_for_regex(common_suffix)}", 0.8)
    
    # Check if middle parts are all alphanumeric
    all_alnum = all(part.isalnum() for part in middle_parts if part)
    if all_alnum:
        alnum_lengths = [len(part) for part in middle_parts if part]
        if len(set(alnum_lengths)) == 1:
            # Fixed number of alphanumeric chars
            alnum_count = alnum_lengths[0]
            return (f"{_escape_for_regex(common_prefix)}\\w{{{alnum_count}}}{_escape_for_regex(common_suffix)}", 0.85)
        else:
            # Variable number of alphanumeric chars
            min_len = min(alnum_lengths)
            max_len = max(alnum_lengths)
            return (f"{_escape_for_regex(common_prefix)}\\w{{{min_len},{max_len}}}{_escape_for_regex(common_suffix)}", 0.75)
    
    # More generic approach - identify character classes at each position
    if all(len(part) == len(middle_parts[0]) for part in middle_parts):
        # All middle parts have the same length
        classes = []
        for i in range(len(middle_parts[0])):
            chars = [part[i] for part in middle_parts]
            unique_chars = set(chars)
            
            if len(unique_chars) == 1:
                # Fixed character
                classes.append(_escape_for_regex(unique_chars.pop()))
            elif all(c.isdigit() for c in unique_chars):
                classes.append("\\d")
            elif all(c.isalpha() for c in unique_chars):
                if all(c.islower() for c in unique_chars):
                    classes.append("[a-z]")
                elif all(c.isupper() for c in unique_chars):
                    classes.append("[A-Z]")
                else:
                    classes.append("[a-zA-Z]")
            elif all(c.isalnum() for c in unique_chars):
                classes.append("\\w")
            else:
                # Mixed characters, list them explicitly
                char_list = "".join(_escape_for_regex(c) for c in sorted(unique_chars))
                classes.append(f"[{char_list}]")
        
        pattern = _escape_for_regex(common_prefix) + "".join(classes) + _escape_for_regex(common_suffix)
        return (pattern, 0.85)
    
    # Variable length, more generic pattern
    # Try to identify most common character classes
    class_counts = Counter()
    for part in middle_parts:
        for c in part:
            class_counts[_classify_character(c)] += 1
    
    total_chars = sum(class_counts.values())
    
    # Determine predominant character classes
    class_patterns = []
    if class_counts["d"] / total_chars > 0.7:
        class_patterns.append("\\d+")
    elif class_counts["l"] / total_chars > 0.7:
        class_patterns.append("[a-z]+")
    elif class_counts["U"] / total_chars > 0.7:
        class_patterns.append("[A-Z]+")
    elif (class_counts["d"] + class_counts["l"] + class_counts["U"]) / total_chars > 0.7:
        class_patterns.append("\\w+")
    else:
        class_patterns.append(".+")
    
    # Determine length quantifiers
    lengths = [len(part) for part in middle_parts]
    min_len = min(lengths)
    max_len = max(lengths)
    if min_len == max_len:
        quantifier = f"{{{min_len}}}"
    else:
        quantifier = f"{{{min_len},{max_len}}}"
    
    class_pattern = class_patterns[0]
    # Replace + with the quantifier
    class_pattern = class_pattern[:-1] + quantifier
    
    pattern = _escape_for_regex(common_prefix) + class_pattern + _escape_for_regex(common_suffix)
    confidence = 0.7  # Lower confidence for variable-length patterns
    
    return (pattern, confidence)


def _evaluate_regex(
    regex: str,
    positive_examples: List[str],
    negative_examples: List[str] = None
) -> Tuple[float, float, List[str], List[str]]:
    """
    Evaluate a regex against positive and negative examples.
    
    Args:
        regex: Regular expression to evaluate
        positive_examples: Examples that should match
        negative_examples: Examples that should not match
        
    Returns:
        Tuple of (precision, recall, matched_examples, missed_examples)
    """
    if not positive_examples:
        return 0.0, 0.0, [], []
    
    try:
        pattern = re.compile(regex)
    except re.error:
        return 0.0, 0.0, [], []
    
    # Check positive examples
    matched = []
    missed = []
    for example in positive_examples:
        if pattern.fullmatch(example):
            matched.append(example)
        else:
            missed.append(example)
    
    recall = len(matched) / len(positive_examples) if positive_examples else 0
    
    # Check negative examples
    if negative_examples:
        false_positives = sum(1 for ex in negative_examples if pattern.fullmatch(ex))
        precision = (len(matched) / (len(matched) + false_positives)) if (len(matched) + false_positives) > 0 else 0
    else:
        precision = 1.0  # No negative examples to evaluate precision
    
    return precision, recall, matched, missed


def _merge_patterns(patterns: List[Tuple[str, float]]) -> List[RegexCandidate]:
    """
    Merge similar patterns into stronger candidates.
    
    Args:
        patterns: List of (pattern, confidence) tuples
        
    Returns:
        List of merged RegexCandidate objects
    """
    if not patterns:
        return []
    
    # Group by similarity
    candidates = []
    for pattern, confidence in patterns:
        description = "Automatically generated pattern"
        
        # Add descriptive labels
        if r"\d" in pattern:
            if r"[a-zA-Z]" in pattern or r"\w" in pattern:
                description = "Alphanumeric pattern"
            else:
                description = "Numeric pattern"
        elif r"[a-zA-Z]" in pattern or r"\w" in pattern:
            description = "Alphabetic pattern"
        
        # Add more specific descriptions
        if pattern.startswith(r"^") and pattern.endswith(r"$"):
            if re.search(r"\{\d+\}$", pattern[:-1]):
                description = f"Fixed-length {description.lower()}"
            elif re.search(r"\{\d+,\d+\}$", pattern[:-1]):
                description = f"Variable-length {description.lower()}"
        
        candidates.append(RegexCandidate(
            pattern=pattern,
            confidence=confidence,
            description=description
        ))
    
    return candidates


async def generate_regex(params: RegexGeneratorParams) -> RegexGeneratorResult:
    """
    Generate a regular expression from examples.
    
    Args:
        params: Parameters for regex generation
        
    Returns:
        Results of regex generation
    """
    try:
        candidates = []
        
        # Generate basic pattern
        basic_pattern, basic_confidence = _generalize_pattern(params.positive_examples)
        basic_precision, basic_recall, matched, missed = _evaluate_regex(
            basic_pattern,
            params.positive_examples,
            params.negative_examples
        )
        
        candidates.append(RegexCandidate(
            pattern=basic_pattern,
            confidence=basic_confidence,
            description="Basic pattern matching common structure",
            examples_matched=matched,
            examples_missed=missed,
            precision=basic_precision,
            recall=basic_recall
        ))
        
        # Identify additional patterns
        identified_patterns = _identify_patterns(params.positive_examples)
        
        if identified_patterns:
            # Evaluate each identified pattern
            for pattern, confidence in identified_patterns:
                precision, recall, matched, missed = _evaluate_regex(
                    pattern,
                    params.positive_examples,
                    params.negative_examples
                )
                
                candidates.append(RegexCandidate(
                    pattern=pattern,
                    confidence=confidence,
                    description="Specialized pattern",  # Will be updated when merging
                    examples_matched=matched,
                    examples_missed=missed,
                    precision=precision,
                    recall=recall
                ))
        
        # If we don't have a viable candidate yet, try a fallback
        if all(c.recall < 0.5 for c in candidates):
            # Try a more permissive approach
            parts = []
            for ex in params.positive_examples:
                parts.append(_escape_for_regex(ex))
            
            if parts:
                # Create a pattern that matches any of the examples
                fallback_pattern = f"^({'|'.join(parts)})$"
                precision, recall, matched, missed = _evaluate_regex(
                    fallback_pattern,
                    params.positive_examples,
                    params.negative_examples
                )
                
                candidates.append(RegexCandidate(
                    pattern=fallback_pattern,
                    confidence=0.5,  # Low confidence for fallback
                    description="Fallback pattern (exact match of examples)",
                    examples_matched=matched,
                    examples_missed=missed,
                    precision=precision,
                    recall=recall
                ))
        
        # Merge and rank candidates
        merged_candidates = _merge_patterns([(c.pattern, c.confidence) for c in candidates])
        
        # Update merged candidates with evaluation results
        for i, candidate in enumerate(merged_candidates):
            # Find the original candidate to get evaluation results
            for orig in candidates:
                if orig.pattern == candidate.pattern:
                    candidate.precision = orig.precision
                    candidate.recall = orig.recall
                    candidate.examples_matched = orig.examples_matched
                    candidate.examples_missed = orig.examples_missed
                    break
        
        # Sort by F1 score, then by confidence
        ranked_candidates = sorted(
            merged_candidates,
            key=lambda c: (c.f1_score, c.confidence),
            reverse=True
        )
        
        # Return results
        return RegexGeneratorResult(
            regex_candidates=[vars(c) for c in ranked_candidates],
            best_regex=ranked_candidates[0].pattern if ranked_candidates else "",
            best_regex_confidence=ranked_candidates[0].confidence if ranked_candidates else 0.0,
            example_count=len(params.positive_examples),
            generated_at=params.timestamp
        )
    
    except Exception as e:
        logger.error(
            f"Error generating regex: {str(e)}",
            component="composite",
            operation="regex_generator"
        )
        
        # Return error result
        return RegexGeneratorResult(
            regex_candidates=[],
            best_regex="",
            best_regex_confidence=0.0,
            example_count=len(params.positive_examples),
            generated_at=params.timestamp,
            error=str(e)
        )


async def generate_regex_from_examples(
    examples: List[str],
    negative_examples: List[str] = None,
    exact_match: bool = False
) -> Dict[str, Any]:
    """
    Generate a regular expression from examples.
    
    Args:
        examples: Positive examples
        negative_examples: Negative examples
        exact_match: Whether to match whole strings
        
    Returns:
        Dictionary with generated regex
    """
    import datetime
    
    # Create parameters
    params = RegexGeneratorParams(
        positive_examples=examples,
        negative_examples=negative_examples or [],
        exact_match=exact_match,
        timestamp=datetime.datetime.now().isoformat()
    )
    
    # Generate regex
    result = await generate_regex(params)
    
    return result.dict()