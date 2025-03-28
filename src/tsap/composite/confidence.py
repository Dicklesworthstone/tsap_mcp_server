"""
Confidence scoring mechanisms for composite operations.

This module provides utilities for calculating confidence scores for various
types of operations, including pattern matching, data extraction, transformations,
and analysis operations.
"""

import re
from enum import Enum
from typing import Callable, Dict, List, Any, Optional, Union, Tuple

from tsap.composite.base import ConfidenceLevel


class ConfidenceFactorType(str, Enum):
    """Types of confidence factors that can be used in scoring."""
    PATTERN_QUALITY = "pattern_quality"
    MATCH_QUALITY = "match_quality"
    SAMPLE_SIZE = "sample_size"
    CONTEXT_QUALITY = "context_quality"
    DATA_QUALITY = "data_quality"
    VALIDATION_RESULT = "validation_result"
    CONSISTENCY = "consistency"
    HEURISTIC = "heuristic"
    EXACT_MATCH = "exact_match"
    COVERAGE = "coverage"
    EXPERTISE_LEVEL = "expertise_level"
    EXTERNAL_VALIDATION = "external_validation"


class ConfidenceFactor:
    """
    Represents a single factor contributing to a confidence score.
    
    Attributes:
        name: Name of the factor
        type: Type of the factor
        value: Value of the factor (0.0 to 1.0)
        weight: Weight of the factor (default 1.0)
        description: Description of the factor
    """
    
    def __init__(
        self,
        name: str,
        type: Union[ConfidenceFactorType, str],
        value: float,
        weight: float = 1.0,
        description: Optional[str] = None
    ) -> None:
        """
        Initialize a new confidence factor.
        
        Args:
            name: Name of the factor
            type: Type of the factor
            value: Value of the factor (0.0 to 1.0)
            weight: Weight of the factor (default 1.0)
            description: Description of the factor
            
        Raises:
            ValueError: If value is not between 0.0 and 1.0
            ValueError: If weight is not positive
        """
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Confidence factor value must be between 0.0 and 1.0, got {value}")
        
        if weight <= 0.0:
            raise ValueError(f"Confidence factor weight must be positive, got {weight}")
        
        self.name = name
        self.type = type if isinstance(type, ConfidenceFactorType) else ConfidenceFactorType(type)
        self.value = value
        self.weight = weight
        self.description = description or f"{name} confidence factor"
    
    def __repr__(self) -> str:
        return f"ConfidenceFactor(name={self.name}, type={self.type}, value={self.value:.2f}, weight={self.weight:.2f})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "type": self.type.value,
            "value": self.value,
            "weight": self.weight,
            "description": self.description,
            "level": ConfidenceLevel.to_str(self.value)
        }


class ConfidenceCalculator:
    """
    Calculator for confidence scores based on multiple factors.
    
    Provides methods for calculating weighted confidence scores
    and explaining the factors that contribute to the score.
    """
    
    @staticmethod
    def calculate(factors: List[ConfidenceFactor]) -> float:
        """
        Calculate an overall confidence score from multiple factors.
        
        Args:
            factors: List of confidence factors
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not factors:
            return 0.0
        
        total_weight = sum(factor.weight for factor in factors)
        weighted_sum = sum(factor.value * factor.weight for factor in factors)
        
        return min(1.0, max(0.0, weighted_sum / total_weight))
    
    @staticmethod
    def explain(factors: List[ConfidenceFactor], score: float) -> Dict[str, Any]:
        """
        Generate an explanation of a confidence score.
        
        Args:
            factors: List of confidence factors
            score: Overall confidence score
            
        Returns:
            Dictionary with explanation of the confidence score
        """
        # Sort factors by contribution (value * weight)
        sorted_factors = sorted(
            factors,
            key=lambda f: f.value * f.weight,
            reverse=True
        )
        
        # Calculate contribution percentage for each factor
        total_weight = sum(factor.weight for factor in factors)
        contributions = []
        
        for factor in sorted_factors:
            contribution = (factor.value * factor.weight) / total_weight
            contributions.append({
                "name": factor.name,
                "type": factor.type.value,
                "value": factor.value,
                "weight": factor.weight,
                "contribution": contribution,
                "contribution_percent": contribution * 100,
                "description": factor.description,
                "level": ConfidenceLevel.to_str(factor.value)
            })
        
        return {
            "score": score,
            "confidence_level": ConfidenceLevel.to_str(score),
            "factors": {f["name"]: f for f in contributions},
            "top_factors": contributions[:3],
            "total_factors": len(factors),
            "average_factor_value": sum(f.value for f in factors) / len(factors),
            "explanation": _generate_explanation_text(score, contributions[:3])
        }
    
    @staticmethod
    def from_dict(factors_dict: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> List[ConfidenceFactor]:
        """
        Create confidence factors from a dictionary of values.
        
        Args:
            factors_dict: Dictionary mapping factor names to values
            weights: Optional dictionary mapping factor names to weights
            
        Returns:
            List of confidence factors
        """
        factors = []
        
        for name, value in factors_dict.items():
            weight = weights.get(name, 1.0) if weights else 1.0
            factor_type = _infer_factor_type(name)
            
            factors.append(ConfidenceFactor(
                name=name,
                type=factor_type,
                value=min(1.0, max(0.0, value)),
                weight=weight
            ))
        
        return factors


def _infer_factor_type(name: str) -> ConfidenceFactorType:
    """
    Infer the type of a confidence factor from its name.
    
    Args:
        name: Name of the factor
        
    Returns:
        Inferred factor type
    """
    name_lower = name.lower()
    
    if any(p in name_lower for p in ["pattern", "regex", "expression"]):
        return ConfidenceFactorType.PATTERN_QUALITY
    elif any(p in name_lower for p in ["match", "matches"]):
        return ConfidenceFactorType.MATCH_QUALITY
    elif any(p in name_lower for p in ["sample", "samples", "count"]):
        return ConfidenceFactorType.SAMPLE_SIZE
    elif any(p in name_lower for p in ["context"]):
        return ConfidenceFactorType.CONTEXT_QUALITY
    elif any(p in name_lower for p in ["data", "quality"]):
        return ConfidenceFactorType.DATA_QUALITY
    elif any(p in name_lower for p in ["validate", "validation"]):
        return ConfidenceFactorType.VALIDATION_RESULT
    elif any(p in name_lower for p in ["consistent", "consistency"]):
        return ConfidenceFactorType.CONSISTENCY
    elif any(p in name_lower for p in ["heuristic"]):
        return ConfidenceFactorType.HEURISTIC
    elif any(p in name_lower for p in ["exact"]):
        return ConfidenceFactorType.EXACT_MATCH
    elif any(p in name_lower for p in ["coverage", "covered"]):
        return ConfidenceFactorType.COVERAGE
    else:
        return ConfidenceFactorType.HEURISTIC


def _generate_explanation_text(score: float, top_contributions: List[Dict[str, Any]]) -> str:
    """
    Generate a human-readable explanation of a confidence score.
    
    Args:
        score: Overall confidence score
        top_contributions: Top contributing factors
        
    Returns:
        Human-readable explanation
    """
    level = ConfidenceLevel.to_str(score)
    
    if not top_contributions:
        return f"Confidence is {level.lower()} ({score:.2f}) with no specific contributing factors."
    
    top_factor = top_contributions[0]
    factor_desc = top_factor.get("description", top_factor["name"])
    
    if len(top_contributions) == 1:
        return (
            f"Confidence is {level.lower()} ({score:.2f}), primarily based on "
            f"{factor_desc.lower()} ({top_factor['value']:.2f})."
        )
    
    secondary_factors = ", ".join(
        f"{c.get('description', c['name']).lower()} ({c['value']:.2f})"
        for c in top_contributions[1:]
    )
    
    return (
        f"Confidence is {level.lower()} ({score:.2f}), primarily based on "
        f"{factor_desc.lower()} ({top_factor['value']:.2f}), with "
        f"additional influence from {secondary_factors}."
    )


# Pattern confidence calculators

def calculate_regex_confidence(
    pattern: str,
    test_cases: Dict[str, bool],
    allow_partial: bool = False
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate confidence score for a regular expression based on test cases.
    
    Args:
        pattern: Regular expression pattern
        test_cases: Dictionary mapping test strings to expected match results
        allow_partial: Whether to allow partial matches (re.search vs re.fullmatch)
        
    Returns:
        Tuple of (confidence score, details dictionary)
    """
    if not test_cases:
        return 0.0, {"error": "No test cases provided"}
    
    try:
        regex = re.compile(pattern)
    except Exception as e:
        return 0.0, {"error": f"Invalid regular expression: {str(e)}"}
    
    # Evaluate test cases
    correct = 0
    false_positives = 0
    false_negatives = 0
    total = len(test_cases)
    
    for test_string, expected_match in test_cases.items():
        if allow_partial:
            actual_match = bool(regex.search(test_string))
        else:
            actual_match = bool(regex.fullmatch(test_string))
        
        if actual_match == expected_match:
            correct += 1
        elif actual_match and not expected_match:
            false_positives += 1
        else:  # not actual_match and expected_match
            false_negatives += 1
    
    # Calculate precision, recall, and F1 score
    precision = 1.0 if false_positives == 0 else correct / (correct + false_positives)
    recall = 1.0 if false_negatives == 0 else correct / (correct + false_negatives)
    
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    # Calculate pattern complexity factor
    complexity_factor = _calculate_pattern_complexity(pattern)
    
    # Calculate final confidence
    if total < 5:
        # Penalize for small sample size
        sample_size_factor = 0.5 + (0.1 * total)
    else:
        sample_size_factor = 1.0
    
    confidence_factors = [
        ConfidenceFactor(
            name="precision",
            type=ConfidenceFactorType.VALIDATION_RESULT,
            value=precision,
            weight=1.0,
            description="Precision (true positives / (true positives + false positives))"
        ),
        ConfidenceFactor(
            name="recall",
            type=ConfidenceFactorType.VALIDATION_RESULT,
            value=recall,
            weight=1.0,
            description="Recall (true positives / (true positives + false negatives))"
        ),
        ConfidenceFactor(
            name="pattern_complexity",
            type=ConfidenceFactorType.PATTERN_QUALITY,
            value=complexity_factor,
            weight=0.5,
            description="Quality of the pattern based on complexity and features"
        ),
        ConfidenceFactor(
            name="sample_size",
            type=ConfidenceFactorType.SAMPLE_SIZE,
            value=min(1.0, sample_size_factor),
            weight=0.25,
            description="Confidence based on the number of test cases"
        )
    ]
    
    confidence = ConfidenceCalculator.calculate(confidence_factors)
    
    return confidence, {
        "accuracy": correct / total,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "complexity_factor": complexity_factor,
        "sample_size_factor": sample_size_factor,
        "correct": correct,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total": total,
        "factors": confidence_factors
    }


def _calculate_pattern_complexity(pattern: str) -> float:
    """
    Calculate a complexity factor for a regular expression.
    
    Higher complexity patterns that still work correctly should have higher confidence.
    
    Args:
        pattern: Regular expression pattern
        
    Returns:
        Complexity factor between 0.0 and 1.0
    """
    # Count advanced regex features
    features = {
        "character_classes": len(re.findall(r'\[[^\]]*\]', pattern)),
        "quantifiers": len(re.findall(r'[*+?]|\{\d+(?:,\d*)?\}', pattern)),
        "groups": len(re.findall(r'\((?!\?)', pattern)),
        "named_groups": len(re.findall(r'\(\?P<[^>]+>', pattern)),
        "non_capturing_groups": len(re.findall(r'\(\?:', pattern)),
        "lookaheads": len(re.findall(r'\(\?=|\(\?!', pattern)),
        "lookbehinds": len(re.findall(r'\(\?<=|\(\?<!', pattern)),
        "backreferences": len(re.findall(r'\\[1-9]', pattern)),
        "named_backreferences": len(re.findall(r'\(\?P=[^)]+\)', pattern)),
        "anchors": len(re.findall(r'^|\$|\\b', pattern)),
        "alternation": len(re.findall(r'\|', pattern))
    }
    
    # Calculate base score
    feature_count = sum(features.values())
    
    if len(pattern) < 5:
        # Very simple patterns get lower confidence
        base_score = 0.5
    elif feature_count == 0 and len(pattern) < 10:
        # Simple literal patterns get lower confidence
        base_score = 0.6
    else:
        # More complex patterns get higher confidence
        base_score = 0.7 + min(0.3, feature_count * 0.05)
    
    return base_score


# Match confidence calculators

def calculate_match_confidence(
    match_text: str,
    pattern: Optional[str] = None,
    expected_format: Optional[str] = None,
    context_quality: Optional[float] = None,
    typical_examples: Optional[List[str]] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate confidence score for a text match based on various factors.
    
    Args:
        match_text: The matched text
        pattern: Regular expression pattern that produced the match
        expected_format: Description of the expected format (e.g., "email", "url")
        context_quality: Quality of the context (0.0 to 1.0)
        typical_examples: List of typical examples for this type of match
        
    Returns:
        Tuple of (confidence score, details dictionary)
    """
    factors = []
    details = {}
    
    # Basic quality checks
    if not match_text or match_text.isspace():
        return 0.1, {"error": "Empty or whitespace-only match"}
    
    # Length factor - extremely short or long matches may be less reliable
    length = len(match_text)
    if length < 3:
        length_factor = 0.5  # Short matches are less confident
    elif length > 200:
        length_factor = 0.7  # Very long matches are somewhat less confident
    else:
        length_factor = 0.9  # Normal length matches are fairly confident
    
    factors.append(ConfidenceFactor(
        name="length",
        type=ConfidenceFactorType.MATCH_QUALITY,
        value=length_factor,
        weight=0.5,
        description="Match length quality"
    ))
    
    # Pattern match quality if pattern is provided
    if pattern:
        try:
            regex = re.compile(pattern)
            full_match = regex.fullmatch(match_text)
            if full_match:
                pattern_match_factor = 1.0  # Full match is highly confident
                match_groups = len(full_match.groups())
            else:
                partial_match = regex.search(match_text)
                if partial_match:
                    pattern_match_factor = 0.7  # Partial match is less confident
                    match_groups = len(partial_match.groups())
                else:
                    pattern_match_factor = 0.0  # No match is not confident
                    match_groups = 0
            
            factors.append(ConfidenceFactor(
                name="pattern_match",
                type=ConfidenceFactorType.MATCH_QUALITY,
                value=pattern_match_factor,
                weight=1.0,
                description="Pattern match quality"
            ))
            
            details["match_groups"] = match_groups
        except Exception as e:
            details["pattern_error"] = str(e)
    
    # Format validation if expected format is provided
    if expected_format:
        format_validator = FORMAT_VALIDATORS.get(expected_format.lower())
        if format_validator:
            format_valid = format_validator(match_text)
            format_factor = 1.0 if format_valid else 0.3
            
            factors.append(ConfidenceFactor(
                name="format_validation",
                type=ConfidenceFactorType.VALIDATION_RESULT,
                value=format_factor,
                weight=1.5,  # Higher weight for format validation
                description=f"Validation as {expected_format} format"
            ))
            
            details["format_valid"] = format_valid
    
    # Context quality if provided
    if context_quality is not None:
        factors.append(ConfidenceFactor(
            name="context_quality",
            type=ConfidenceFactorType.CONTEXT_QUALITY,
            value=context_quality,
            weight=0.75,
            description="Quality of the surrounding context"
        ))
    
    # Similarity to typical examples if provided
    if typical_examples and len(typical_examples) > 0:
        similarities = [_calculate_string_similarity(match_text, example) for example in typical_examples]
        avg_similarity = sum(similarities) / len(similarities)
        max_similarity = max(similarities)
        
        factors.append(ConfidenceFactor(
            name="example_similarity",
            type=ConfidenceFactorType.HEURISTIC,
            value=avg_similarity * 0.5 + max_similarity * 0.5,  # Weighted average of avg and max
            weight=0.5,
            description="Similarity to typical examples"
        ))
        
        details["avg_similarity"] = avg_similarity
        details["max_similarity"] = max_similarity
    
    # Calculate overall confidence
    confidence = ConfidenceCalculator.calculate(factors)
    details["factors"] = factors
    
    return confidence, details


# Format validator functions
def _validate_email(text: str) -> bool:
    """Validate if text is an email address."""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, text))

def _validate_url(text: str) -> bool:
    """Validate if text is a URL."""
    url_pattern = r'^(https?|ftp)://[^\s/$.?#].[^\s]*$'
    return bool(re.match(url_pattern, text))

def _validate_date(text: str) -> bool:
    """Validate if text is a date in common formats."""
    date_patterns = [
        r'^\d{4}-\d{2}-\d{2}$',  # ISO format: YYYY-MM-DD
        r'^\d{1,2}/\d{1,2}/\d{2,4}$',  # MM/DD/YYYY or DD/MM/YYYY
        r'^\d{1,2}-\d{1,2}-\d{2,4}$'  # MM-DD-YYYY or DD-MM-YYYY
    ]
    return any(bool(re.match(pattern, text)) for pattern in date_patterns)

def _validate_ip_address(text: str) -> bool:
    """Validate if text is an IP address (IPv4 or IPv6)."""
    ip_patterns = [
        r'^(\d{1,3}\.){3}\d{1,3}$',  # IPv4
        r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'  # Simple IPv6
    ]
    return any(bool(re.match(pattern, text)) for pattern in ip_patterns)


# Dictionary of format validators
FORMAT_VALIDATORS = {
    'email': _validate_email,
    'url': _validate_url,
    'date': _validate_date,
    'ip': _validate_ip_address,
    'ipaddress': _validate_ip_address,
    'ip_address': _validate_ip_address
}


def _calculate_string_similarity(a: str, b: str) -> float:
    """
    Calculate similarity between two strings using a combination of
    length ratio and character overlap.
    
    Args:
        a: First string
        b: Second string
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not a or not b:
        return 0.0
    
    # Convert to lowercase for comparison
    a_lower = a.lower()
    b_lower = b.lower()
    
    # Calculate length ratio
    len_a = len(a_lower)
    len_b = len(b_lower)
    len_ratio = min(len_a, len_b) / max(len_a, len_b)
    
    # Calculate character set overlap
    set_a = set(a_lower)
    set_b = set(b_lower)
    overlap = len(set_a.intersection(set_b)) / len(set_a.union(set_b))
    
    # Calculate prefix/suffix match
    prefix_len = min(3, min(len_a, len_b))
    suffix_len = min(3, min(len_a, len_b))
    
    prefix_match = a_lower[:prefix_len] == b_lower[:prefix_len]
    suffix_match = a_lower[-suffix_len:] == b_lower[-suffix_len:]
    
    affix_factor = 0.5
    if prefix_match:
        affix_factor += 0.25
    if suffix_match:
        affix_factor += 0.25
    
    # Weighted combination
    return 0.4 * len_ratio + 0.4 * overlap + 0.2 * affix_factor


# Data transformation confidence calculators

def calculate_transformation_confidence(
    input_data: Any,
    output_data: Any,
    expected_output_type: Optional[str] = None,
    validation_checks: Optional[List[Callable[[Any], bool]]] = None,
    sample_validation: Optional[Dict[Any, Any]] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate confidence score for a data transformation.
    
    Args:
        input_data: Input data before transformation
        output_data: Output data after transformation
        expected_output_type: Expected type of the output data
        validation_checks: List of validation functions for the output data
        sample_validation: Dictionary mapping sample inputs to expected outputs
        
    Returns:
        Tuple of (confidence score, details dictionary)
    """
    factors = []
    details = {}
    
    # Check if output data exists
    if output_data is None:
        return 0.0, {"error": "No output data produced"}
    
    # Type check
    if expected_output_type:
        expected_type = _resolve_type_string(expected_output_type)
        type_match = isinstance(output_data, expected_type)
        type_factor = 1.0 if type_match else 0.2
        
        factors.append(ConfidenceFactor(
            name="type_check",
            type=ConfidenceFactorType.VALIDATION_RESULT,
            value=type_factor,
            weight=1.0,
            description=f"Output type matches expected type ({expected_output_type})"
        ))
        
        details["type_match"] = type_match
    
    # Data quality check - non-empty output
    if isinstance(output_data, (list, dict, str, bytes)):
        empty = len(output_data) == 0
    else:
        empty = False
    
    quality_factor = 0.3 if empty else 0.9
    
    factors.append(ConfidenceFactor(
        name="data_quality",
        type=ConfidenceFactorType.DATA_QUALITY,
        value=quality_factor,
        weight=0.75,
        description="Output data is non-empty and valid"
    ))
    
    # Custom validation checks
    if validation_checks:
        valid_checks = 0
        total_checks = len(validation_checks)
        
        for i, check in enumerate(validation_checks):
            try:
                if check(output_data):
                    valid_checks += 1
            except Exception:
                pass  # Ignore validation check errors
        
        validation_factor = valid_checks / total_checks if total_checks > 0 else 0.5
        
        factors.append(ConfidenceFactor(
            name="validation_checks",
            type=ConfidenceFactorType.VALIDATION_RESULT,
            value=validation_factor,
            weight=1.5,  # Higher weight for validation
            description=f"Passed {valid_checks}/{total_checks} validation checks"
        ))
        
        details["validation_results"] = {
            "passed": valid_checks,
            "total": total_checks
        }
    
    # Sample validation
    if sample_validation:
        valid_samples = 0
        total_samples = len(sample_validation)
        
        # Create a transformation function that takes a single input
        # This is a simplified approach - in a real implementation, you'd need
        # to provide the actual transformation function
        
        # Instead, we'll validate the given input/output
        if input_data in sample_validation:
            expected = sample_validation[input_data]
            if output_data == expected:
                valid_samples = 1
                total_samples = 1
        
        sample_factor = valid_samples / total_samples if total_samples > 0 else 0.5
        
        factors.append(ConfidenceFactor(
            name="sample_validation",
            type=ConfidenceFactorType.VALIDATION_RESULT,
            value=sample_factor,
            weight=1.2,
            description=f"Matched {valid_samples}/{total_samples} known samples"
        ))
        
        details["sample_validation"] = {
            "valid": valid_samples,
            "total": total_samples
        }
    
    # Transformation complexity factor
    # This is a placeholder - in a real implementation, you might
    # analyze the complexity of the transformation
    complexity_factor = 0.8  # Moderate complexity
    
    factors.append(ConfidenceFactor(
        name="transformation_complexity",
        type=ConfidenceFactorType.HEURISTIC,
        value=complexity_factor,
        weight=0.3,
        description="Estimated complexity of the transformation"
    ))
    
    # Calculate overall confidence
    confidence = ConfidenceCalculator.calculate(factors)
    details["factors"] = factors
    
    return confidence, details


def _resolve_type_string(type_str: str) -> type:
    """
    Resolve a type string to a Python type.
    
    Args:
        type_str: String representation of a type
        
    Returns:
        Python type object
    """
    type_map = {
        "str": str,
        "string": str,
        "int": int,
        "integer": int,
        "float": float,
        "bool": bool,
        "boolean": bool,
        "list": list,
        "dict": dict,
        "dictionary": dict,
        "set": set,
        "tuple": tuple,
        "none": type(None)
    }
    
    return type_map.get(type_str.lower(), object)


# Analysis confidence calculators

def calculate_analysis_confidence(
    analysis_type: str,
    result_count: int,
    error_count: int = 0,
    coverage_percent: Optional[float] = None,
    validation_score: Optional[float] = None,
    complexity: str = "medium",
    consistency_score: Optional[float] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate confidence score for an analysis operation.
    
    Args:
        analysis_type: Type of analysis performed
        result_count: Number of results produced
        error_count: Number of errors encountered
        coverage_percent: Percentage of data covered by the analysis
        validation_score: Score from validation of results
        complexity: Complexity of the analysis ("low", "medium", "high")
        consistency_score: Score for internal consistency of results
        
    Returns:
        Tuple of (confidence score, details dictionary)
    """
    factors = []
    details = {}
    
    # Basic quality check - errors vs results
    if result_count == 0:
        return 0.2, {"error": "No results produced"}
    
    error_ratio = error_count / (result_count + error_count) if (result_count + error_count) > 0 else 0
    quality_factor = 1.0 - error_ratio
    
    factors.append(ConfidenceFactor(
        name="error_ratio",
        type=ConfidenceFactorType.DATA_QUALITY,
        value=quality_factor,
        weight=1.0,
        description=f"Ratio of successful results ({result_count} results, {error_count} errors)"
    ))
    
    # Coverage factor
    if coverage_percent is not None:
        coverage_factor = min(1.0, coverage_percent / 100.0)
        
        factors.append(ConfidenceFactor(
            name="coverage",
            type=ConfidenceFactorType.COVERAGE,
            value=coverage_factor,
            weight=0.8,
            description=f"Data coverage percentage ({coverage_percent:.1f}%)"
        ))
    
    # Validation score
    if validation_score is not None:
        factors.append(ConfidenceFactor(
            name="validation",
            type=ConfidenceFactorType.VALIDATION_RESULT,
            value=validation_score,
            weight=1.2,
            description="Validation score for results"
        ))
    
    # Complexity factor
    complexity_map = {
        "low": 0.9,  # Simpler analyses can be more reliable
        "medium": 0.75,
        "high": 0.6  # More complex analyses may be less reliable
    }
    
    complexity_factor = complexity_map.get(complexity.lower(), 0.75)
    
    factors.append(ConfidenceFactor(
        name="complexity",
        type=ConfidenceFactorType.HEURISTIC,
        value=complexity_factor,
        weight=0.4,
        description=f"Analysis complexity ({complexity})"
    ))
    
    # Consistency score
    if consistency_score is not None:
        factors.append(ConfidenceFactor(
            name="consistency",
            type=ConfidenceFactorType.CONSISTENCY,
            value=consistency_score,
            weight=0.9,
            description="Internal consistency of results"
        ))
    
    # Analysis type expertise factor
    # This is a placeholder - in a real implementation, you might
    # have a more sophisticated way to determine expertise
    expertise_map = {
        "code": 0.85,
        "text": 0.9,
        "data": 0.8,
        "security": 0.75,
        "performance": 0.7,
        "default": 0.65
    }
    
    expertise_factor = 0.65  # Default
    for key, value in expertise_map.items():
        if key in analysis_type.lower():
            expertise_factor = value
            break
    
    factors.append(ConfidenceFactor(
        name="expertise",
        type=ConfidenceFactorType.EXPERTISE_LEVEL,
        value=expertise_factor,
        weight=0.6,
        description=f"Tool expertise in {analysis_type}"
    ))
    
    # Calculate overall confidence
    confidence = ConfidenceCalculator.calculate(factors)
    details["factors"] = factors
    
    return confidence, details


# Composite operation confidence calculators

def calculate_composite_confidence(
    operation_confidences: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
    dependencies: Optional[Dict[str, List[str]]] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate confidence score for a composite operation.
    
    Args:
        operation_confidences: Dictionary mapping operation names to confidence scores
        weights: Optional dictionary mapping operation names to weights
        dependencies: Optional dictionary mapping operations to dependencies
        
    Returns:
        Tuple of (confidence score, details dictionary)
    """
    if not operation_confidences:
        return 0.0, {"error": "No operations provided"}
    
    # Use default weights if not provided
    if weights is None:
        weights = {op: 1.0 for op in operation_confidences}
    else:
        # Ensure all operations have weights
        for op in operation_confidences:
            if op not in weights:
                weights[op] = 1.0
    
    factors = []
    details = {}
    
    # Add each operation as a confidence factor
    for op, confidence in operation_confidences.items():
        factor = ConfidenceFactor(
            name=op,
            type=ConfidenceFactorType.HEURISTIC,
            value=confidence,
            weight=weights.get(op, 1.0),
            description=f"Confidence of {op} operation"
        )
        factors.append(factor)
    
    # Apply dependency adjustment
    if dependencies:
        for op, deps in dependencies.items():
            if op in operation_confidences:
                # A operation's confidence cannot be higher than its lowest dependency
                for dep in deps:
                    if dep in operation_confidences:
                        dep_confidence = operation_confidences[dep]
                        op_confidence = operation_confidences[op]
                        
                        if op_confidence > dep_confidence:
                            # Adjust weight to reflect dependency impact
                            dep_factor = ConfidenceFactor(
                                name=f"{op}_dep_{dep}",
                                type=ConfidenceFactorType.HEURISTIC,
                                value=dep_confidence,
                                weight=weights.get(op, 1.0) * 0.5,  # Half the weight of the main operation
                                description=f"Dependency of {op} on {dep}"
                            )
                            factors.append(dep_factor)
    
    # Calculate overall confidence
    confidence = ConfidenceCalculator.calculate(factors)
    details["factors"] = factors
    
    # Provide operation breakdown
    details["operations"] = {
        op: {
            "confidence": conf,
            "weight": weights.get(op, 1.0)
        }
        for op, conf in operation_confidences.items()
    }
    
    return confidence, details