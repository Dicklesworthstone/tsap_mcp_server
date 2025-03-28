"""
TSAP Security Audit Template.

This template provides a comprehensive security audit workflow for code repositories,
searching for common security issues and vulnerabilities.
"""

import time
from typing import Dict, List, Any

from tsap.utils.logging import logger
from tsap.templates.base import (
    Template, 
    TemplateResult,
    TemplateError,
    template_decorator,
    parameter,
)
from tsap.composite.parallel import parallel_search, ParallelSearchParams, SearchPattern
from tsap.composite.context import extract_context, ContextExtractParams


@template_decorator(
    name="Security Audit",
    description="Performs a comprehensive security audit on code repositories",
    version="1.0.0",
    author="TSAP Team",
)
class SecurityAuditTemplate(Template):
    """Template for performing security audits on code repositories."""
    
    # Define template parameters
    parameters = [
        parameter(
            name="paths",
            type="list",
            description="List of file or directory paths to audit",
            required=True,
        ),
        parameter(
            name="file_types",
            type="list",
            description="List of file extensions to audit (e.g., 'py', 'js')",
            required=False,
            default=None,
        ),
        parameter(
            name="exclude_patterns",
            type="list",
            description="List of patterns to exclude from the audit",
            required=False,
            default=None,
        ),
        parameter(
            name="audit_types",
            type="list",
            description="Types of security issues to audit",
            required=False,
            default=["secrets", "vulnerabilities", "injection", "authentication"],
            options=["secrets", "vulnerabilities", "injection", "authentication", "authorization", "xss", "csrf", "file_access", "all"],
        ),
        parameter(
            name="context_lines",
            type="int",
            description="Number of context lines to include with each finding",
            required=False,
            default=5,
        ),
        parameter(
            name="severity_threshold",
            type="str",
            description="Minimum severity level to report",
            required=False,
            default="medium",
            options=["low", "medium", "high", "critical"],
        ),
        parameter(
            name="max_files",
            type="int",
            description="Maximum number of files to analyze",
            required=False,
            default=None,
        ),
    ]
    
    def __init__(self):
        """Initialize the security audit template."""
        super().__init__()
        
        # Define security patterns
        self.patterns = self._load_security_patterns()
    
    def _load_security_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load security patterns for different audit types.
        
        Returns:
            Dictionary mapping audit types to patterns
        """
        # These are basic examples; in a real implementation, these would be more comprehensive
        # and possibly loaded from a file or database
        return {
            "secrets": [
                {
                    "pattern": "(?i)api_?key",
                    "description": "Potential API key",
                    "severity": "high",
                },
                {
                    "pattern": "(?i)secret",
                    "description": "Potential secret",
                    "severity": "high",
                },
                {
                    "pattern": "(?i)password",
                    "description": "Potential password",
                    "severity": "high",
                },
                {
                    "pattern": "(?i)token",
                    "description": "Potential token",
                    "severity": "medium",
                },
            ],
            "vulnerabilities": [
                {
                    "pattern": "(?i)eval\\s*\\(",
                    "description": "Use of eval() function",
                    "severity": "high",
                },
                {
                    "pattern": "(?i)exec\\s*\\(",
                    "description": "Use of exec() function",
                    "severity": "high",
                },
                {
                    "pattern": "(?i)subprocess\\.(?:call|run|Popen)",
                    "description": "Subprocess execution",
                    "severity": "medium",
                },
                {
                    "pattern": "(?i)os\\.system",
                    "description": "OS command execution",
                    "severity": "high",
                },
            ],
            "injection": [
                {
                    "pattern": "(?i)sql.*?\\+|\\+.*?sql",
                    "description": "Potential SQL injection",
                    "severity": "critical",
                },
                {
                    "pattern": "(?i)execute\\s*\\(.*?\\$\\{|\\$\\{.*?execute",
                    "description": "Potential command injection",
                    "severity": "critical",
                },
                {
                    "pattern": "(?i)(?:fromJson|JSON\\.parse)\\s*\\(",
                    "description": "Potential JSON injection",
                    "severity": "medium",
                },
            ],
            "authentication": [
                {
                    "pattern": "(?i)md5\\s*\\(",
                    "description": "Use of weak hash function (MD5)",
                    "severity": "high",
                },
                {
                    "pattern": "(?i)sha1\\s*\\(",
                    "description": "Use of weak hash function (SHA1)",
                    "severity": "medium",
                },
                {
                    "pattern": "(?i)auth.*?bypass|bypass.*?auth",
                    "description": "Potential authentication bypass",
                    "severity": "critical",
                },
            ],
            "authorization": [
                {
                    "pattern": "(?i)role.*?check|permission.*?check",
                    "description": "Authorization check",
                    "severity": "low",
                },
                {
                    "pattern": "(?i)admin.*?role|role.*?admin",
                    "description": "Admin role reference",
                    "severity": "low",
                },
            ],
            "xss": [
                {
                    "pattern": "(?i)innerHTML|document\\.write\\s*\\(",
                    "description": "Potential XSS vulnerability",
                    "severity": "high",
                },
                {
                    "pattern": "(?i)dangerouslySetInnerHTML",
                    "description": "Potential XSS vulnerability in React",
                    "severity": "high",
                },
            ],
            "csrf": [
                {
                    "pattern": "(?i)csrf.*?token|token.*?csrf",
                    "description": "CSRF token reference",
                    "severity": "low",
                },
                {
                    "pattern": "(?i)same.*?origin|origin.*?check",
                    "description": "Same-origin check",
                    "severity": "low",
                },
            ],
            "file_access": [
                {
                    "pattern": "(?i)open\\s*\\(.*?[\"']w[\"']",
                    "description": "File write operation",
                    "severity": "medium",
                },
                {
                    "pattern": "(?i)fs\\.(?:write|append)File",
                    "description": "File write operation (Node.js)",
                    "severity": "medium",
                },
                {
                    "pattern": "(?i)path\\.(?:join|resolve)\\s*\\(.*?\\.\\.",
                    "description": "Potential path traversal",
                    "severity": "high",
                },
            ],
        }
    
    def _get_severity_level(self, severity: str) -> int:
        """Convert severity string to numeric level.
        
        Args:
            severity: Severity string ('low', 'medium', 'high', 'critical')
            
        Returns:
            Numeric severity level
        """
        levels = {
            "low": 1,
            "medium": 2,
            "high": 3,
            "critical": 4,
        }
        return levels.get(severity.lower(), 0)
    
    def _filter_findings_by_severity(
        self, findings: List[Dict[str, Any]], threshold: str
    ) -> List[Dict[str, Any]]:
        """Filter findings by severity threshold.
        
        Args:
            findings: List of findings
            threshold: Minimum severity threshold
            
        Returns:
            Filtered findings
        """
        threshold_level = self._get_severity_level(threshold)
        return [
            finding for finding in findings
            if self._get_severity_level(finding.get("severity", "low")) >= threshold_level
        ]
    
    def _create_search_patterns(self, audit_types: List[str]) -> List[SearchPattern]:
        """Create search patterns for the specified audit types.
        
        Args:
            audit_types: List of audit types
            
        Returns:
            List of search patterns
        """
        search_patterns = []
        
        # Handle "all" audit type
        if "all" in audit_types:
            audit_types = list(self.patterns.keys())
        
        # Create patterns for each audit type
        for audit_type in audit_types:
            if audit_type in self.patterns:
                for pattern_info in self.patterns[audit_type]:
                    search_patterns.append(
                        SearchPattern(
                            pattern=pattern_info["pattern"],
                            description=f"{audit_type}: {pattern_info['description']}",
                            is_regex=True,
                            case_sensitive=False,
                            metadata={
                                "audit_type": audit_type,
                                "severity": pattern_info["severity"],
                            },
                        )
                    )
        
        return search_patterns
    
    async def execute(self, params: Dict[str, Any]) -> TemplateResult:
        """Execute the security audit template.
        
        Args:
            params: Template parameters
            
        Returns:
            Template result
            
        Raises:
            TemplateError: If execution fails
        """
        # Start time
        start_time = time.time()
        
        try:
            # Extract parameters
            paths = params["paths"]
            file_types = params.get("file_types")
            exclude_patterns = params.get("exclude_patterns")
            audit_types = params.get("audit_types", ["secrets", "vulnerabilities", "injection", "authentication"])
            context_lines = params.get("context_lines", 5)
            severity_threshold = params.get("severity_threshold", "medium")
            max_files = params.get("max_files")
            
            # Create search patterns
            search_patterns = self._create_search_patterns(audit_types)
            
            # Create parallel search parameters
            parallel_params = ParallelSearchParams(
                patterns=search_patterns,
                paths=paths,
                file_types=file_types,
                exclude_patterns=exclude_patterns,
                max_files=max_files,
                max_matches_per_file=None,
                max_total_matches=None,
                context_lines=0,  # We'll extract context separately
                case_sensitive=False,
                multiline=True,
            )
            
            # Perform parallel search
            logger.info(f"Running security audit on {len(paths)} paths with {len(search_patterns)} patterns")
            search_results = await parallel_search(parallel_params)
            
            # Extract findings from search results
            findings = []
            for match in search_results.get("matches", []):
                # Extract metadata from pattern
                pattern_index = match.get("pattern_index", 0)
                if 0 <= pattern_index < len(search_patterns):
                    pattern = search_patterns[pattern_index]
                    metadata = pattern.metadata or {}
                else:
                    metadata = {}
                
                # Create finding
                finding = {
                    "file_path": match.get("file_path", ""),
                    "line_number": match.get("line_number", 0),
                    "column": match.get("column", 0),
                    "pattern": match.get("pattern", ""),
                    "content": match.get("content", ""),
                    "audit_type": metadata.get("audit_type", "unknown"),
                    "description": match.get("description", ""),
                    "severity": metadata.get("severity", "low"),
                }
                
                findings.append(finding)
            
            # Filter findings by severity
            filtered_findings = self._filter_findings_by_severity(findings, severity_threshold)
            
            # Extract context for each finding
            enriched_findings = []
            for finding in filtered_findings:
                # Extract context
                context_params = ContextExtractParams(
                    file_path=finding["file_path"],
                    line_number=finding["line_number"],
                    context_type="line",
                    context_lines=context_lines,
                )
                
                try:
                    context_result = await extract_context(context_params)
                    contexts = context_result.get("contexts", [])
                    
                    if contexts:
                        context = contexts[0]
                        # Add context to finding
                        finding["context"] = {
                            "content": context.get("content", ""),
                            "start_line": context.get("start_line", finding["line_number"]),
                            "end_line": context.get("end_line", finding["line_number"]),
                        }
                except Exception as e:
                    logger.warning(f"Failed to extract context for finding: {e}")
                    finding["context"] = {
                        "content": finding["content"],
                        "start_line": finding["line_number"],
                        "end_line": finding["line_number"],
                    }
                
                enriched_findings.append(finding)
            
            # Group findings by file
            findings_by_file = {}
            for finding in enriched_findings:
                file_path = finding["file_path"]
                if file_path not in findings_by_file:
                    findings_by_file[file_path] = []
                findings_by_file[file_path].append(finding)
            
            # Group findings by audit type
            findings_by_type = {}
            for finding in enriched_findings:
                audit_type = finding["audit_type"]
                if audit_type not in findings_by_type:
                    findings_by_type[audit_type] = []
                findings_by_type[audit_type].append(finding)
            
            # Group findings by severity
            findings_by_severity = {}
            for finding in enriched_findings:
                severity = finding["severity"]
                if severity not in findings_by_severity:
                    findings_by_severity[severity] = []
                findings_by_severity[severity].append(finding)
            
            # Calculate statistics
            stats = {
                "total_findings": len(enriched_findings),
                "files_with_findings": len(findings_by_file),
                "findings_by_severity": {
                    severity: len(findings)
                    for severity, findings in findings_by_severity.items()
                },
                "findings_by_type": {
                    audit_type: len(findings)
                    for audit_type, findings in findings_by_type.items()
                },
            }
            
            # Create result summary
            summary = {
                "total_findings": stats["total_findings"],
                "severity_counts": stats["findings_by_severity"],
                "top_files": sorted(
                    [(file, len(findings)) for file, findings in findings_by_file.items()],
                    key=lambda x: x[1],
                    reverse=True,
                )[:5],
                "audit_types": list(findings_by_type.keys()),
            }
            
            # Create template result
            result = TemplateResult(
                template_id=self.id,
                template_name=self.name,
                status="success",
                message=f"Security audit completed with {stats['total_findings']} findings",
                execution_time=time.time() - start_time,
                timestamp=time.time(),
                parameters=params,
                results={
                    "findings": enriched_findings,
                    "findings_by_file": findings_by_file,
                    "findings_by_type": findings_by_type,
                    "findings_by_severity": findings_by_severity,
                    "stats": stats,
                    "summary": summary,
                },
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Security audit failed: {e}")
            # Re-raise as TemplateError
            raise TemplateError(
                f"Security audit failed: {str(e)}",
                template_id=self.id,
                template_name=self.name,
            ) from e


# Convenience function to run the template
async def run_security_audit(params: Dict[str, Any]) -> TemplateResult:
    """Run the security audit template.
    
    Args:
        params: Template parameters
        
    Returns:
        Template result
        
    Raises:
        TemplateError: If execution fails
    """
    from tsap.templates.base import get_template_runner
    
    template_runner = get_template_runner()
    return await template_runner.run_template_by_name("Security Audit", params)