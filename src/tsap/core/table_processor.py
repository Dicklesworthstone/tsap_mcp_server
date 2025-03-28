"""
Table processor for handling tabular data across various formats.

This module provides functionality for processing, transforming, and analyzing
data in tabular formats like CSV, Excel, etc.
"""

import os
import csv
import tempfile
from typing import Dict, List, Optional, Union

import tsap.utils.logging as logging
from tsap.core.base import BaseCoreTool, register_tool
from tsap.utils.errors import TSAPError
from tsap.mcp.models import TableProcessParams, TableProcessResult


class TableProcessingError(TSAPError):
    """Exception raised for errors in table processing."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.details = details or {}


class TableProcessor(BaseCoreTool):
    """Tool for processing tabular data."""
    
    def __init__(self) -> None:
        """Initialize the table processor tool."""
        super().__init__(name="table_processor")
        self._supported_formats = ["csv", "tsv", "txt"]
        
        # Try to import optional dependencies
        self._pandas_available = False
        self._openpyxl_available = False
        
        try:
            self._pandas_available = True
            self._supported_formats.extend(["xlsx", "xls", "parquet", "json"])
            logging.debug("Pandas is available for advanced table processing", component="table_processor")
        except ImportError:
            logging.debug("Pandas is not available; some table processing features will be limited", component="table_processor")
        
        try:
            self._openpyxl_available = True
            if "xlsx" not in self._supported_formats:
                self._supported_formats.append("xlsx")
            logging.debug("OpenPyXL is available for Excel processing", component="table_processor")
        except ImportError:
            logging.debug("OpenPyXL is not available; Excel processing will be limited", component="table_processor")
    
    def _get_file_format(self, file_path: str) -> str:
        """Determine the format of a file based on its extension."""
        _, ext = os.path.splitext(file_path)
        if not ext:
            raise TableProcessingError(f"Could not determine file format for: {file_path}")
        
        ext = ext.lower().lstrip(".")
        if ext in self._supported_formats:
            return ext
        else:
            raise TableProcessingError(
                f"Unsupported file format: {ext}",
                {"supported_formats": self._supported_formats}
            )
    
    def _read_csv(self, file_path: str, params: Dict) -> List[Dict]:
        """Read data from a CSV file."""
        delimiter = params.get("delimiter", ",")
        quotechar = params.get("quotechar", '"')
        encoding = params.get("encoding", "utf-8")
        has_header = params.get("has_header", True)
        
        try:
            with open(file_path, "r", encoding=encoding) as f:
                # Read the CSV into a list of dictionaries
                if has_header:
                    reader = csv.DictReader(f, delimiter=delimiter, quotechar=quotechar)
                    result = list(reader)
                else:
                    reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
                    result = []
                    for row in reader:
                        result.append(dict((str(i), v) for i, v in enumerate(row)))
                
                return result
        except Exception as e:
            raise TableProcessingError(f"Error reading CSV file: {str(e)}", {"file": file_path})
    
    def _read_pandas(self, file_path: str, format: str, params: Dict) -> List[Dict]:
        """Read data using pandas."""
        if not self._pandas_available:
            raise TableProcessingError(
                "Pandas is required for this file format but is not installed",
                {"format": format}
            )
        
        try:
            import pandas as pd
            
            # Set pandas options for reading
            read_kwargs = {}
            if params.get("header") is not None:
                read_kwargs["header"] = params.get("header")
            
            if params.get("sheet_name") and format in ["xlsx", "xls"]:
                read_kwargs["sheet_name"] = params.get("sheet_name")
            
            if params.get("skiprows"):
                read_kwargs["skiprows"] = params.get("skiprows")
                
            if params.get("encoding"):
                read_kwargs["encoding"] = params.get("encoding")
            
            # Read based on format
            if format in ["csv", "tsv", "txt"]:
                delimiter = params.get("delimiter", "," if format == "csv" else "\t")
                df = pd.read_csv(file_path, delimiter=delimiter, **read_kwargs)
            elif format in ["xlsx", "xls"]:
                df = pd.read_excel(file_path, **read_kwargs)
            elif format == "parquet":
                df = pd.read_parquet(file_path, **read_kwargs)
            elif format == "json":
                df = pd.read_json(file_path, **read_kwargs)
            else:
                raise TableProcessingError(f"Unsupported format for pandas: {format}")
            
            # Convert DataFrame to list of dictionaries
            result = df.to_dict(orient="records")
            return result
            
        except Exception as e:
            raise TableProcessingError(f"Error reading file with pandas: {str(e)}", {"file": file_path})
    
    def _transform_data(self, data: List[Dict], params: Dict) -> List[Dict]:
        """Apply transformations to the data."""
        result = data.copy()
        
        # Filter rows
        if params.get("filter_expr"):
            filter_expr = params.get("filter_expr")
            try:
                # Basic expression evaluation for filtering
                # Note: In a real implementation, this would need more robust safety measures
                filter_func = eval(f"lambda row: {filter_expr}")
                result = [row for row in result if filter_func(row)]
            except Exception as e:
                raise TableProcessingError(f"Error in filter expression: {str(e)}")
        
        # Select columns
        if params.get("columns"):
            columns = params.get("columns")
            result = [{k: row.get(k) for k in columns if k in row} for row in result]
        
        # Sort data
        if params.get("sort_by"):
            sort_key = params.get("sort_by")
            reverse = params.get("sort_desc", False)
            try:
                result.sort(key=lambda x: x.get(sort_key), reverse=reverse)
            except Exception as e:
                raise TableProcessingError(f"Error sorting data: {str(e)}")
        
        # Transform columns
        if params.get("transformations"):
            transformations = params.get("transformations")
            for column, transform_expr in transformations.items():
                try:
                    transform_func = eval(f"lambda val, row: {transform_expr}")
                    for row in result:
                        if column in row:
                            row[column] = transform_func(row.get(column), row)
                except Exception as e:
                    raise TableProcessingError(f"Error in column transformation: {str(e)}")
        
        # Add computed columns
        if params.get("computed_columns"):
            for column, expr in params.get("computed_columns").items():
                try:
                    compute_func = eval(f"lambda row: {expr}")
                    for row in result:
                        row[column] = compute_func(row)
                except Exception as e:
                    raise TableProcessingError(f"Error in computed column: {str(e)}")
        
        return result
    
    def _analyze_data(self, data: List[Dict], params: Dict) -> Dict:
        """Generate analytics for the data."""
        analysis = {
            "row_count": len(data),
            "column_stats": {},
        }
        
        # Skip analysis if no data
        if not data:
            return analysis
        
        # Get all column names
        columns = set()
        for row in data:
            columns.update(row.keys())
        
        # Analyze each column
        for column in columns:
            values = [row.get(column) for row in data if column in row]
            non_null_values = [v for v in values if v is not None and v != ""]
            
            col_stats = {
                "count": len(values),
                "null_count": len(values) - len(non_null_values),
                "null_percentage": (len(values) - len(non_null_values)) / len(values) * 100 if values else 0,
                "unique_count": len(set(non_null_values)),
            }
            
            # Try numeric statistics if possible
            numeric_values = []
            for v in non_null_values:
                try:
                    if isinstance(v, (int, float)):
                        numeric_values.append(v)
                    else:
                        numeric_values.append(float(v))
                except (ValueError, TypeError):
                    pass
            
            if numeric_values:
                col_stats["min"] = min(numeric_values)
                col_stats["max"] = max(numeric_values)
                col_stats["sum"] = sum(numeric_values)
                col_stats["mean"] = sum(numeric_values) / len(numeric_values)
                
                if len(numeric_values) >= 2:
                    try:
                        import statistics
                        col_stats["median"] = statistics.median(numeric_values)
                        col_stats["stdev"] = statistics.stdev(numeric_values)
                    except (ImportError, statistics.StatisticsError):
                        # Handle statistics module not available or insufficient data
                        pass
            
            analysis["column_stats"][column] = col_stats
        
        return analysis
    
    def _write_output(self, data: List[Dict], format: str, params: Dict) -> str:
        """Write processed data to a file and return its path."""
        output_path = params.get("output_path")
        
        # Generate a temporary file if none specified
        if not output_path:
            suffix = f".{format}"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
                output_path = temp.name
                logging.debug(f"Created temporary output file: {output_path}", component="table_processor")
        
        if format == "csv":
            delimiter = params.get("delimiter", ",")
            quotechar = params.get("quotechar", '"')
            encoding = params.get("encoding", "utf-8")
            
            # Get all possible fieldnames from all rows
            fieldnames = set()
            for row in data:
                fieldnames.update(row.keys())
            fieldnames = sorted(fieldnames)
            
            try:
                with open(output_path, "w", newline="", encoding=encoding) as f:
                    writer = csv.DictWriter(
                        f, 
                        fieldnames=fieldnames,
                        delimiter=delimiter,
                        quotechar=quotechar,
                        quoting=csv.QUOTE_MINIMAL
                    )
                    writer.writeheader()
                    writer.writerows(data)
            except Exception as e:
                raise TableProcessingError(f"Error writing to CSV: {str(e)}")
                
        elif format in ["xlsx", "xls", "parquet", "json"]:
            if not self._pandas_available:
                raise TableProcessingError(
                    f"Pandas is required to write {format} format but is not installed"
                )
            
            try:
                import pandas as pd
                df = pd.DataFrame(data)
                
                if format in ["xlsx", "xls"]:
                    sheet_name = params.get("sheet_name", "Sheet1")
                    df.to_excel(output_path, sheet_name=sheet_name, index=False)
                elif format == "parquet":
                    df.to_parquet(output_path, index=False)
                elif format == "json":
                    orient = params.get("json_orient", "records")
                    df.to_json(output_path, orient=orient)
            except Exception as e:
                raise TableProcessingError(f"Error writing to {format}: {str(e)}")
        else:
            raise TableProcessingError(f"Unsupported output format: {format}")
        
        return output_path
    
    def process(self, params: Union[Dict, TableProcessParams]) -> TableProcessResult:
        """Process tabular data."""
        if isinstance(params, dict):
            # Convert to TableProcessParams if needed
            # In a real implementation, this would need more thorough validation
            pass
        
        with self._measure_execution_time():
            input_path = params.get("input_path")
            input_data = params.get("input_data")
            input_format = params.get("input_format")
            output_format = params.get("output_format")
            transform_params = params.get("transform", {})
            analyze = params.get("analyze", False)
            
            # Load data from either file or directly provided data
            data = []
            if input_path:
                # Determine format if not explicitly specified
                if not input_format:
                    input_format = self._get_file_format(input_path)
                
                # Read data based on format
                if input_format in ["csv", "tsv", "txt"]:
                    data = self._read_csv(input_path, params)
                else:
                    # Use pandas for other formats
                    data = self._read_pandas(input_path, input_format, params)
            
            elif input_data:
                # Direct data input (assumed to be a list of dictionaries)
                if isinstance(input_data, list):
                    data = input_data
                else:
                    raise TableProcessingError("Input data must be a list of dictionaries")
            else:
                raise TableProcessingError("Either input_path or input_data must be provided")
            
            # Apply transformations
            processed_data = self._transform_data(data, transform_params)
            
            # Generate analytics if requested
            analysis_results = {}
            if analyze:
                analysis_results = self._analyze_data(processed_data, params)
            
            # Write output if format specified
            output_path = None
            if output_format:
                output_path = self._write_output(processed_data, output_format, params)
            
            # Prepare result
            result = TableProcessResult(
                success=True,
                data=processed_data[:params.get("max_rows_return", 1000)] if params.get("max_rows_return") else processed_data,
                row_count=len(processed_data),
                analysis=analysis_results,
                output_path=output_path,
                execution_time=self.get_statistics().get("execution_time", 0)
            )
            
            return result


@register_tool("table_processor")
def get_table_processor() -> TableProcessor:
    """Get a TableProcessor instance."""
    return TableProcessor()


def process_table(params: Union[Dict, TableProcessParams]) -> TableProcessResult:
    """Process a table with the specified parameters."""
    processor = get_table_processor()
    return processor.process(params)