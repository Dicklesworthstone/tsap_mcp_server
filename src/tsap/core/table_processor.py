"""
Table processor for handling tabular data across various formats.

This module provides functionality for processing, transforming, and analyzing
data in tabular formats like CSV, Excel, etc.
"""

import os
import csv
import tempfile
import time
from typing import Dict, List, Optional

import tsap.utils.logging as logging
from tsap.core.base import BaseCoreTool, register_tool
from tsap.utils.errors import TSAPError
from tsap.mcp.models import TableProcessParams, TableProcessResult, TableTransformParams


# Define a dictionary of safe built-ins to allow in eval expressions
SAFE_BUILTINS = {
    "float": float,
    "int": int,
    "str": str,
    "round": round,
    "len": len,
    "abs": abs,
    "max": max,
    "min": min,
    "sum": sum,
    "any": any,
    "all": all,
    "bool": bool,
    "dict": dict,
    "list": list,
    "tuple": tuple,
    "set": set,
}


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
    
    def _read_csv(self, file_path: str, **kwargs) -> List[Dict]:
        """Read data from a CSV file."""
        if "delimiter" not in kwargs: # Only default if not provided
             kwargs["delimiter"] = ","

        logging.info(f"Reading CSV/TSV file: {file_path} with delimiter: '{kwargs.get('delimiter')}' encoding: {kwargs.get('encoding')}", component="table_processor")

        try:
            with open(file_path, "r", encoding=kwargs.get("encoding", "utf-8")) as f:
                if kwargs.get("has_header", True):
                    reader = csv.DictReader(f, delimiter=kwargs.get("delimiter", ","), quotechar=kwargs.get("quotechar", '"'))
                    result = list(reader)
                else:
                    reader = csv.reader(f, delimiter=kwargs.get("delimiter", ","), quotechar=kwargs.get("quotechar", '"'))
                    result = [dict((str(i), v) for i, v in enumerate(row)) for row in reader]
                return result
        except Exception as e:
            raise TableProcessingError(f"Error reading CSV file: {str(e)}", {"file": file_path})
    
    def _read_pandas(self, file_path: str, format: str, **kwargs) -> List[Dict]:
        """Read data using pandas."""
        if not self._pandas_available:
            raise TableProcessingError(
                "Pandas is required for this file format but is not installed",
                {"format": format}
            )
        
        try:
            import pandas as pd
            read_kwargs = {}
            if "header" in kwargs: 
                read_kwargs["header"] = kwargs["header"]
            if "sheet_name" in kwargs and format in ["xlsx", "xls"]: 
                read_kwargs["sheet_name"] = kwargs["sheet_name"]
            if "skiprows" in kwargs: 
                read_kwargs["skiprows"] = kwargs["skiprows"]
            if "encoding" in kwargs: 
                read_kwargs["encoding"] = kwargs["encoding"]

            if format in ["csv", "tsv", "txt"]:
                delimiter = kwargs.get("delimiter", "," if format == "csv" else "\t")
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
    
    def _transform_data(self, data: List[Dict], transform_options: Optional[TableTransformParams]) -> List[Dict]:
        """Apply transformations to the data."""
        if not transform_options:
             return data # No transformations requested

        result = data.copy()
        logging.debug(f"Applying transformations with options: {transform_options.model_dump(exclude_unset=True).keys()}", component="table_processor")

        # Filter rows
        filter_expr = transform_options.filter_expr
        if filter_expr:
            try:
                # Refined filter logic with row-by-row eval
                logging.debug(f"Applying filter: {filter_expr}", component="table_processor")
                filtered_result = []
                for row in result:
                    try:
                         # Provide row and safe builtins in the eval context
                         if eval(filter_expr, {"__builtins__": SAFE_BUILTINS}, {"row": row}):
                              filtered_result.append(row)
                    except Exception as eval_err:
                         logging.warning(f"Filter eval error on row: {eval_err}", component="table_processor")
                         continue # Skip row on error
                logging.debug(f"Filter applied. Rows before: {len(result)}, Rows after: {len(filtered_result)}", component="table_processor")
                result = filtered_result # Update result with filtered data
            except Exception as e:
                logging.error(f"Error applying filter expression '{filter_expr}': {e}", component="table_processor")
                raise TableProcessingError(f"Error in filter expression: {str(e)}")

        # Sort data
        sort_key = transform_options.sort_by
        if sort_key:
            reverse = transform_options.sort_desc
            logging.debug(f"Sorting by '{sort_key}', descending={reverse}", component="table_processor")
            try:
                def get_sort_key_func(row):
                    val = row.get(sort_key)
                    try:
                        return float(val) if val is not None else float('-inf')
                    except (ValueError, TypeError):
                        return str(val)
                result.sort(key=get_sort_key_func, reverse=reverse)
            except Exception as e:
                logging.error(f"Error sorting data by key '{sort_key}': {e}", component="table_processor")
                raise TableProcessingError(f"Error sorting data: {str(e)}")

        # Add computed columns
        computed_columns = transform_options.computed_columns
        if computed_columns:
            logging.debug(f"Adding computed columns: {list(computed_columns.keys())}", component="table_processor")
            temp_result = []
            # Use safe builtins for computed columns as well
            safe_globals = {"__builtins__": SAFE_BUILTINS}
            for row in result:
                new_row = row.copy()
                row_context = {"row": new_row} # Provide row context
                for column, expr in computed_columns.items():
                    try:
                        new_row[column] = eval(expr, safe_globals, row_context)
                    except Exception as e:
                         logging.warning(f"Computed column '{column}' eval error: {e}", component="table_processor")
                         new_row[column] = "ERROR"
                temp_result.append(new_row)
            result = temp_result # Update result with computed columns

        # Select columns (apply LAST)
        columns_to_select = transform_options.columns
        if columns_to_select:
            logging.debug(f"Selecting final columns: {columns_to_select}", component="table_processor")
            final_result = []
            for row in result:
                selected_row = {k: row.get(k) for k in columns_to_select if k in row}
                final_result.append(selected_row)
            result = final_result # Update result with selected columns
            logging.debug(f"Columns selected. First row keys: {list(result[0].keys()) if result else 'N/A'}", component="table_processor")

        return result
    
    def _analyze_data(self, data: List[Dict]) -> Dict:
        """Generate analytics for the data."""
        logging.debug(f"Starting data analysis for {len(data)} rows.", component="table_processor")
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
        logging.debug(f"Found columns for analysis: {columns}", component="table_processor")
        
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
                    except ImportError:
                        logging.warning("Could not import 'statistics' module for median/stdev calculation.", component="table_processor")
                    except statistics.StatisticsError as stat_err:
                        logging.warning(f"StatisticsError for column '{column}': {stat_err}", component="table_processor")
                        # Handle statistics error (e.g., insufficient data for stdev)
                        pass
                    except Exception as e:
                        logging.error(f"Unexpected error during statistics calculation for column '{column}': {e}", component="table_processor")
            
            analysis["column_stats"][column] = col_stats
            logging.debug(f"Calculated stats for column '{column}': {list(col_stats.keys())}", component="table_processor")
        
        logging.debug(f"Finished data analysis. Result keys: {list(analysis.keys())}", component="table_processor")
        return analysis
    
    def _write_output(self, data: List[Dict], format: str, output_path: Optional[str], **kwargs) -> str:
        """Write processed data to a file and return its path."""
        # Determine output path
        effective_output_path = output_path
        if not effective_output_path:
            suffix = f".{format}"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
                effective_output_path = temp.name
                logging.debug(f"Created temporary output file: {effective_output_path}", component="table_processor")
        
        if format == "csv":
            delimiter = kwargs.get("delimiter", ",")
            quotechar = kwargs.get("quotechar", '"')
            encoding = kwargs.get("encoding", "utf-8")
            
            # Get all possible fieldnames from all rows
            fieldnames = set()
            for row in data:
                fieldnames.update(row.keys())
            fieldnames = sorted(fieldnames)
            
            try:
                with open(effective_output_path, "w", newline="", encoding=encoding) as f:
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
                    sheet_name = kwargs.get("sheet_name", "Sheet1")
                    df.to_excel(effective_output_path, sheet_name=sheet_name, index=False)
                elif format == "parquet":
                    df.to_parquet(effective_output_path, index=False)
                elif format == "json":
                    orient = kwargs.get("json_orient", "records")
                    df.to_json(effective_output_path, orient=orient)
            except Exception as e:
                raise TableProcessingError(f"Error writing to {format}: {str(e)}")
        else:
            raise TableProcessingError(f"Unsupported output format: {format}")
        
        logging.debug(f"Output file generated at: {effective_output_path}", component="table_processor") # Log path
        return effective_output_path
    
    def process(self, params: TableProcessParams) -> TableProcessResult:
        """Process tabular data using Pydantic model for params."""
        logging.info(f"--- Entering TableProcessor.process for file: {params.file_path} ---", component="table_processor")
        start_time = time.perf_counter()

        # Now params is guaranteed to be TableProcessParams type
        # Extract parameters using dot notation
        file_path = params.file_path
        input_data = params.input_data
        input_format = params.input_format
        output_format = params.output_format
        transform_params_model = params.transform # This is a TableTransformParams model or None
        analyze = params.analyze

        # Prepare kwargs for reading/writing based on params model fields
        # EXCLUDE output_path here to prevent passing it twice later
        read_write_kwargs = params.model_dump(exclude={"file_path", "input_data", "transform", "analyze", "max_rows_return", "output_path"}, exclude_unset=True)

        # Load data
        data = []
        if file_path:
            effective_input_format = input_format
            if not effective_input_format:
                effective_input_format = self._get_file_format(file_path)

            if effective_input_format in ["csv", "tsv", "txt"]:
                 # Pass specific options to _read_csv
                 data = self._read_csv(file_path, **read_write_kwargs)
            else:
                 # Pass specific options to _read_pandas
                 data = self._read_pandas(file_path, effective_input_format, **read_write_kwargs)
        elif input_data:
            if isinstance(input_data, list):
                data = input_data
            else:
                raise TableProcessingError("Input data must be a list of dictionaries")
        else:
             raise TableProcessingError("Either file_path or input_data must be provided")

        # Apply transformations passing the TableTransformParams model
        processed_data = self._transform_data(data, transform_params_model)

        # Generate analytics if requested
        analysis_results = {} if not analyze else self._analyze_data(processed_data)

        # Write output if format specified
        final_output_path = None
        if output_format:
            final_output_path = self._write_output(processed_data, output_format, params.output_path, **read_write_kwargs)
            logging.debug(f"Output file generated at: {final_output_path}", component="table_processor")

        # Calculate final shape
        final_row_count = len(processed_data)
        final_columns = list(processed_data[0].keys()) if final_row_count > 0 else []
        final_column_count = len(final_columns)
        logging.debug(f"Final data: {final_row_count} rows, {final_column_count} columns. Columns: {final_columns}", component="table_processor")

        # Limit data for response
        max_rows = params.max_rows_return if params.max_rows_return is not None else 1000 # Default limit
        returned_data = processed_data[:max_rows]
        logging.debug(f"Returning {len(returned_data)} rows in response.", component="table_processor")

        end_time = time.perf_counter()
        duration = end_time - start_time

        # Prepare result
        result = TableProcessResult(
            success=True,
            result=returned_data,
            columns=final_columns,
            row_count=final_row_count,
            column_count=final_column_count,
            analysis=analysis_results,
            output_path=final_output_path,
            execution_time=duration
        )

        return result


@register_tool("table_processor")
def get_table_processor() -> TableProcessor:
    """Get a TableProcessor instance."""
    return TableProcessor()


def process_table(params: TableProcessParams) -> TableProcessResult:
    """Process a table with the specified parameters."""
    processor = get_table_processor()
    return processor.process(params)