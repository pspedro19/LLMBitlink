"""
Database management system for tourism data using CSV/Excel files.
"""

import os
import pandas as pd
from typing import List, Dict, Any, Optional
from pandasql import sqldf
from app.utils.logger import get_logger
from app.utils.config import DATABASE_PATHS

logger = get_logger(__name__)

class CSVDatabaseManager:
    def __init__(self):
        self.dataframes = {}
        self.load_csv_data()

    def load_csv_data(self) -> None:
        try:
            # Check if files exist and log their absolute paths
            missing_files = []
            for name, path in DATABASE_PATHS.items():
                if not path.exists():
                    missing_files.append(path)
                else:
                    logger.info(f"Found file {name} at {path.absolute()}")

            if missing_files:
                raise FileNotFoundError(
                    f"Missing Excel files: {', '.join(str(f) for f in missing_files)}"
                )

            # Define Excel files and their expected columns
            excel_files_info = {
                "realistic_curacao_activities": {
                    "path": "activities",  # Removed .xlsx extension
                    "expected_columns": [
                        "id_activity", "name", "type", "location", "duration_hours",
                        "cost", "rating", "description", "recommended_for",
                        "contact_info", "website", "social_media", "accessibility",
                        "parking", "payment_options", "languages_spoken", "season",
                        "accessible", "languages", "contact_number"
                    ]
                },
                "realistic_curacao_tourist_spots": {
                    "path": "tourist_spots",  # Removed .xlsx extension
                    "expected_columns": [
                        "id_spot", "name", "type", "location", "opening_hours",
                        "entry_fee", "rating", "description", "ideal_for",
                        "contact_info", "website", "social_media", "accessibility",
                        "parking", "payment_options", "languages_spoken", "season",
                        "accessible", "languages", "contact_number"
                    ]
                },
                "realistic_curacao_restaurants": {
                    "path": "restaurants",  # Removed .xlsx extension
                    "expected_columns": [
                        "id_restaurant", "name", "cuisine_type", "average_person_expense", "price_range",
                        "location", "opening_hours", "rating", "recommended_for",
                        "description", "contact_info", "website", "social_media",
                        "accessibility", "parking", "payment_options",
                        "languages_spoken", "season", "accessible", "languages"
                    ]
                },
                "realistic_curacao_nightclubs": {
                    "path": "nightclubs",  # Removed .xlsx extension
                    "expected_columns": [
                        "id_nightclub", "name", "music_type", "average_person_expense", "price_range",
                        "location", "opening_hours", "rating", "recommended_for",
                        "description", "contact_info", "website", "social_media",
                        "accessibility", "parking", "payment_options",
                        "languages_spoken", "season", "accessible", "languages",
                        "dress_code"
                    ]
                },
                "realistic_curacao_packages": {
                    "path": "tourism_packages",  # Removed .xlsx extension
                    "expected_columns": [
                        "id_package", "name", "description", "price",
                        "duration_days", "includes", "categories", "contact_info",
                        "website", "social_media", "accessibility", "parking",
                        "payment_options", "languages_spoken", "season",
                        "accessible", "languages", "contact_number"
                    ]
                }
            }

            # Load each Excel file
            for table_name, file_info in excel_files_info.items():
                try:
                    # Get the file path from DATABASE_PATHS using the correct key
                    file_path = DATABASE_PATHS[file_info["path"]]
                    logger.info(f"Attempting to load file: {file_path.absolute()}")
                    
                    # Check if file exists again (redundant but safe)
                    if not file_path.exists():
                        logger.error(f"File does not exist at path: {file_path.absolute()}")
                        continue

                    # Read the Excel file with additional error handling
                    try:
                        df = pd.read_excel(
                            file_path,
                            dtype=str,
                            na_filter=True,  # Handle missing values
                            engine='openpyxl'  # Explicitly specify engine
                        )
                    except Exception as excel_error:
                        logger.error(f"Excel reading error for {file_path}: {str(excel_error)}")
                        continue

                    if df.empty:
                        logger.error(f"Empty DataFrame loaded from {file_path}")
                        continue

                    logger.info(f"Successfully read {file_path} with {len(df)} rows and {len(df.columns)} columns")

                    # Convert column names to lowercase and strip whitespace
                    df.columns = df.columns.str.lower().str.strip()

                    # Add missing columns with None values
                    for col in file_info["expected_columns"]:
                        col_lower = col.lower()
                        if col_lower not in df.columns:
                            df[col_lower] = None
                            logger.warning(f"Added missing column {col} to {table_name}")

                    # Convert numeric columns to appropriate types with improved error handling
                    numeric_columns = {
                        'cost': 'float64',
                        'entry_fee': 'float64',
                        'price': 'float64',
                        'rating': 'float64',
                        'duration_hours': 'float64',
                        'duration_days': 'int64',
                        'average_person_expense': 'float64'
                    }

                    for col, dtype in numeric_columns.items():
                        if col in df.columns:
                            try:
                                df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
                            except Exception as e:
                                logger.warning(f"Could not convert {col} to {dtype}: {str(e)}")

                    # Store the dataframe
                    self.dataframes[table_name] = df
                    logger.info(f"Successfully loaded {table_name} with {len(df)} rows")

                except Exception as e:
                    logger.error(f"Error processing {table_name}: {str(e)}")
                    continue

            if not self.dataframes:
                raise Exception("No Excel files were successfully loaded")

            # Log summary of loaded data
            logger.info("Data loading summary:")
            for table_name, df in self.dataframes.items():
                logger.info(f"{table_name}: {len(df)} rows, {len(df.columns)} columns")

        except Exception as e:
            logger.error(f"Error in Excel data loading: {str(e)}")
            raise

    def execute_query(self, query: str, params: Optional[List[Any]] = None) -> List[Dict]:
        try:
            # Extraer nombre de tabla
            table_name = query.split('FROM')[1].split('WHERE')[0].strip()
            df = self.dataframes[table_name].copy()
            
            # Procesar WHERE
            if 'WHERE' in query:
                where_clause = query.split('WHERE')[1].split('ORDER BY')[0].strip()
                
                # Procesar condiciones de ubicación
                if 'location LIKE' in where_clause:
                    location_conditions = where_clause.split('AND')[0].strip('()')
                    locations = [
                        loc.split("'%")[1].split("%'")[0] 
                        for loc in location_conditions.split('OR')
                    ]
                    location_mask = df['location'].str.lower().str.contains('|'.join(locations), na=False)
                    df = df[location_mask]
                
                # Procesar condiciones numéricas
                if '<=' in where_clause:
                    for condition in where_clause.split('AND'):
                        if '<=' in condition:
                            field, value = condition.strip().split('<=')
                            field = field.strip()
                            value = float(value.strip())
                            df = df[df[field] <= value]
            
            # Ordenar
            if 'ORDER BY' in query:
                sort_field = query.split('ORDER BY')[1].split('LIMIT')[0].strip().split()[0]
                ascending = 'DESC' not in query
                df = df.sort_values(sort_field, ascending=ascending)
            
            # Límite
            if 'LIMIT' in query:
                limit = int(query.split('LIMIT')[-1].strip())
                df = df.head(limit)

            # Convertir a diccionarios
            records = []
            for _, row in df.iterrows():
                record = {}
                for column in row.index:
                    value = row[column]
                    if pd.isna(value):
                        record[column] = None
                    elif isinstance(value, pd.Timestamp):
                        record[column] = value.isoformat()
                    else:
                        record[column] = value
                records.append(record)

            return records

        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}\nQuery: {query}")
            return []
    
    def get_table_info(self, table_name: str) -> Dict:
        """
        Get enhanced information about a specific table

        Args:
            table_name (str): Name of the table

        Returns:
            Dict: Table information including detailed statistics
        """
        try:
            if table_name not in self.dataframes:
                raise KeyError(f"Table {table_name} not found")

            df = self.dataframes[table_name]
            info = {
                "columns": df.columns.tolist(),
                "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
                "row_count": len(df),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "null_counts": df.isna().sum().to_dict(),
                "duplicate_rows": df.duplicated().sum(),
                "unique_counts": {col: df[col].nunique() for col in df.columns}
            }

            # Add basic statistics for numeric columns
            numeric_stats = {}
            for col in df.select_dtypes(include=['int64', 'float64']).columns:
                numeric_stats[col] = {
                    "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    "median": float(df[col].median()) if not pd.isna(df[col].median()) else None
                }
            info["numeric_stats"] = numeric_stats

            return info

        except Exception as e:
            logger.error(f"Error getting table info for {table_name}: {str(e)}")
            return {}

    def get_column_stats(self, table_name: str, column_name: str) -> Dict:
        """
        Get enhanced statistical information about a specific column

        Args:
            table_name (str): Name of the table
            column_name (str): Name of the column

        Returns:
            Dict: Column statistics with enhanced metrics
        """
        try:
            if table_name not in self.dataframes:
                raise KeyError(f"Table {table_name} not found")

            df = self.dataframes[table_name]
            if column_name not in df.columns:
                raise KeyError(f"Column {column_name} not found in {table_name}")

            column = df[column_name]
            stats = {
                "null_count": int(column.isna().sum()),
                "null_percentage": float(column.isna().mean() * 100),
                "unique_values": int(column.nunique()),
                "unique_percentage": float(column.nunique() / len(column) * 100),
                "data_type": str(column.dtype)
            }

            # Enhanced numeric statistics
            if pd.api.types.is_numeric_dtype(column):
                valid_data = column.dropna()
                if not valid_data.empty:
                    stats.update({
                        "min": float(valid_data.min()),
                        "max": float(valid_data.max()),
                        "mean": float(valid_data.mean()),
                        "median": float(valid_data.median()),
                        "std": float(valid_data.std()),
                        "quartiles": {
                            "25%": float(valid_data.quantile(0.25)),
                            "50%": float(valid_data.quantile(0.50)),
                            "75%": float(valid_data.quantile(0.75))
                        }
                    })

            # Enhanced categorical statistics
            if pd.api.types.is_string_dtype(column):
                value_counts = column.value_counts()
                stats.update({
                    "top_values": value_counts.head(10).to_dict(),
                    "value_distribution": {
                        "unique": len(value_counts),
                        "top_10_percentage": float(value_counts.head(10).sum() / len(column) * 100)
                    },
                    "length_stats": {
                        "min_length": int(column.str.len().min()),
                        "max_length": int(column.str.len().max()),
                        "mean_length": float(column.str.len().mean())
                    } if not column.isna().all() else None
                })

            return stats

        except Exception as e:
            logger.error(f"Error getting column stats for {table_name}.{column_name}: {str(e)}")
            return {}

    def validate_data(self) -> Dict[str, List[str]]:
        """
        Enhanced data quality validation across all tables

        Returns:
            Dict[str, List[str]]: Dictionary of validation issues by table
        """
        validation_issues = {}

        try:
            for table_name, df in self.dataframes.items():
                issues = []

                # Check for duplicate IDs with details
                id_columns = [col for col in df.columns if 'id_' in col]
                for id_col in id_columns:
                    duplicates = df[df[id_col].duplicated()]
                    if not duplicates.empty:
                        duplicate_ids = duplicates[id_col].tolist()
                        issues.append(f"Duplicate IDs found in {id_col}: {duplicate_ids}")

                # Enhanced required value checking
                required_columns = ['name', 'type', 'location']
                for col in required_columns:
                    if col in df.columns:
                        missing_count = df[col].isna().sum()
                        if missing_count > 0:
                            issues.append(f"Missing values in required column {col}: {missing_count} rows")

                # Enhanced rating validation
                if 'rating' in df.columns:
                    invalid_ratings = df[
                        (df['rating'].notna()) & 
                        ((df['rating'] < 0) | (df['rating'] > 5))
                    ]
                    if not invalid_ratings.empty:
                        issues.append(f"Invalid rating values found: {len(invalid_ratings)} rows")

                # Enhanced price/cost validation
                price_columns = ['cost', 'price', 'entry_fee']
                for col in price_columns:
                    if col in df.columns:
                        invalid_prices = df[
                            (df[col].notna()) & 
                            (df[col] < 0)
                        ]
                            
                        if not invalid_prices.empty:
                            issues.append(f"Negative values found in {col}: {len(invalid_prices)} rows")

                # Validate date/time formats if present
                datetime_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                for col in datetime_columns:
                    try:
                        pd.to_datetime(df[col], errors='raise')
                    except Exception:
                        issues.append(f"Invalid datetime format in column {col}")

                # Validate URL formats
                url_columns = ['website', 'social_media']
                for col in url_columns:
                    if col in df.columns:
                        invalid_urls = df[
                            df[col].notna() & 
                            ~df[col].str.contains('^https?://', case=False, na=False)
                        ]
                        if not invalid_urls.empty:
                            issues.append(f"Invalid URL format in {col}: {len(invalid_urls)} rows")

                # Validate consistency of related fields
                if 'duration_hours' in df.columns and 'duration_days' in df.columns:
                    inconsistent = df[
                        (df['duration_hours'].notna()) & 
                        (df['duration_days'].notna()) & 
                        (df['duration_hours'] > df['duration_days'] * 24)
                    ]
                    if not inconsistent.empty:
                        issues.append(f"Inconsistent duration values: {len(inconsistent)} rows")

                if issues:
                    validation_issues[table_name] = issues

            # Add overall data quality metrics
            if self.dataframes:
                overall_metrics = {
                    "total_rows": sum(len(df) for df in self.dataframes.values()),
                    "total_missing_values": sum(df.isna().sum().sum() for df in self.dataframes.values()),
                    "tables_with_issues": len(validation_issues)
                }
                validation_issues["overall_metrics"] = overall_metrics

            return validation_issues

        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            return {"error": [str(e)]}

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get enhanced summary of all loaded data with detailed metrics

        Returns:
            Dict[str, Any]: Summary statistics for all tables
        """
        try:
            summary = {}
            total_memory = 0
            total_rows = 0

            for table_name, df in self.dataframes.items():
                # Calculate memory usage
                memory_usage = df.memory_usage(deep=True).sum()
                total_memory += memory_usage
                total_rows += len(df)

                # Calculate data quality metrics
                null_percentage = (df.isna().sum().sum() / (df.size)) * 100
                duplicate_percentage = (df.duplicated().sum() / len(df)) * 100

                # Get column type distribution
                column_types = df.dtypes.value_counts().to_dict()
                column_types = {str(k): int(v) for k, v in column_types.items()}

                # Calculate value distributions for categorical columns
                categorical_stats = {}
                for col in df.select_dtypes(include=['object']).columns:
                    if df[col].nunique() < 50:  # Only for columns with reasonable cardinality
                        categorical_stats[col] = df[col].value_counts().head(5).to_dict()

                # Calculate numeric column statistics
                numeric_stats = {}
                for col in df.select_dtypes(include=['int64', 'float64']).columns:
                    if not df[col].isna().all():
                        numeric_stats[col] = {
                            "min": float(df[col].min()),
                            "max": float(df[col].max()),
                            "mean": float(df[col].mean()),
                            "std": float(df[col].std())
                        }

                summary[table_name] = {
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "memory_usage_bytes": memory_usage,
                    "memory_usage_mb": memory_usage / 1024 / 1024,
                    "null_percentage": null_percentage,
                    "duplicate_row_percentage": duplicate_percentage,
                    "column_types": column_types,
                    "column_list": list(df.columns),
                    "categorical_stats": categorical_stats,
                    "numeric_stats": numeric_stats,
                    "last_modified": pd.Timestamp.now().isoformat()
                }

            # Add overall statistics
            summary["overall_statistics"] = {
                "total_tables": len(self.dataframes),
                "total_rows": total_rows,
                "total_memory_mb": total_memory / 1024 / 1024,
                "average_rows_per_table": total_rows / len(self.dataframes) if self.dataframes else 0,
                "data_freshness": pd.Timestamp.now().isoformat()
            }

            return summary

        except Exception as e:
            logger.error(f"Error generating data summary: {str(e)}")
            return {}

    def get_table_sample(self, table_name: str, n_rows: int = 5) -> List[Dict]:
        """
        Get a sample of rows from a specific table

        Args:
            table_name (str): Name of the table
            n_rows (int): Number of rows to sample

        Returns:
            List[Dict]: Sample rows as list of dictionaries
        """
        try:
            if table_name not in self.dataframes:
                raise KeyError(f"Table {table_name} not found")

            df = self.dataframes[table_name]
            if df.empty:
                logger.warning(f"Table {table_name} is empty")
                return []

            sample = df.sample(min(n_rows, len(df)))
            return sample.to_dict('records')

        except Exception as e:
            logger.error(f"Error getting sample from {table_name}: {str(e)}")
            return []

    def search_data(self, query: str, tables: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
        """
        Search across all tables or specified tables for matching data

        Args:
            query (str): Search query
            tables (List[str], optional): List of tables to search in

        Returns:
            Dict[str, List[Dict]]: Search results by table
        """
        try:
            if not query:
                logger.warning("Empty search query provided")
                return {}

            results = {}
            tables_to_search = tables if tables else self.dataframes.keys()

            for table_name in tables_to_search:
                if table_name not in self.dataframes:
                    logger.warning(f"Table {table_name} not found, skipping...")
                    continue

                df = self.dataframes[table_name]
                
                # Search in string columns
                string_cols = df.select_dtypes(include=['object']).columns
                mask = pd.Series(False, index=df.index)
                
                for col in string_cols:
                    try:
                        mask |= df[col].fillna('').str.contains(query, case=False, na=False)
                    except Exception as e:
                        logger.warning(f"Error searching in column {col}: {str(e)}")
                        continue

                matches = df[mask]
                if not matches.empty:
                    results[table_name] = matches.to_dict('records')

            if not results:
                logger.info(f"No matches found for query: {query}")

            return results

        except Exception as e:
            logger.error(f"Error searching data: {str(e)}")
            return {} 