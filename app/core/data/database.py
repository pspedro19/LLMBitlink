"""
Database management system for tourism data using CSV/Excel files.
"""

import os
import pandas as pd
from typing import List, Dict, Any, Optional
from pandasql import sqldf


# app/core/data/database.py

from app.utils.logger import get_logger
from app.utils.config import DATABASE_PATHS



logger = get_logger(__name__)

class CSVDatabaseManager:
    def __init__(self):
        self.dataframes = {}
        self.load_csv_data()

    def load_csv_data(self) -> None:
        try:
            # Check if files exist
            missing_files = [
                path for path in DATABASE_PATHS.values() 
                if not path.exists()
            ]

            if missing_files:
                raise FileNotFoundError(
                    f"Missing Excel files: {', '.join(str(f) for f in missing_files)}"
                )


            # Define Excel files and their expected columns
            excel_files_info = {
                "realistic_curacao_activities": {
                    "path": "activities.xlsx",
                    "expected_columns": [
                        "id_activity", "name", "type", "location", "duration_hours",
                        "cost", "rating", "description", "recommended_for",
                        "contact_info", "website", "social_media", "accessibility",
                        "parking", "payment_options", "languages_spoken", "season",
                        "accessible", "languages", "contact_number"
                    ]
                },
                "realistic_curacao_tourist_spots": {
                    "path": "tourist_spots.xlsx",
                    "expected_columns": [
                        "id_spot", "name", "type", "location", "opening_hours",
                        "entry_fee", "rating", "description", "ideal_for",
                        "contact_info", "website", "social_media", "accessibility",
                        "parking", "payment_options", "languages_spoken", "season",
                        "accessible", "languages", "contact_number"
                    ]
                },
                "realistic_curacao_restaurants": {
                    "path": "restaurants.xlsx",
                    "expected_columns": [
                        "id_restaurant", "name", "cuisine_type", "price_range",
                        "location", "opening_hours", "rating", "recommended_for",
                        "description", "contact_info", "website", "social_media",
                        "accessibility", "parking", "payment_options",
                        "languages_spoken", "season", "accessible", "languages"
                    ]
                },
                "realistic_curacao_nightclubs": {
                    "path": "nightclubs.xlsx",
                    "expected_columns": [
                        "id_nightclub", "name", "music_type", "price_range",
                        "location", "opening_hours", "rating", "recommended_for",
                        "description", "contact_info", "website", "social_media",
                        "accessibility", "parking", "payment_options",
                        "languages_spoken", "season", "accessible", "languages",
                        "dress_code"
                    ]
                },
                "realistic_curacao_packages": {
                    "path": "tourism_packages.xlsx",
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
                    # Read Excel file with all columns as strings initially
                    df = pd.read_excel(file_info["path"], dtype=str)
                    logger.info(f"Successfully read {file_info['path']}")

                    # Convert column names to lowercase
                    df.columns = df.columns.str.lower()

                    # Add missing columns with None values
                    for col in file_info["expected_columns"]:
                        if col.lower() not in df.columns:
                            df[col.lower()] = None
                            logger.warning(f"Added missing column {col} to {table_name}")

                    # Convert numeric columns to appropriate types
                    numeric_columns = {
                        'cost': 'float64',
                        'entry_fee': 'float64',
                        'price': 'float64',
                        'rating': 'float64',
                        'duration_hours': 'float64',
                        'duration_days': 'int64'
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

                except FileNotFoundError:
                    logger.error(f"Excel file not found: {file_info['path']}")
                    continue
                except pd.errors.EmptyDataError:
                    logger.error(f"Excel file is empty: {file_info['path']}")
                    continue
                except Exception as e:
                    logger.error(f"Error loading {table_name}: {str(e)}")
                    continue

            if not self.dataframes:
                raise Exception("No Excel files were successfully loaded")

        except Exception as e:
            logger.error(f"Error in Excel data loading: {str(e)}")
            raise

    def execute_query(self, query: str, params: Optional[List[Any]] = None) -> List[Dict]:
        """
        Execute SQL query with enhanced error handling

        Args:
            query (str): SQL query to execute
            params (List[Any], optional): Parameters to substitute in query

        Returns:
            List[Dict]: Query results as list of dictionaries
        """
        try:
            # Parameter substitution with type checking
            if params:
                for param in params:
                    if isinstance(param, (int, float)):
                        query = query.replace('?', str(param), 1)
                    else:
                        # Escape single quotes in string parameters
                        param_str = str(param).replace("'", "''")
                        query = query.replace('?', f"'{param_str}'", 1)

            # Log query for debugging
            logger.debug(f"Executing query: {query}")

            # Execute query
            result = sqldf(query, self.dataframes)

            # Convert to list of dictionaries with null handling
            records = []
            for _, row in result.iterrows():
                record = {}
                for column in row.index:
                    value = row[column]
                    if pd.isna(value):
                        record[column] = None
                    else:
                        record[column] = value
                records.append(record)

            # Log result summary
            logger.debug(f"Query returned {len(records)} records")
            return records

        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}\nQuery: {query}")
            return []

    def get_table_info(self, table_name: str) -> Dict:
        """
        Get information about a specific table

        Args:
            table_name (str): Name of the table

        Returns:
            Dict: Table information including columns and data types
        """
        try:
            if table_name not in self.dataframes:
                raise KeyError(f"Table {table_name} not found")

            df = self.dataframes[table_name]
            return {
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict(),
                "row_count": len(df),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "null_counts": df.isna().sum().to_dict()
            }
        except Exception as e:
            logger.error(f"Error getting table info for {table_name}: {str(e)}")
            return {}

    def get_column_stats(self, table_name: str, column_name: str) -> Dict:
        """
        Get statistical information about a specific column

        Args:
            table_name (str): Name of the table
            column_name (str): Name of the column

        Returns:
            Dict: Column statistics
        """
        try:
            if table_name not in self.dataframes:
                raise KeyError(f"Table {table_name} not found")

            df = self.dataframes[table_name]
            if column_name not in df.columns:
                raise KeyError(f"Column {column_name} not found in {table_name}")

            column = df[column_name]
            stats = {
                "null_count": column.isna().sum(),
                "unique_values": column.nunique(),
                "data_type": str(column.dtype)
            }

            # Add numeric stats if applicable
            if pd.api.types.is_numeric_dtype(column):
                stats.update({
                    "min": float(column.min()) if not pd.isna(column.min()) else None,
                    "max": float(column.max()) if not pd.isna(column.max()) else None,
                    "mean": float(column.mean()) if not pd.isna(column.mean()) else None,
                    "median": float(column.median()) if not pd.isna(column.median()) else None
                })

            # Add categorical stats if applicable
            if pd.api.types.is_string_dtype(column):
                value_counts = column.value_counts().head(10).to_dict()
                stats["top_values"] = value_counts

            return stats

        except Exception as e:
            logger.error(f"Error getting column stats for {table_name}.{column_name}: {str(e)}")
            return {}

    def validate_data(self) -> Dict[str, List[str]]:
        """
        Validate data quality across all tables

        Returns:
            Dict[str, List[str]]: Dictionary of validation issues by table
        """
        validation_issues = {}

        try:
            for table_name, df in self.dataframes.items():
                issues = []

                # Check for duplicate IDs
                id_column = next((col for col in df.columns if 'id_' in col), None)
                if id_column and df[id_column].duplicated().any():
                    issues.append(f"Duplicate IDs found in {id_column}")

                # Check for missing required values
                required_columns = ['name', 'type', 'location']
                for col in required_columns:
                    if col in df.columns and df[col].isna().any():
                        issues.append(f"Missing values in required column: {col}")

                # Check for invalid ratings
                if 'rating' in df.columns:
                    invalid_ratings = df[
                        (df['rating'].notna()) & 
                        ((df['rating'] < 0) | (df['rating'] > 5))
                    ]
                    if not invalid_ratings.empty:
                        issues.append("Invalid rating values found")

                # Check for invalid prices/costs
                price_columns = ['cost', 'price', 'entry_fee']
                for col in price_columns:
                    if col in df.columns:
                        invalid_prices = df[
                            (df[col].notna()) & 
                            (df[col] < 0)
                        ]
                        if not invalid_prices.empty:
                            issues.append(f"Negative values found in {col}")

                if issues:
                    validation_issues[table_name] = issues

            return validation_issues

        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            return {"error": [str(e)]}

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all loaded data

        Returns:
            Dict[str, Any]: Summary statistics for all tables
        """
        try:
            summary = {}
            for table_name, df in self.dataframes.items():
                summary[table_name] = {
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "memory_usage": df.memory_usage(deep=True).sum(),
                    "null_percentage": (df.isna().sum().sum() / (df.size)) * 100,
                    "columns": list(df.columns)
                }
            return summary
        except Exception as e:
            logger.error(f"Error generating data summary: {str(e)}")
            return {}