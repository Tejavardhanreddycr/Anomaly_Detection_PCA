import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from scipy.stats import chi2
import duckdb
from duckdb import DuckDBPyConnection
import logging
from logging.handlers import RotatingFileHandler
import copy
import json
import re
from typing import List, Dict, Union, Tuple
from pathlib import Path
import argparse
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

# Suppress warnings
warnings.filterwarnings("ignore")

# Custom exceptions
class DatabaseError(Exception):
    """Custom exception for database-related errors."""
    pass

class DataProcessingError(Exception):
    """Custom exception for data processing errors."""
    pass

# Set up logging
def setup_logging(log_file: str = "anomaly_detection.log") -> logging.Logger:
    """Configure logging with both file and console handlers."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Rotating file handler
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10485760, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

LOGGER = setup_logging()

# Configuration
class Config:
    """Configuration class to store all constants and file paths."""
    BASE_DIR = Path("/")
    
    # File paths will be set dynamically
    UNIVARIATE_FILE = None
    MULTIVARIATE_FILE = None
    SYSDIG_EVENTS = None
    SYSDIG_NOTIFICATION = None
    
    # Database configuration
    DB_CONFIG = {
        "database": str(Path(__file__).parent / "processed-data/anomaly_detection.db")
    }
    
    # Constants
    PROCESSED_DATA_DIR = Path("processed-data")
    THRESHOLD = 50
    TABLE_NAME_UNIVARIATE = "anomaly_detection_score_sdn_2"
    TABLE_NAME_MULTIVARIATE = "anomaly_detection_score_multivariate_sdn_2"
    MULTIVARIATE_LOADINGS = "anomaly_detection_multivariate_loadings_sdn_1"
    SYSDIG_EVENTS_TABLE = "sysdig_events"
    SYSDIG_NOTIFICATIONS = "sysdig_notifications"
    UNIVARIATE_MEDIAN_VALUE = "anomaly_detection_univarient_sysdig_metrics_median_values"

    @classmethod
    def initialize_paths(cls, args):
        """Initialize file paths from command line arguments"""
        cls.UNIVARIATE_FILE = Path(args.univariate_file)
        cls.MULTIVARIATE_FILE = Path(args.multivariate_file)
        cls.SYSDIG_EVENTS = Path(args.sysdig_events)
        cls.SYSDIG_NOTIFICATION = Path(args.sysdig_notifications)
        # Create processed data directory if it doesn't exist
        cls.PROCESSED_DATA_DIR.mkdir(exist_ok=True)

#====================================================================================================================================
# Database connection
def connect_duckdb(**kwargs) -> DuckDBPyConnection:
    """
    Create a DuckDB connection.
    
    Args:
        kwargs: Keyword arguments for DuckDB connection
        
    Returns:
        DuckDBPyConnection object
        
    Raises:
        DatabaseError: If connection cannot be established
    """
    try:
        conn = duckdb.connect(**kwargs)
        LOGGER.info("Successfully connected to database")
        return conn
    except Exception as e:
        LOGGER.error(f"Database connection failed: {str(e)}")
        raise DatabaseError(f"Failed to connect to database: {str(e)}")

#====================================================================================================================================
# Define constants
def get_metric_columns(df: pd.DataFrame) -> List[str]:
    """
    Dynamically identify metric columns from DataFrame.
    Selects numeric columns while excluding time and anomaly-related columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of identified metric column names
    """
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Define patterns for columns to exclude
    exclude_patterns = [
        'time', 'timestamp', '_at$', 'date',  # Time-related
        'anomaly', 'score', 'label',          # Anomaly-related
        'id$', '_id$', 'index'                # ID/Index columns
    ]
    
    # Filter columns
    metric_columns = [
        col for col in numeric_cols 
        if not any(pattern in col.lower() for pattern in exclude_patterns)
    ]
    
    LOGGER.info(f"Identified {len(metric_columns)} metric columns")
    LOGGER.debug(f"Metric columns: {', '.join(metric_columns)}")
    
    return metric_columns

#====================================================================================================================================
# Load data from a file, ignoring any existing anomaly scores
def load_data(file_path: Union[str, Path], file_type: str = "csv") -> pd.DataFrame:
    """
    Load data from a file, ignoring any existing anomaly scores.
    Only loads the metric data and essential columns.
    
    Args:
        file_path: Path to the input file
        file_type: Type of file ('csv' or 'excel')
        
    Returns:
        DataFrame containing only the metric data and essential columns
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Load the data
        if file_type.lower() == "csv":
            df = pd.read_csv(file_path)
        elif file_type.lower() == "excel":
            df = pd.read_excel(file_path, engine="openpyxl")
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
        if df.empty:
            raise DataProcessingError("Loaded DataFrame is empty")
        
        # Filter out anomaly-related columns
        columns_to_keep = [
            col for col in df.columns 
            if not any(term in col.lower() for term in ['anomaly', 'score', 'label'])
        ]
        
        # Keep only metric and essential columns
        df = df[columns_to_keep]
        
        LOGGER.info(f"Loaded {len(df)} rows with {len(columns_to_keep)} columns from {file_path}")
        return df
        
    except Exception as e:
        raise DataProcessingError(f"Error loading data from {file_path}: {str(e)}")

#====================================================================================================================================
# Preprocess data by filling NaN values and identifying non-metric columns
def preprocess_data(data: pd.DataFrame, metric_columns: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Preprocess data by filling NaN values and identifying non-metric columns.
    Works with raw metric data only.
    
    Args:
        data: Input DataFrame with metric data
        metric_columns: List of metric column names
        
    Returns:
        Tuple containing processed DataFrame and list of non-metric columns
    """
    try:
        # Create a copy to avoid modifying original data
        processed_data = data.copy()
        
        # Get non-metric columns
        other_columns = [
            col for col in processed_data.columns 
            if col not in metric_columns
        ]
        
        # We don't fill NaN values anymore, letting them remain as NULL
        return processed_data, other_columns
        
    except Exception as e:
        raise DataProcessingError(f"Error preprocessing data: {str(e)}")

#====================================================================================================================================
class AnomalyDetector:
    """Class to handle anomaly detection operations."""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            random_state=random_state,
            contamination=contamination
        )
        self.logger = logging.getLogger(__name__)

    def _handle_missing_values(self, data: pd.Series) -> pd.Series:
        """Handle missing values in a data series."""
        if data.isna().any():
            self.logger.warning(f"Found {data.isna().sum()} missing values")
            return data.fillna(data.mean())
        return data

    def _detect_single_metric(self, data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies for a single metric using raw data.
        
        Args:
            data: Raw metric data series
            
        Returns:
            Tuple of (anomaly_scores, anomaly_labels)
        """
        # Scale the data
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        
        # Detect anomalies
        self.model.fit(scaled_data)
        predictions = self.model.predict(scaled_data).reshape(-1, 1)
        
        # Calculate scores
        reconstruction_errors = np.square(scaled_data - predictions)
        p_values = chi_square_test(reconstruction_errors)
        
        # Generate labels
        labels = np.where(p_values < 0.0001, 1, 0)
        
        return p_values.flatten(), labels.flatten()

    def process_metrics(self, data: pd.DataFrame, metric_columns: List[str]) -> pd.DataFrame:
        """Process multiple metrics for anomaly detection using raw data."""
        results = []
        
        for col in tqdm(metric_columns, desc="Processing metrics"):
            self.logger.debug(f"Processing metric: {col}")
            try:
                # Work with raw metric data
                df = data.copy()
                df[col] = self._handle_missing_values(df[col])
                
                # Calculate new anomaly scores
                scores, labels = self._detect_single_metric(df[col])
                
                # Add results
                df["anomaly_score"] = scores
                df["anomaly_label"] = labels
                df["sysdig_metric"] = col
                
                results.append(df)
                self.logger.info(f"Successfully processed {col}: found {labels.sum()} anomalies")
                
            except Exception as e:
                self.logger.error(f"Error processing metric {col}: {str(e)}")
                continue
                
        if not results:
            raise DataProcessingError("No metrics were successfully processed")
            
        return pd.concat(results, axis=0, ignore_index=True)

#====================================================================================================================================
#chi sq test
def chi_square_test(errors):
    """Perform chi-square test on errors."""
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    chi_scores = np.square((errors - mean_error) / std_error)
    return 1 - chi2.cdf(chi_scores, df=1)

#====================================================================================================================================
#min max scaler
def normalize_anomaly_scores(df, group_col="sysdig_metric"):
    """Normalize anomaly scores using inverted Min-Max normalization."""
    def inverted_min_max(group):
        min_val = group.min()
        max_val = group.max()
        return (max_val - group) / (max_val - min_val) * 100 if max_val != min_val else group * 0

    df["anomaly_score"] = df.groupby(group_col)["anomaly_score"].transform(inverted_min_max)
    df["anomaly_label"] = (df["anomaly_score"] >= Config.THRESHOLD).astype(int)
    return df

#====================================================================================================================================
#univariate analysis
def process_univarient_median(local_conn: DuckDBPyConnection, df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """
    Process univariate median values and save to database.
    
    Args:
        local_conn: DuckDB connection
        df: DataFrame containing the data
        metrics: List of metric columns to process
        
    Returns:
        DataFrame containing median values
    """
    try:
        LOGGER.info("Processing univariate median values...")
        LOGGER.info(f"Identified {len(metrics)} metric columns")
        
        result_df = pd.DataFrame()
        
        with tqdm(total=len(metrics), desc="Processing metrics") as pbar:
            for metric in metrics:
                median_value = df[metric].median()
                new_row = pd.DataFrame({
                    'sysdig_metric': [metric],
                    'median_value': [median_value]
                })
                result_df = pd.concat([result_df, new_row], ignore_index=True)
                pbar.update(1)
        
        # Push data to DuckDB
        try:
            local_conn.register('result_df_temp', result_df)
            local_conn.execute(f"CREATE TABLE IF NOT EXISTS {Config.UNIVARIATE_MEDIAN_VALUE} AS SELECT * FROM result_df_temp")
            local_conn.commit()
            
            LOGGER.info(
                f"Successfully processed and saved {result_df.shape[0]} median values to "
                f"{Config.UNIVARIATE_MEDIAN_VALUE}"
            )
            
            return result_df
            
        except Exception as e:
            LOGGER.error(f"Error saving median values: {str(e)}")
            raise DatabaseError(f"Failed to save median values: {str(e)}")
            
    except Exception as e:
        LOGGER.error(f"Error processing median values: {str(e)}")
        raise DataProcessingError(f"Failed to process median values: {str(e)}")

#====================================================================================================================================
#Push to DB
def save_to_duckdb(df, table_name, conn):
    """Save DataFrame to DuckDB."""
    try:
        conn.register('df_temp', df)
        conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df_temp")
        conn.commit()
        LOGGER.info(f"Successfully pushed {df.shape[0]} records to {table_name}")
    except Exception as e:
        LOGGER.error(f"Error saving to DuckDB: {e}")

#====================================================================================================================================
# Load datasets
def process_notifications(sysdig_notification: pd.DataFrame) -> pd.DataFrame:
    """Process Sysdig notifications data."""
    flat_data = []
    
    for _, row in sysdig_notification.iterrows():
        flat_entry = row.to_dict()
        
        # Parse 'metricValues' if it contains JSON
        metric_values = row.get("metricValues")
        if isinstance(metric_values, str):
            try:
                metric_values = json.loads(metric_values)
                if isinstance(metric_values, list) and metric_values:
                    flat_entry.update(metric_values[0])
            except json.JSONDecodeError:
                LOGGER.warning(f"Skipping invalid metricValues entry: {metric_values}")
        flat_data.append(flat_entry)
    
    # Convert the flattened data to a DataFrame
    df_notif = pd.DataFrame(flat_data)
    
    try:
        x = df_notif["filter"].apply(find_hosts).values
        df_notif.loc[:, "host_hostname"] = [t[0] for t in x]
        df_notif.loc[:, "kube_pod_name"] = [t[1] for t in x]
        df_notif.loc[:, "kube_cluster_name"] = [t[2] for t in x]
    except Exception as e:
        LOGGER.warning(f'No active notifications: {str(e)}')
        df_notif = pd.DataFrame(columns=['id', 'timestamp'], index=[0])
    
    return df_notif

def perform_pca_analysis(
    metrics_table: pa.Table,
    metric_columns: List[str],
    local_engine: duckdb.DuckDBPyConnection
) -> None:
    """
    Apply PCA analysis on the metrics data.
    Args:
        metrics_table: PyArrow Table containing metric data
        metric_columns: List of metric columns to analyze
        local_engine: DuckDB connection for database operations
    """
    try:
        LOGGER.info("Starting PCA analysis...")
        # Convert to pandas for PCA processing
        metrics_df = metrics_table.to_pandas()
        # Initialize results DataFrame to store all PCA results
        all_loadings = []
        # Process data in chunks with progress bar
        chunk_size = 1000
        total_chunks = metrics_table.num_rows // chunk_size + (1 if metrics_table.num_rows % chunk_size != 0 else 0)
        for i in tqdm(range(0, metrics_table.num_rows, chunk_size), desc="Processing PCA chunks", total=total_chunks):
            chunk = metrics_df.iloc[i:i+chunk_size]
            # Create processed copy for PCA while keeping original data
            original_chunk = chunk.copy()
            processed_chunk = chunk.copy()
            # Fill NaN values in processed chunk
            for col in metric_columns:
                if processed_chunk[col].isna().any():
                    processed_chunk[col] = processed_chunk[col].fillna(processed_chunk[col].mean())
            # First detect anomalies if not already done
            if 'anomaly_label' not in chunk.columns:
                detector = AnomalyDetector()
                chunk = detector.process_metrics(original_chunk, metric_columns)
                chunk = normalize_anomaly_scores(chunk)
            # Check if there are any anomalies
            if 1 not in chunk.anomaly_label.unique():
                LOGGER.info(f"No anomalies detected in chunk {i//chunk_size + 1}/{total_chunks}, skipping PCA analysis")
                continue
            LOGGER.info(f"Found anomalies in chunk {i//chunk_size + 1}/{total_chunks}, proceeding with PCA")
            # Extract metric data for PCA from processed chunk
            filtered_df = processed_chunk[metric_columns]
            # Normalize the data
            data_normalized = (filtered_df - filtered_df.mean()) / filtered_df.std()
            data_normalized.fillna(0, inplace=True)
            # Apply PCA
            pca = PCA()
            principal_components = pca.fit_transform(data_normalized)
            # Calculate variance ratios
            variance_ratios = pca.explained_variance_ratio_
            cumulative_variance_ratios = np.cumsum(variance_ratios)
            # Determine number of components
            num_components = len(cumulative_variance_ratios[cumulative_variance_ratios <= 0.95])
            # Get loadings
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(pca.n_components_)],
                index=filtered_df.columns
            )
            loadings.reset_index(inplace=True)
            loadings.rename(columns={'index': 'metric'}, inplace=True)
            print(f"\nLoadings for chunk {i//chunk_size + 1}:")
            print(loadings)
            all_loadings.append(loadings)
            LOGGER.info(f"Added loadings for chunk {i//chunk_size + 1}/{total_chunks}. Current loadings count: {len(all_loadings)}")
            # Examine the scores of the principal components for the anomalous instances to determine which variables are contributing the most to the anomalies
            anomalies = list(
            metrics_df[metrics_df.anomaly_label == 1.0].index
)           # example anomalous instances
            anomaly_scores = pd.DataFrame(principal_components, index=filtered_df.index).iloc[
            anomalies
            ]
            # Log PCA results
            LOGGER.info(f"Number of components explaining 95% variance: {num_components}")
            # Log top contributing variables
            top_variables = set()
            for i in range(num_components):
                top_loading_vars = loadings.nlargest(1, f'PC{i+1}')['metric'].values
                LOGGER.info(f"PC{i+1} top loading variables: {', '.join(top_loading_vars)}")
                top_variables.update(top_loading_vars)
            LOGGER.info(f"Top contributing variables: {', '.join(top_variables)}")
        # Combine all loadings
        if all_loadings:
            LOGGER.info(f"Combining {len(all_loadings)} loading matrices")
            final_loadings = pd.concat(all_loadings, ignore_index=True)
            print("\nFinal Combined Loadings:")
            print(final_loadings)
            # Save loadings to database
            try:
                local_engine.execute(f"DROP TABLE IF EXISTS {Config.MULTIVARIATE_LOADINGS}")
                final_loadings.to_sql(
                    Config.MULTIVARIATE_LOADINGS,
                    local_engine,
                    if_exists="replace",
                    index=False
                )
                LOGGER.info(f"Successfully saved PCA loadings to {Config.MULTIVARIATE_LOADINGS}")
                
                # Also save as parquet file
                save_as_parquet(final_loadings, Config.MULTIVARIATE_LOADINGS)
                LOGGER.info(f"Successfully saved PCA loadings as parquet file")
            except Exception as e:
                LOGGER.error(f"Error saving PCA loadings: {str(e)}")
                raise DatabaseError(f"Failed to save PCA loadings: {str(e)}")
        else:
            LOGGER.warning("No PCA results to save")
    except Exception as e:
        LOGGER.error(f"Error in PCA analysis: {str(e)}")
        raise

#====================================================================================================================================
#find Host
def find_hosts(string):
    try:
        host_hostname = re.search(r'host_hostname\s*=\s*"([^"]+)"', string).group(1)
    except AttributeError:
        host_hostname = "Unknown"

    try:
        kube_pod_name = re.search(r'kube_pod_name\s*=\s*"([^"]+)"', string).group(1)
    except AttributeError:
        kube_pod_name = "Unknown"

    try:
        kube_cluster_name = re.search(
            r'kube_cluster_name\s*=\s*"([^"]+)"', string
        ).group(1)
    except AttributeError:
        kube_cluster_name = "Unknown"

    return host_hostname, kube_pod_name, kube_cluster_name

def parse_arguments():
    parser = argparse.ArgumentParser(description='Anomaly Detection Script')
    parser.add_argument('--univariate-file', type=str,
                      help='Path to univariate CSV file',
                      required=True)
    parser.add_argument('--multivariate-file', type=str,
                      help='Path to multivariate CSV file',
                      required=True)
    parser.add_argument('--sysdig-events', type=str,
                      help='Path to sysdig events CSV file',
                      required=True)
    parser.add_argument('--sysdig-notifications', type=str,
                      help='Path to sysdig notifications CSV file',
                      required=True)
    return parser.parse_args()

def save_as_parquet(df: pd.DataFrame, table_name: str) -> None:
    """Save DataFrame as a parquet file in the processed-data directory using pyarrow engine."""
    try:
        output_path = Config.PROCESSED_DATA_DIR / f"{table_name}.parquet"
        # Convert pandas DataFrame to PyArrow Table
        table = pa.Table.from_pandas(df)
        # Write using pyarrow.parquet
        pq.write_table(table, str(output_path))
        LOGGER.info(f"Successfully saved {df.shape[0]} records to {output_path}")
    except Exception as e:
        LOGGER.error(f"Error saving parquet file: {e}")
        raise

def main():
    """Main function to run the anomaly detection pipeline."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Initialize Config paths with parsed arguments
        Config.initialize_paths(args)
        
        # Validate that all required files exist
        required_files = [
            (Config.UNIVARIATE_FILE, "Univariate"),
            (Config.MULTIVARIATE_FILE, "Multivariate"),
            (Config.SYSDIG_EVENTS, "Sysdig events"),
            (Config.SYSDIG_NOTIFICATION, "Sysdig notifications")
        ]
        
        LOGGER.info("Validating input files...")
        for file_path, file_type in tqdm(required_files, desc="Checking files"):
            if not file_path.exists():
                raise FileNotFoundError(f"{file_type} file not found: {file_path}")
        
        # Initialize database connection
        LOGGER.info("Connecting to database...")
        local_conn = connect_duckdb(**Config.DB_CONFIG)
        
        # Load datasets with progress tracking
        LOGGER.info("Loading datasets...")
        with tqdm(total=4, desc="Loading datasets") as pbar:
            # Load univariate data
            univariate_df = pd.read_csv(Config.UNIVARIATE_FILE)
            pbar.update(1)
            
            # Load multivariate data
            multivariate_df = pd.read_csv(Config.MULTIVARIATE_FILE)
            pbar.update(1)
            
            # Load Sysdig events
            sysdig_events = pd.read_csv(Config.SYSDIG_EVENTS)
            pbar.update(1)
            
            # Load Sysdig notifications
            sysdig_notifications = pd.read_csv(Config.SYSDIG_NOTIFICATION)
            pbar.update(1)
        
        # Process notifications
        LOGGER.info("Processing notifications...")
        df_notif = process_notifications(sysdig_notifications)
        
        # Dynamic metric selection
        LOGGER.info("Identifying metric columns...")
        univariate_metrics = get_metric_columns(univariate_df)
        multivariate_metrics = get_metric_columns(multivariate_df)
        
        # PCA Analysis
        LOGGER.info("Performing PCA analysis...")
        multivariate_table = pa.Table.from_pandas(multivariate_df)
        perform_pca_analysis(multivariate_table, multivariate_metrics, local_conn)
        
        # Preprocess data
        LOGGER.info("Preprocessing data...")
        univariate_df, univariate_other_cols = preprocess_data(univariate_df, univariate_metrics)
        multivariate_df, multivariate_other_cols = preprocess_data(multivariate_df, multivariate_metrics)
        
        # Detect anomalies
        LOGGER.info("Detecting anomalies...")
        anomaly_detector = AnomalyDetector()
        univariate_results = anomaly_detector.process_metrics(univariate_df, univariate_metrics)
        multivariate_results = anomaly_detector.process_metrics(multivariate_df, multivariate_metrics)
        
        # Normalize scores
        LOGGER.info("Normalizing anomaly scores...")
        univariate_results = normalize_anomaly_scores(univariate_results)
        multivariate_results = normalize_anomaly_scores(multivariate_results)
        
        # Save results to database
        LOGGER.info("Saving results to database...")
        save_to_duckdb(univariate_results, Config.TABLE_NAME_UNIVARIATE, local_conn)
        
        # Process univariate median values
        LOGGER.info("Processing univariate median values...")
        median_values_df = process_univarient_median(local_conn, univariate_df, univariate_metrics)
        
        save_to_duckdb(multivariate_results, Config.TABLE_NAME_MULTIVARIATE, local_conn)
        save_to_duckdb(sysdig_events, Config.SYSDIG_EVENTS_TABLE, local_conn)
        save_to_duckdb(sysdig_notifications, Config.SYSDIG_NOTIFICATIONS, local_conn)
        
        # Save results as parquet files
        LOGGER.info("Saving results as parquet files...")
        save_as_parquet(univariate_results, Config.TABLE_NAME_UNIVARIATE)
        save_as_parquet(multivariate_results, Config.TABLE_NAME_MULTIVARIATE)
        save_as_parquet(sysdig_events, Config.SYSDIG_EVENTS_TABLE)
        save_as_parquet(sysdig_notifications, Config.SYSDIG_NOTIFICATIONS)
        
        # Save derived analysis tables as parquet
        save_as_parquet(median_values_df, Config.UNIVARIATE_MEDIAN_VALUE)
        
        LOGGER.info("Anomaly detection pipeline completed successfully")
        
    except Exception as e:
        LOGGER.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
