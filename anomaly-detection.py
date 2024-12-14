import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from srom.deep_learning.anomaly_detector import DNNAutoEncoder
from scipy.stats import chi2
import sqlalchemy
from sqlalchemy import create_engine, types, text
from sqlalchemy.exc import SQLAlchemyError
import logging
from logging.handlers import RotatingFileHandler
import copy
import json
import re
from sklearn.decomposition import PCA
from typing import List, Dict, Union, Tuple
from pathlib import Path
from tqdm import tqdm
from sdcclient import SdMonitorClient
from datetime import datetime, timedelta
import yaml
import time

# Suppress warnings
warnings.filterwarnings("ignore")

# Custom exceptions
class DatabaseError(Exception):
    """Custom exception for database-related errors."""
    pass

class DataProcessingError(Exception):
    """Custom exception for data processing errors."""
    pass

class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass

def load_config(config_path: str = "sysdig_postgres_config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        raise ConfigurationError(f"Error loading configuration: {str(e)}")

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

# Load configuration
CONFIG = load_config()
LOGGER = setup_logging(CONFIG['paths']['log_file'])

# Configuration
class Config:
    """Configuration class to store all constants and file paths."""
    BASE_DIR = Path(CONFIG['paths']['base_dir'])

    # Database configuration
    DB_CONFIG = {
        "user": CONFIG['database']['user'],
        "password": CONFIG['database']['password'],
        "host_url": CONFIG['database']['host_url'],
        "port": CONFIG['database']['port'],
        "db_name": CONFIG['database']['db_name']
    }

    # Sysdig API configuration
    SYSDIG_API_TOKEN = CONFIG['sysdig']['api_token']
    SYSDIG_SDC_URL = CONFIG['sysdig']['sdc_url']

    # Constants
    THRESHOLD = CONFIG['model']['threshold']
    SCHEMA_NAME = CONFIG['schema']['name']
    TABLE_NAME_UNIVARIATE = CONFIG['schema']['tables']['univariate']
    TABLE_NAME_MULTIVARIATE = CONFIG['schema']['tables']['multivariate']
    MULTIVARIATE_LOADINGS = CONFIG['schema']['tables']['multivariate_loadings']
    #MULTIVARIATE_TOP_VARIABLES = "anomaly_detection_multivariate_top_variables"
    SYSDIG_EVENTS_TABLE = CONFIG['schema']['tables']['sysdig_events']
    SYSDIG_NOTIFICATIONS = CONFIG['schema']['tables']['sysdig_notifications']
    UNIVARIATE_MEDIAN_VALUE = CONFIG['schema']['tables']['univariate_median_value']

def create_database_if_not_exists(config: dict) -> None:
    """Create the database if it doesn't exist."""
    try:
        # Connect to default postgres database first
        default_engine = create_engine(
            f"postgresql://{config['user']}:{config['password']}@{config['host_url']}:{config['port']}/postgres"
        )
        
        with default_engine.connect() as conn:
            # Disconnect all other connections to the target database
            conn.execute(text(f"""
                SELECT pg_terminate_backend(pid) 
                FROM pg_stat_activity 
                WHERE datname = '{config['db_name']}' AND pid != pg_backend_pid()
            """))
            conn.execute(text("commit"))
            
            # Check if database exists
            result = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{config['db_name']}'"))
            if not result.scalar():
                # Create database if it doesn't exist
                conn.execute(text("commit"))
                conn.execute(text(f"CREATE DATABASE {config['db_name']}"))
                LOGGER.info(f"Created database {config['db_name']}")
    except SQLAlchemyError as e:
        raise DatabaseError(f"Failed to create database: {str(e)}")
    finally:
        default_engine.dispose()

def connect_postgres_sqlalchemy(config: dict) -> sqlalchemy.engine.Engine:
    """Create a SQLAlchemy engine for PostgreSQL connection and ensure database and schema exist."""
    try:
        db_config = config['database']
        
        # First ensure database exists
        create_database_if_not_exists(db_config)
        
        # Connect to the target database
        engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host_url']}:{db_config['port']}/{db_config['db_name']}"
        )
        
        # Create schema if it doesn't exist
        with engine.connect() as conn:
            schema_name = config['schema']['name']
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))
            conn.execute(text("commit"))
            LOGGER.info(f"Ensured schema {schema_name} exists")
        
        return engine
    except SQLAlchemyError as e:
        raise DatabaseError(f"Failed to connect to database: {str(e)}")

def get_metric_columns(df: pd.DataFrame) -> List[str]:
    """Identify metric columns from DataFrame."""
    exclude_columns = ['time', 'anomaly_score', 'anomaly_label', 'sysdig_metric']
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    return [col for col in numeric_columns if col not in exclude_columns]

def load_data_from_api(sdclient: SdMonitorClient, metrics: List[Dict[str, Union[str, Dict[str, str]]]]) -> pd.DataFrame:
    """Load data from Sysdig API."""
    try:
        current_time = datetime.now() - timedelta(minutes=2)
        end = current_time
        start = end - timedelta(minutes=15)

        PAGE_SIZE = 20000
        paging = {"from": 0, "to": PAGE_SIZE}
        sampling = 60

        dfs_list = []

        for i in range(3):
            interval_start = int((end - (i + 1) * timedelta(minutes=5)).timestamp())
            interval_end = int((end - i * timedelta(minutes=5)).timestamp())

            for metric in metrics:
                ok, results = sdclient.get_data(
                    metrics=[metric],
                    start_ts=interval_start,
                    end_ts=interval_end,
                    sampling_s=sampling,
                    filter=None,
                    paging=paging,
                )
                if not ok:
                    LOGGER.error(f"Error fetching data for metric {metric['id']}: {results}")
                    continue

                m_df = pd.DataFrame.from_dict(results["data"])
                if not m_df.empty:
                    m_df[[x["id"] for x in [metric]]] = pd.DataFrame(
                        m_df["d"].values.tolist(), index=m_df.index
                    )
                    m_df.drop(columns=["d"], inplace=True)
                    m_df.rename(columns={"t": "time"}, inplace=True)
                    dfs_list.append(m_df)

        if dfs_list:
            metrics_df = pd.concat(dfs_list, ignore_index=True)
            LOGGER.info(f"Loaded {len(metrics_df)} rows from Sysdig API")
            return metrics_df
        else:
            raise DataProcessingError("No data fetched from Sysdig API")

    except Exception as e:
        raise DataProcessingError(f"Error loading data from Sysdig API: {str(e)}")

def chi_square_test(errors: np.ndarray) -> np.ndarray:
    """Perform chi-square test on errors."""
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    chi_scores = np.square((errors - mean_error) / std_error)
    p_values = 1 - chi2.cdf(chi_scores, df=1)
    return p_values

def normalize_anomaly_scores(df: pd.DataFrame, group_col: str = "anomaly_label") -> pd.DataFrame:
    """Normalize anomaly scores using inverted Min-Max normalization."""
    df = df.copy()
    df["anomaly_score"] = df.groupby(group_col)["anomaly_score"].transform(
        lambda x: ((x.max() - x) / (x.max() - x.min())) * 100
    )
    return df

def process_univariate_anomalies(metrics_df: pd.DataFrame, metric_columns: List[str]) -> pd.DataFrame:
    """Process univariate anomalies using DNNAutoEncoder for each metric separately."""
    LOGGER.info("Processing univariate anomalies...")

    model = DNNAutoEncoder()
    scaler = StandardScaler()

    # Data Preprocessing
    data = metrics_df[metric_columns].values
    # Fill NaNs with zero
    data = np.nan_to_num(data, nan=0.0)
    scaled_data = scaler.fit_transform(data)

    other_columns = [col for col in metrics_df.columns if col not in metric_columns]
    result_all = []

    # Process each metric individually
    for idx, col in enumerate(tqdm(metric_columns, desc="Processing metrics")):
        df = metrics_df.loc[:, other_columns].copy()
        df.loc[:, col] = metrics_df.loc[:, col]

        # Fit and predict on single metric
        model.fit(scaled_data[:, idx].reshape(-1, 1))
        reconstructed_data = model.predict(scaled_data[:, idx].reshape(-1, 1))

        # Calculate reconstruction errors
        reconstruction_errors = np.mean(np.square(
            scaled_data[:, idx].reshape(-1, 1) - reconstructed_data
        ), axis=1)

        # Apply chi-square test
        p_values = chi_square_test(reconstruction_errors)

        # Label anomalies
        threshold = 0.0001
        df["anomaly_score"] = p_values
        df["anomaly_label"] = np.where(p_values < threshold, 1, 0)
        df["sysdig_metric"] = col

        result_all.append(df)

    df_final = pd.concat(result_all, axis=0, ignore_index=True)

    # Restore NaNs after processing
    metrics_df[metric_columns] = metrics_df[metric_columns].mask(metrics_df[metric_columns].isna(), np.nan)

    return df_final

def process_multivariate_anomalies(
    metrics_df: pd.DataFrame,
    metric_columns: List[str],
    schema_name: str,
    local_engine: sqlalchemy.engine.Engine
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process multivariate anomalies using PCA."""
    LOGGER.info("Processing multivariate anomalies...")

    # Initialize results
    loadings_df = pd.DataFrame()

    # Prepare data
    filtered_df = metrics_df.loc[:, metric_columns]

    # Fill NaNs with zero for PCA processing
    filtered_df = filtered_df.fillna(0.0)

    # Normalize the data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(filtered_df)

    # Apply PCA
    pca = PCA()
    principal_components = pca.fit_transform(data_normalized)

    # Calculate variance ratios and determine components
    variance_ratios = pca.explained_variance_ratio_
    cumulative_variance_ratios = np.cumsum(variance_ratios)
    num_components = max(1, len(cumulative_variance_ratios[cumulative_variance_ratios <= 0.95]))

    # Create loadings dataframe with available components
    loadings_df = pd.DataFrame(
        pca.components_[:num_components].T,  # Only use the determined number of components
        columns=[f"pc_{i+1}" for i in range(num_components)],  # Use lowercase pc_n format
        index=filtered_df.columns,
    )
    
    # Add metric column and reset index once
    loadings_df = loadings_df.reset_index().rename(columns={'index': 'metric'})

    # Find the single most influential variable across all components
    # First, get absolute values of all loadings
    abs_loadings = loadings_df.set_index('metric').abs()
    
    # Find the maximum loading value and its corresponding metric and PC
    max_loading_value = abs_loadings.max().max()
    max_loading_pc = abs_loadings.max().idxmax()  # This gives us the PC
    max_loading_metric = abs_loadings[max_loading_pc].idxmax()  # This gives us the metric
    
    # Add the top variable information as the last row in loadings_df
    top_variable_row = pd.DataFrame([{
        'metric': f"{max_loading_metric}",
        **{col: np.nan for col in loadings_df.columns if col != 'metric'}
    }])
    loadings_df = pd.concat([loadings_df, top_variable_row], ignore_index=True)
    loadings_df['index'] = range(len(loadings_df))
    # Calculate Hotelling's T2 statistic
    t2_scores = np.zeros(len(principal_components))
    for i in range(num_components):
        t2_scores += (principal_components[:, i] ** 2) / variance_ratios[i]

    # Create results DataFrame
    results_df = metrics_df.copy()
    results_df['anomaly_score'] = t2_scores
    results_df['anomaly_label'] = np.where(
        t2_scores > chi2.ppf(0.999, num_components),
        1,
        0
    )

    return results_df, loadings_df

def calculate_median_values(metrics_df: pd.DataFrame, metric_columns: List[str]) -> pd.DataFrame:
    """Calculate median values for each metric."""
    median_values = metrics_df[metric_columns].median().reset_index()
    median_values.columns = ['metric', 'median_value']
    return median_values

def save_to_postgres(
    df: pd.DataFrame,
    table_name: str,
    schema_name: str,
    engine: sqlalchemy.engine.Engine,
    type_mapping: Dict
) -> None:
    """Save DataFrame to PostgreSQL."""
    try:
        with engine.begin() as connection:
            connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))

        df.to_sql(
            table_name,
            engine,
            schema=schema_name,
            if_exists="replace",
            index=False,
            dtype=type_mapping,
            chunksize=500000
        )

        LOGGER.info(f"Successfully saved data to {schema_name}.{table_name}")

    except SQLAlchemyError as e:
        raise DatabaseError(f"Failed to save data to {table_name}: {str(e)}")
    

def get_flattened_notifications(sdclient, from_ts=None, to_ts=None):
    """
    Fetch notifications from the given client, flatten nested dictionaries, 
    and extract relevant fields for further processing.

    Parameters:
        sdclient: The client object used to fetch notifications.
        from_ts (int): Start timestamp for the query.
        to_ts (int): End timestamp for the query.

    Returns:
        pd.DataFrame: A DataFrame containing flattened notification data.
    """
    if from_ts is None:
        from_ts = int(time.time() - 86400*10)  # Default to last 24 hours
    if to_ts is None:
        to_ts = int(time.time())

    # Fetch notifications
    res = sdclient.get_notifications(from_ts=from_ts, to_ts=to_ts)
    print("Fetched notifications:", res)  # Logging the response structure

    flat_data = []
    for entry in res[1]["notifications"]:
        flat_entry = {key: value for key, value in entry.items() if key != "entities"}
        entities = entry.get("entities", [])
        for entity in entities:
            flat_entry.update(entity)
            metric_values = entity.get("metricValues", [])
            if metric_values:
                flat_entry.update(metric_values[0])
            flat_data.append(flat_entry.copy())

    return pd.DataFrame(flat_data)

def extract_host_details(df, filter_column="filter"):
    """
    Extract host-related details (host_hostname, kube_pod_name, kube_cluster_name) 
    from the specified filter column in the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing notification data.
        filter_column (str): Column name containing the filter strings.

    Returns:
        pd.DataFrame: The DataFrame with additional columns for extracted host details.
    """
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
            kube_cluster_name = re.search(r'kube_cluster_name\s*=\s*"([^"]+)"', string).group(1)
        except AttributeError:
            kube_cluster_name = "Unknown"

        return host_hostname, kube_pod_name, kube_cluster_name

    if filter_column in df.columns:
        extracted_values = df[filter_column].apply(find_hosts)
        df["host_hostname"] = [t[0] for t in extracted_values]
        df["kube_pod_name"] = [t[1] for t in extracted_values]
        df["kube_cluster_name"] = [t[2] for t in extracted_values]
    else:
        print(f"Column '{filter_column}' not found in DataFrame")

    return df

def main():
    """Main function to run the anomaly detection pipeline."""
    try:
        # Initialize database connection
        LOGGER.info("Connecting to database...")
        local_engine = connect_postgres_sqlalchemy(CONFIG)

        # Initialize Sysdig API client
        LOGGER.info("Connecting to Sysdig API...")
        sdclient = SdMonitorClient(
            token=CONFIG['sysdig']['api_token'],
            sdc_url=CONFIG['sysdig']['sdc_url']
        )

        # Define metrics to fetch
        metrics = [
            {"id": "ibm_location"},
            {"id": "ibm_is_instance_average_cpu_usage_percentage", "aggregations": {"time": "avg"}},
            {"id": "ibm_is_instance_cpu_usage_percentage", "aggregations": {"time": "avg"}},
            {"id": "ibm_is_instance_memory_free_kib", "aggregations": {"time": "avg"}},
            {"id": "ibm_is_instance_memory_used_kib", "aggregations": {"time": "avg"}},
            {"id": "ibm_is_instance_network_in_bytes", "aggregations": {"time": "sum"}},
            {"id": "ibm_is_instance_network_in_dropped_packets", "aggregations": {"time": "sum"}},
            {"id": "ibm_is_instance_network_in_errors", "aggregations": {"time": "sum"}},
            {"id": "ibm_is_instance_network_in_packets", "aggregations": {"time": "sum"}},
            {"id": "ibm_is_instance_network_out_bytes", "aggregations": {"time": "sum"}},
            {"id": "ibm_is_instance_network_out_dropped_packets", "aggregations": {"time": "sum"}},
            {"id": "ibm_is_instance_network_out_errors", "aggregations": {"time": "sum"}},
            {"id": "ibm_is_instance_network_out_packets", "aggregations": {"time": "sum"}},
            {"id": "ibm_is_instance_volume_read_bytes", "aggregations": {"time": "sum"}},
            {"id": "ibm_is_instance_volume_read_requests", "aggregations": {"time": "sum"}},
            {"id": "ibm_is_instance_volume_write_bytes", "aggregations": {"time": "sum"}},
            {"id": "ibm_is_instance_volume_write_requests", "aggregations": {"time": "sum"}}
        ]

        # Load data
        LOGGER.info("Loading data from Sysdig API...")
        metrics_df = load_data_from_api(sdclient, metrics)
                # Create a table if it doesn't exist
        create_table_query = text(f'''
        CREATE SCHEMA IF NOT EXISTS {Config.SCHEMA_NAME};
        DROP TABLE IF EXISTS {Config.SCHEMA_NAME}.sysdig_metrics;
        CREATE TABLE IF NOT EXISTS {Config.SCHEMA_NAME}.sysdig_metrics (
            id VARCHAR(255),
            version VARCHAR(255),
            customerid VARCHAR(255),
            teamid VARCHAR(255),
            timespan INTEGER,
            timestamp BIGINT,
            alert VARCHAR(255),
            alertname VARCHAR(255),
            severity INTEGER,
            alertthreshold FLOAT,
            filter TEXT,
            state VARCHAR(50),
            resolved BOOLEAN,
            nodata BOOLEAN,
            rateofchange FLOAT,
            severitylabel VARCHAR(50),
            notificationtype VARCHAR(100),
            condition TEXT,
            segment TEXT,
            metricvalues TEXT,
            metric VARCHAR(255),
            aggregation VARCHAR(100),
            groupaggregation VARCHAR(100),
            value FLOAT,
            alertfilter TEXT,
            host_hostname VARCHAR(255),
            kube_pod_name VARCHAR(255),
            kube_cluster_name VARCHAR(255)
        );
        ''')

        # Execute SQL commands
        with local_engine.connect() as conn:
            conn.execute(create_table_query)
            conn.commit()
            
            # Get notifications data
            df_notif = get_flattened_notifications(sdclient)
            
            if df_notif.empty:
                # Create an empty DataFrame with the correct columns
                columns = ['id', 'version', 'customerid', 'teamid', 'timespan', 'timestamp',
                          'alert', 'alertname', 'severity', 'alertthreshold', 'filter', 'state',
                          'resolved', 'nodata', 'rateofchange', 'severitylabel', 'notificationtype',
                          'condition', 'segment', 'metricvalues', 'metric', 'aggregation',
                          'groupaggregation', 'value', 'alertfilter', 'host_hostname',
                          'kube_pod_name', 'kube_cluster_name']
                df_empty = pd.DataFrame(columns=columns)
                # Insert a single row with all NaN values
                df_empty.loc[0] = pd.NA
                df_empty.to_sql('sysdig_metrics', conn, schema=Config.SCHEMA_NAME, 
                              if_exists='append', index=False)
                LOGGER.info("No data available. Inserted row with NULL values.")
            else:
                LOGGER.info(f"Found {len(df_notif)} notifications to process")
                df_notif = extract_host_details(df_notif)
                # Convert column names to lowercase before saving
                df_notif.columns = df_notif.columns.str.lower()
                df_notif.to_sql('sysdig_metrics', conn, schema=Config.SCHEMA_NAME, 
                              if_exists='append', index=False)

        # Get metric columns
        metric_columns = get_metric_columns(metrics_df)

        # Process univariate anomalies
        univariate_results = process_univariate_anomalies(metrics_df, metric_columns)
        univariate_results = normalize_anomaly_scores(univariate_results)

        # Process multivariate anomalies
        multivariate_results, loadings_df = process_multivariate_anomalies(
            metrics_df,
            metric_columns,
            Config.SCHEMA_NAME,
            local_engine
        )
        multivariate_results = normalize_anomaly_scores(multivariate_results)

        # Calculate median values
        median_values_df = calculate_median_values(metrics_df, metric_columns)

        # Prepare type mapping for database
        type_mapping = {
            col: types.VARCHAR(256) if univariate_results[col].dtype == "object" else types.FLOAT()
            for col in univariate_results.columns
        }

        # Save results to database
        LOGGER.info("Saving results to database...")
        save_to_postgres(
            univariate_results,
            Config.TABLE_NAME_UNIVARIATE,
            Config.SCHEMA_NAME,
            local_engine,
            type_mapping
        )

        save_to_postgres(
            multivariate_results,
            Config.TABLE_NAME_MULTIVARIATE,
            Config.SCHEMA_NAME,
            local_engine,
            type_mapping
        )

        # Create dynamic type mapping for PCs
        pc_type_mapping = {
            'metric': types.VARCHAR(256)
        }
        # Add dynamic PC columns based on actual components
        for i in range(1, len(loadings_df.columns)):
            if f'pc_{i}' in loadings_df.columns.str.lower():
                pc_type_mapping[f'pc_{i}'] = types.FLOAT()

        save_to_postgres(
            loadings_df,
            Config.MULTIVARIATE_LOADINGS,
            Config.SCHEMA_NAME,
            local_engine,
            pc_type_mapping
        )

        # Save median values to database
        save_to_postgres(
            median_values_df,
            Config.UNIVARIATE_MEDIAN_VALUE,
            Config.SCHEMA_NAME,
            local_engine,
            {
                'metric': types.VARCHAR(256),
                'median_value': types.FLOAT()
            }
        )

        LOGGER.info("Anomaly detection pipeline completed successfully")

    except Exception as e:
        LOGGER.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()