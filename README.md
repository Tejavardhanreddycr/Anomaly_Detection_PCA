## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Data Flow](#data-flow)
4. [Core Components](#core-components)
5. [Data Science Methodology](#data-science-methodology)
6. [Configuration](#configuration)
7. [Usage Guide](#usage-guide)

## Overview

The Anomaly Detection System is a robust solution for detecting anomalies in system metrics using both univariate and multivariate analysis approaches. It processes metrics data in real-time, performs anomaly detection using advanced machine learning techniques, and stores results for further analysis.

## System Architecture

### Components

1. **Data Input Layer**
   - Real-time metrics ingestion
   - Configurable data sources
   - Automated data validation

2. **Processing Layer**
   - Univariate Analysis Module
   - Multivariate Analysis Module (PCA-based)
   - DNNAutoEncoder for anomaly detection
   - Real-time processing pipeline

3. **Storage Layer**
   - Configurable database backend
   - Efficient data storage and retrieval
   - Structured schema design

### Design Patterns

- Configuration Management using YAML
- Comprehensive Error Handling
- Rotating Log System
- Connection Pool Management
- Custom Exception Classes

## Data Flow

1. **Input Processing**
```
Real-time Metrics → Data Loading → Preprocessing → Analysis → Results Storage
```

2. **Analysis Pipeline**
```
Metrics → Standardization → PCA/Univariate Analysis → Anomaly Detection → Score Normalization
```

3. **Output Generation**
```
Results → Database Storage → Loadings/Scores Tables
```

## Core Components

1. **Anomaly Detection Engine**
   - DNNAutoEncoder for deep learning-based detection
   - PCA for multivariate analysis
   - Chi-square test for statistical validation
   - Adaptive thresholding

2. **Data Processing**
   - Automated metric column identification
   - Real-time data normalization
   - Robust error handling
   - Configurable sampling intervals

3. **Monitoring and Logging**
   - Rotating file handlers
   - Comprehensive logging system
   - Performance monitoring
   - Error tracking

## Data Science Methodology

1. **Feature Processing**
   - Automated metric selection
   - Standardization using StandardScaler
   - PCA for dimensionality reduction
   - Median value calculations for baseline

2. **Anomaly Detection**
   - Univariate analysis per metric
   - Multivariate analysis using PCA
   - Score normalization
   - Statistical validation

## Configuration

The system uses a YAML-based configuration file (`sysdig_postgres_config.yaml`) for easy setup and maintenance. Key configuration areas include:

- Logging settings
- Model parameters
- Schema configuration
- Processing intervals
- Storage settings

## Usage Guide

1. **Setup**
   - Clone the repository
   - Install required dependencies
   - Configure the YAML file with your settings

2. **Running Anomaly Detection using the Sysdig API**
   ```bash
   python3 anomaly-detection.py
   ```

3. **Running Cloud Object Storage Upload**
   ```bash
   python anomaly_cos.py \
   --univariate-file <path_to_univariate_file> \
   --multivariate-file <path_to_multivariate_file> \
   --sysdig-events <path_to_events_file> \
   --sysdig-notifications <path_to_notifications_file>
   ```

   Example:
   ```bash
   python anomaly_cos.py \
   --univariate-file anomaly_detection_score_sdn_2_202411131654.csv \
   --multivariate-file anomaly_detection_score_multivariate_sdn_2_202411192133.csv \
   --sysdig-events sysdig_events.csv \
   --sysdig-notifications sysdig_notifications.csv
   ```

   Required Arguments for COS Upload:
   - `--univariate-file`: Path to univariate analysis file
   - `--multivariate-file`: Path to multivariate analysis file
   - `--sysdig-events`: Path to sysdig events file
   - `--sysdig-notifications`: Path to sysdig notifications file

4. **Configuration**
   - Update `sysdig_postgres_config.yaml` with your settings
   - Ensure all required paths are correctly set
   - Configure logging as needed
   - For COS upload, ensure the configuration file contains the necessary storage credentials

5. **Monitoring**
   - Check `anomaly_detection.log` for system status
   - Monitor the output database for results
   - Review anomaly scores and notifications
   - For COS uploads, verify the files in your cloud storage bucket