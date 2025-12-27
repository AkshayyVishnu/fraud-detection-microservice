"""
Data Processor for Fraud Network Graph
Loads creditcard.csv and builds transaction network relationships
OPTIMIZED for fast loading
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Path to dataset
DATA_PATH = Path(__file__).parent / 'data' / 'dataset' / 'creditcard.csv'

# Cache for loaded data
_df_cache = None
_network_cache = None


def load_dataset(sample_size=500):
    """Load the credit card dataset with aggressive sampling for fast performance"""
    global _df_cache
    
    if _df_cache is not None:
        return _df_cache
    
    if not DATA_PATH.exists():
        print(f"Dataset not found at {DATA_PATH}")
        return None
    
    print("Loading dataset (optimized)...")
    
    # Read only needed columns to reduce memory and speed up loading
    needed_cols = ['Time', 'Amount', 'Class', 'V14', 'V17', 'V1', 'V2', 'V3', 'V4', 'V10', 'V12']
    
    # Use chunked reading for large file - only get fraud + small sample
    chunks = []
    fraud_rows = []
    legit_sample = []
    
    for chunk in pd.read_csv(DATA_PATH, usecols=needed_cols, chunksize=50000):
        # Collect all fraud
        fraud_chunk = chunk[chunk['Class'] == 1]
        fraud_rows.append(fraud_chunk)
        
        # Get small random sample of legitimate (only if we need more)
        if len(legit_sample) < sample_size:
            legit_chunk = chunk[chunk['Class'] == 0].sample(n=min(100, len(chunk[chunk['Class'] == 0])), random_state=42)
            legit_sample.append(legit_chunk)
    
    # Combine fraud + sample of legitimate
    fraud_df = pd.concat(fraud_rows, ignore_index=True)
    legit_df = pd.concat(legit_sample, ignore_index=True).head(sample_size)
    
    _df_cache = pd.concat([fraud_df, legit_df]).sort_values('Time').reset_index(drop=True)
    
    print(f"✓ Loaded {len(_df_cache)} transactions ({len(fraud_df)} fraud, {len(legit_df)} legit)")
    
    return _df_cache


def calculate_feature_similarity(row1, row2):
    """Calculate simplified similarity using key fraud indicators only"""
    # Use only V14 and V17 (fastest - they're the key fraud indicators)
    v14_diff = abs(row1['V14'] - row2['V14'])
    v17_diff = abs(row1['V17'] - row2['V17'])
    
    # Similarity = 1 if very close, 0 if far apart
    sim = max(0, 1 - (v14_diff + v17_diff) / 20)
    return sim


def build_fraud_network(time_window_seconds=1800, similarity_threshold=0.5, max_nodes=50):
    """
    Build a network graph of fraud transactions and their relationships.
    OPTIMIZED: Reduced nodes, simplified edge calculation.
    """
    global _network_cache
    
    if _network_cache is not None:
        return _network_cache
    
    print("Building fraud network...")
    
    df = load_dataset()
    if df is None:
        return {'nodes': [], 'edges': [], 'sessions': [], 'stats': {}}
    
    # Focus on fraud transactions only for speed
    fraud_df = df[df['Class'] == 1].head(max_nodes).copy()
    
    # Build nodes
    nodes = []
    for idx, row in fraud_df.iterrows():
        node = {
            'id': f"TXN_{idx:06d}",
            'amount': float(row['Amount']),
            'time': float(row['Time']),
            'time_label': format_time(row['Time']),
            'is_fraud': True,
            'risk_score': 0.85,  # Simplified - all fraud is high risk
            'v14': float(row['V14']),
            'v17': float(row['V17']),
        }
        nodes.append(node)
    
    # Build edges - simplified: only temporal connections between fraud
    edges = []
    node_list = list(fraud_df.iterrows())
    
    for i, (idx1, row1) in enumerate(node_list):
        # Only check next few nodes (temporal ordering) for speed
        for j in range(i+1, min(i+10, len(node_list))):
            idx2, row2 = node_list[j]
            time_diff = abs(row1['Time'] - row2['Time'])
            
            if time_diff < time_window_seconds:
                strength = 0.8 if time_diff < 300 else 0.4
                edges.append({
                    'source': f"TXN_{idx1:06d}",
                    'target': f"TXN_{idx2:06d}",
                    'types': ['temporal_strong' if time_diff < 300 else 'temporal', 'confirmed_fraud'],
                    'strength': strength,
                    'time_diff': time_diff,
                    'similarity': 0.7
                })
    
    # Detect fraud sessions
    sessions = detect_fraud_sessions(fraud_df)
    
    _network_cache = {
        'nodes': nodes,
        'edges': edges,
        'sessions': sessions,
        'stats': {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'fraud_count': len(fraud_df),
            'sessions_detected': len(sessions)
        }
    }
    
    print(f"✓ Network built: {len(nodes)} nodes, {len(edges)} edges")
    
    return _network_cache


def check_connection(row1, row2, idx1, idx2, time_window, sim_threshold):
    """Check if two transactions should be connected"""
    time_diff = abs(row1['Time'] - row2['Time'])
    
    # Must be within time window
    if time_diff > time_window:
        return None
    
    # Calculate feature similarity
    similarity = calculate_feature_similarity(row1, row2)
    
    # Determine connection type and strength
    connection_types = []
    strength = 0
    
    # Temporal proximity (within 5 minutes = strong)
    if time_diff < 300:
        connection_types.append('temporal_strong')
        strength += 0.4
    elif time_diff < time_window:
        connection_types.append('temporal')
        strength += 0.2
    
    # Feature similarity
    if similarity > sim_threshold:
        connection_types.append('signature')
        strength += similarity * 0.4
    
    # Amount pattern (structuring detection)
    amt1, amt2 = row1['Amount'], row2['Amount']
    if (9000 < amt1 < 10000 and 9000 < amt2 < 10000) or \
       (abs(amt1 - amt2) < 100 and amt1 > 500):
        connection_types.append('amount_pattern')
        strength += 0.3
    
    # Both are fraud
    if row1['Class'] == 1 and row2['Class'] == 1:
        connection_types.append('confirmed_fraud')
        strength += 0.3
    
    if not connection_types or strength < 0.3:
        return None
    
    return {
        'source': f"TXN_{idx1:06d}",
        'target': f"TXN_{idx2:06d}",
        'types': connection_types,
        'strength': min(1.0, strength),
        'time_diff': time_diff,
        'similarity': similarity
    }


def detect_fraud_sessions(df):
    """Detect clusters of fraud activity"""
    fraud_df = df[df['Class'] == 1].sort_values('Time')
    
    if len(fraud_df) == 0:
        return []
    
    sessions = []
    current_session = []
    
    for idx, row in fraud_df.iterrows():
        if not current_session:
            current_session = [(idx, row)]
        else:
            last_time = current_session[-1][1]['Time']
            if row['Time'] - last_time < 1800:  # 30 min window
                current_session.append((idx, row))
            else:
                if len(current_session) >= 2:
                    sessions.append(format_session(current_session))
                current_session = [(idx, row)]
    
    # Don't forget last session
    if len(current_session) >= 2:
        sessions.append(format_session(current_session))
    
    return sessions


def format_session(session_data):
    """Format a fraud session for display"""
    indices = [s[0] for s in session_data]
    times = [s[1]['Time'] for s in session_data]
    amounts = [s[1]['Amount'] for s in session_data]
    
    return {
        'id': f"SESSION_{min(indices):06d}",
        'transaction_ids': [f"TXN_{idx:06d}" for idx in indices],
        'count': len(session_data),
        'start_time': format_time(min(times)),
        'end_time': format_time(max(times)),
        'duration_minutes': (max(times) - min(times)) / 60,
        'total_amount': sum(amounts)
    }


def calculate_risk_score(row):
    """Calculate a risk score based on known fraud indicators"""
    score = 0.1  # Base score
    
    # V14 is strongly correlated with fraud
    if row['V14'] < -5:
        score += 0.3
    elif row['V14'] < -2:
        score += 0.15
    
    # V17 is another indicator
    if row['V17'] < -3:
        score += 0.2
    
    # High amount
    if row['Amount'] > 5000:
        score += 0.2
    elif row['Amount'] > 1000:
        score += 0.1
    
    # Night time (assuming Time starts at midnight)
    hour = (row['Time'] / 3600) % 24
    if 2 <= hour <= 6:
        score += 0.15
    
    # If actual fraud, set high
    if row['Class'] == 1:
        score = max(score, 0.75)
    
    return min(1.0, score)


def format_time(seconds):
    """Format seconds since start as readable time"""
    hours = int(seconds // 3600) % 24
    minutes = int((seconds % 3600) // 60)
    return f"{hours:02d}:{minutes:02d}"


def get_temporal_stats():
    """Get temporal distribution statistics for the timeline chart"""
    df = load_dataset()
    if df is None:
        return []
    
    # Create 30-minute buckets
    df['bucket'] = (df['Time'] // 1800).astype(int)
    
    stats = []
    for bucket in sorted(df['bucket'].unique()):
        bucket_df = df[df['bucket'] == bucket]
        hour = (bucket * 30) // 60 % 24
        minute = (bucket * 30) % 60
        
        stats.append({
            'bucket': int(bucket),
            'time_label': f"{hour:02d}:{minute:02d}",
            'total': len(bucket_df),
            'fraud_count': len(bucket_df[bucket_df['Class'] == 1]),
            'fraud_rate': len(bucket_df[bucket_df['Class'] == 1]) / max(1, len(bucket_df)),
            'avg_amount': float(bucket_df['Amount'].mean()),
            'max_amount': float(bucket_df['Amount'].max())
        })
    
    return stats


def get_sample_transactions(n=20, include_fraud=True):
    """Get sample transactions for the transaction log"""
    df = load_dataset()
    if df is None:
        return []
    
    if include_fraud:
        # Mix of fraud and legitimate
        fraud = df[df['Class'] == 1].head(n // 2)
        legit = df[df['Class'] == 0].sample(n=n // 2)
        sample = pd.concat([fraud, legit]).sample(frac=1)
    else:
        sample = df.sample(n=min(n, len(df)))
    
    transactions = []
    for idx, row in sample.iterrows():
        transactions.append({
            'id': f"TXN_{idx:06d}",
            'amount': float(row['Amount']),
            'time': float(row['Time']),
            'time_label': format_time(row['Time']),
            'is_fraud': bool(row['Class'] == 1),
            'risk_score': calculate_risk_score(row),
            'status': 'blocked' if row['Class'] == 1 else 'approved',
            'risk_level': 'HIGH' if calculate_risk_score(row) > 0.7 else 'MEDIUM' if calculate_risk_score(row) > 0.4 else 'LOW'
        })
    
    return sorted(transactions, key=lambda x: x['time'], reverse=True)


# Pre-compute on import if dataset exists
if DATA_PATH.exists():
    print(f"Dataset found at {DATA_PATH}")
