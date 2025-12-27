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
    
    # Get fraud transactions (high risk)
    fraud_df = df[df['Class'] == 1].head(max_nodes).copy()
    fraud_df['risk_score'] = 0.95  # High risk base score
    
    # Get sample of legitimate transactions (low/medium risk)
    # We'll take about 50% as many legit nodes as fraud nodes to keep graph focused but varied
    legit_count = min(len(df[df['Class'] == 0]), max(20, int(max_nodes * 0.5)))
    legit_df = df[df['Class'] == 0].sample(n=legit_count, random_state=42).copy()
    
    # Simulate risk scores for legit transactions (mostly low, some medium)
    # 80% low risk (0.0-0.3), 20% medium risk (0.3-0.7)
    legit_df['risk_score'] = np.random.choice(
        [np.random.uniform(0.05, 0.3), np.random.uniform(0.35, 0.65)],
        size=len(legit_df),
        p=[0.8, 0.2]
    )
    
    # Combine and sort by time
    combined_df = pd.concat([fraud_df, legit_df]).sort_values('Time')
    
    # Build nodes
    nodes = []
    for idx, row in combined_df.iterrows():
        node = {
            'id': f"TXN_{idx:06d}",
            'amount': float(row['Amount']),
            'time': float(row['Time']),
            'time_label': format_time(row['Time']),
            'is_fraud': bool(row['Class'] == 1),
            'risk_score': float(row.get('risk_score', 0)),
            'v14': float(row['V14']),
            'v17': float(row['V17'])
        }
        nodes.append(node)
    
    # Build edges (temporal connections)
    edges = []
    # Create a list for faster iteration
    node_list = list(enumerate(zip(combined_df.index, combined_df.to_dict('records'))))
    
    for i, (idx_pos, (idx1, row1)) in enumerate(node_list):
        # Only check next few nodes (temporal ordering) for speed
        for j in range(i+1, min(i+15, len(node_list))):
            idx_pos2, (idx2, row2) = node_list[j]
            time_diff = abs(row1['Time'] - row2['Time'])
            
            if time_diff < time_window_seconds:
                # Stronger connection for fraud-fraud
                is_both_fraud = (row1['Class'] == 1 and row2['Class'] == 1)
                
                strength = 0.8 if time_diff < 300 else 0.4
                if is_both_fraud:
                    strength += 0.2
                
                # Connect if valid
                if strength > 0.3:
                    edges.append({
                        'source': f"TXN_{idx1:06d}",
                        'target': f"TXN_{idx2:06d}",
                        'types': ['temporal_strong' if time_diff < 300 else 'temporal', 
                                 'confirmed_fraud' if is_both_fraud else 'potential_link'],
                        'strength': min(1.0, strength),
                        'time_diff': time_diff,
                        'similarity': calculate_feature_similarity(row1, row2)
                    })
    
def get_cluster_explanation(session_id, nodes):
    """
    Generate SHAP-like explanations for a cluster/session.
    Since we don't have a live model, we calculate feature deviations
    from the legitimate baseline to simulate SHAP contribution.
    """
    # Filter nodes for this session
    session_nodes = [n for n in nodes if n.get('id') in session_id or session_id == 'all']
    if not session_nodes:
        return []

    # Calculate average feature values for cluster
    avg_v14 = np.mean([n.get('v14', 0) for n in session_nodes])
    avg_v17 = np.mean([n.get('v17', 0) for n in session_nodes])
    avg_amount = np.mean([n.get('amount', 0) for n in session_nodes])

    # Simplified SHAP-like contributors (deviation from 'normal')
    explanations = [
        {
            'feature': 'V14 (Behavior)',
            'value': avg_v14,
            'importance': abs(avg_v14 + 10) / 20.0, # Mock importance based on deviation
            'contribution': 'negative' if avg_v14 < -5 else 'positive',
            'description': 'Abnormal transaction pattern detected'
        },
        {
            'feature': 'V17 (Identity)',
            'value': avg_v17,
            'importance': abs(avg_v17 + 5) / 15.0,
            'contribution': 'negative' if avg_v17 < -5 else 'positive',
            'description': 'Identity verification anomaly'
        },
        {
            'feature': 'Time Burst',
            'value': len(session_nodes),
            'importance': min(0.8, len(session_nodes) * 0.1),
            'contribution': 'positive',
            'description': 'High velocity transaction burst'
        }
    ]
    
    # Sort by importance
    return sorted(explanations, key=lambda x: x['importance'], reverse=True)


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
