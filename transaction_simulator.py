"""
Transaction Simulator
Background thread that streams real-time transactions via WebSocket
"""

import time
import random
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

class TransactionSimulator:
    """Simulates real-time transaction streaming"""
    
    def __init__(self, socketio, model_available=False, data_available=False):
        self.socketio = socketio
        self.model_available = model_available
        self.data_available = data_available
        self.running = False
        self.dataset_df = None
        self.dataset_index = 0
        
        # Load dataset if available
        if data_available:
            try:
                from data_processor import load_dataset
                self.dataset_df = load_dataset(sample_size=1000)
                if self.dataset_df is not None:
                    print(f"✓ Loaded {len(self.dataset_df)} transactions for simulation")
            except Exception as e:
                print(f"⚠ Error loading dataset for simulator: {e}")
    
    def get_next_transaction(self):
        """Get next transaction (from dataset or generate mock)"""
        if self.dataset_df is not None and len(self.dataset_df) > 0:
            # Use real dataset
            row = self.dataset_df.iloc[self.dataset_index % len(self.dataset_df)]
            self.dataset_index += 1
            
            # Build feature dict
            feature_dict = {
                'Amount': float(row['Amount']),
                'Time': float(row['Time'])
            }
            for i in range(1, 29):
                feature_dict[f'V{i}'] = float(row.get(f'V{i}', 0))
            
            # Predict using model if available
            if self.model_available:
                try:
                    from model_explainer import explain_transaction_from_request
                    result = explain_transaction_from_request(feature_dict)
                    fraud_prob = result['fraud_probability']
                    is_fraud = result['is_fraud']
                    shap_explanation = result['shap_explanation']
                except Exception as e:
                    print(f"Error in model prediction: {e}")
                    fraud_prob = float(row.get('Class', 0))
                    is_fraud = bool(row.get('Class', 0) == 1)
                    shap_explanation = []
            else:
                fraud_prob = float(row.get('Class', 0))
                is_fraud = bool(row.get('Class', 0) == 1)
                shap_explanation = []
            
            transaction = {
                'id': f"TXN_{int(row.name):06d}",
                'amount': float(row['Amount']),
                'timestamp': datetime.now().isoformat(),
                'fraud_probability': fraud_prob,
                'is_fraud': is_fraud,
                'risk_level': 'HIGH' if fraud_prob > 0.7 else 'MEDIUM' if fraud_prob > 0.4 else 'LOW',
                'status': 'blocked' if is_fraud else 'approved',
                'shap_explanation': shap_explanation[:3]  # Top 3 features
            }
        else:
            # Generate mock transaction
            amounts = [49.99, 129.00, 88.50, 203.45, 450.00, 67.30, 1250.00, 9999.00, 15.99, 89.00]
            fraud_prob = random.uniform(0.05, 0.95)
            
            if fraud_prob > 0.7:
                risk_level = "HIGH"
                status = "blocked"
            elif fraud_prob > 0.4:
                risk_level = "MEDIUM"
                status = "flagged"
            else:
                risk_level = "LOW"
                status = "approved"
            
            transaction = {
                'id': f"TXN_{random.randint(100000, 999999)}",
                'amount': random.choice(amounts),
                'timestamp': datetime.now().isoformat(),
                'fraud_probability': round(fraud_prob, 4),
                'is_fraud': fraud_prob > 0.5,
                'risk_level': risk_level,
                'status': status
            }
        
        return transaction
    
    def run(self):
        """Main simulation loop"""
        self.running = True
        print("Transaction simulator started (2-5 second intervals)")
        
        while self.running:
            try:
                # Wait random interval (2-5 seconds)
                wait_time = random.uniform(2, 5)
                time.sleep(wait_time)
                
                # Get next transaction
                transaction = self.get_next_transaction()
                
                # Emit via WebSocket
                self.socketio.emit('new_transaction', transaction)
                
                # If fraud detected, emit alert
                if transaction.get('is_fraud', False) and transaction.get('fraud_probability', 0) > 0.7:
                    alert = {
                        'transaction_id': transaction['id'],
                        'amount': transaction['amount'],
                        'fraud_probability': transaction['fraud_probability'],
                        'timestamp': transaction['timestamp'],
                        'risk_level': transaction.get('risk_level', 'HIGH')
                    }
                    self.socketio.emit('fraud_alert', alert)
                    print(f"🚨 Fraud alert: {transaction['id']} (prob: {transaction['fraud_probability']:.2f})")
                
            except Exception as e:
                print(f"Error in transaction simulator: {e}")
                time.sleep(5)  # Wait before retrying
    
    def stop(self):
        """Stop the simulator"""
        self.running = False

