"""
Anomaly Detection Systems
Detect unusual patterns in market, data quality, compliance, and system performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class MarketAnomalyDetector:
    """Detect unusual market movements and patterns"""
    
    def __init__(self):
        self.models = {}
        self.thresholds = {
            'price_spike': 0.20,  # 20% change
            'volume_spike': 3.0,  # 3x normal volume
            'competitor_deviation': 0.15,  # 15% from market
            'demand_anomaly': 2.5  # 2.5 std deviations
        }
        self.baseline_stats = {}
        
    def train_detector(self, historical_data: pd.DataFrame, category: str) -> Dict:
        """Train anomaly detection model for category"""
        
        # Calculate baseline statistics
        baseline = self._calculate_baseline_stats(historical_data)
        self.baseline_stats[category] = baseline
        
        # Prepare features
        features = self._extract_anomaly_features(historical_data)
        
        # Train Isolation Forest
        model = IsolationForest(
            contamination=0.05,  # Expect 5% anomalies
            random_state=42,
            n_estimators=100
        )
        model.fit(features)
        
        self.models[category] = {
            'model': model,
            'scaler': StandardScaler().fit(features),
            'feature_names': features.columns.tolist(),
            'last_update': pd.Timestamp.now()
        }
        
        return {
            'category': category,
            'baseline_established': True,
            'features_tracked': len(features.columns),
            'training_samples': len(features)
        }
    
    def detect_anomalies(self, current_data: pd.DataFrame, category: str) -> Dict:
        """Detect anomalies in current market data"""
        
        anomalies = {
            'price_anomalies': [],
            'volume_anomalies': [],
            'competitive_anomalies': [],
            'pattern_anomalies': [],
            'composite_score': 0
        }
        
        # Price spike detection
        price_anomalies = self._detect_price_anomalies(current_data, category)
        anomalies['price_anomalies'] = price_anomalies
        
        # Volume anomaly detection
        volume_anomalies = self._detect_volume_anomalies(current_data, category)
        anomalies['volume_anomalies'] = volume_anomalies
        
        # Competitive anomaly detection
        competitive_anomalies = self._detect_competitive_anomalies(current_data)
        anomalies['competitive_anomalies'] = competitive_anomalies
        
        # Pattern-based anomaly detection
        if category in self.models:
            pattern_anomalies = self._detect_pattern_anomalies(current_data, category)
            anomalies['pattern_anomalies'] = pattern_anomalies
        
        # Calculate composite anomaly score
        anomalies['composite_score'] = self._calculate_composite_score(anomalies)
        anomalies['alert_level'] = self._determine_alert_level(anomalies['composite_score'])
        
        return anomalies
    
    def _calculate_baseline_stats(self, historical_data: pd.DataFrame) -> Dict:
        """Calculate baseline statistics for normal behavior"""
        
        return {
            'price_mean': historical_data.groupby('product_id')['price'].mean().to_dict(),
            'price_std': historical_data.groupby('product_id')['price'].std().to_dict(),
            'volume_mean': historical_data.groupby('product_id')['units_sold'].mean().to_dict(),
            'volume_std': historical_data.groupby('product_id')['units_sold'].std().to_dict(),
            'price_change_mean': historical_data.groupby('product_id')['price'].pct_change().mean().to_dict(),
            'price_change_std': historical_data.groupby('product_id')['price'].pct_change().std().to_dict()
        }
    
    def _extract_anomaly_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for anomaly detection"""
        
        features = pd.DataFrame()
        
        # Price-based features
        features['price_zscore'] = (data['price'] - data['price'].mean()) / data['price'].std()
        features['price_change'] = data['price'].pct_change()
        features['price_volatility'] = data['price'].rolling(7).std()
        
        # Volume-based features
        features['volume_zscore'] = (data['units_sold'] - data['units_sold'].mean()) / data['units_sold'].std()
        features['volume_change'] = data['units_sold'].pct_change()
        
        # Time-based features
        features['hour'] = pd.to_datetime(data['timestamp']).dt.hour
        features['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek
        features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
        
        # Competition features
        if 'competitor_price' in data.columns:
            features['price_gap'] = data['price'] - data['competitor_price']
            features['price_gap_pct'] = features['price_gap'] / data['competitor_price']
        
        return features.dropna()
    
    def _detect_price_anomalies(self, data: pd.DataFrame, category: str) -> List[Dict]:
        """Detect price-related anomalies"""
        
        anomalies = []
        baseline = self.baseline_stats.get(category, {})
        
        for _, row in data.iterrows():
            product_id = row['product_id']
            current_price = row['price']
            
            # Check against baseline
            if product_id in baseline.get('price_mean', {}):
                mean_price = baseline['price_mean'][product_id]
                std_price = baseline['price_std'][product_id]
                
                z_score = (current_price - mean_price) / std_price if std_price > 0 else 0
                
                if abs(z_score) > 3:
                    anomalies.append({
                        'type': 'price_spike',
                        'product_id': product_id,
                        'severity': abs(z_score),
                        'current_price': current_price,
                        'expected_range': (mean_price - 2*std_price, mean_price + 2*std_price),
                        'timestamp': row.get('timestamp', pd.Timestamp.now())
                    })
            
            # Check for sudden changes
            if 'previous_price' in row:
                price_change = abs(current_price - row['previous_price']) / row['previous_price']
                if price_change > self.thresholds['price_spike']:
                    anomalies.append({
                        'type': 'sudden_price_change',
                        'product_id': product_id,
                        'change_percent': price_change * 100,
                        'timestamp': row.get('timestamp', pd.Timestamp.now())
                    })
        
        return anomalies
    
    def _detect_volume_anomalies(self, data: pd.DataFrame, category: str) -> List[Dict]:
        """Detect volume-related anomalies"""
        
        anomalies = []
        baseline = self.baseline_stats.get(category, {})
        
        for _, row in data.iterrows():
            product_id = row['product_id']
            current_volume = row.get('units_sold', 0)
            
            if product_id in baseline.get('volume_mean', {}):
                mean_volume = baseline['volume_mean'][product_id]
                std_volume = baseline['volume_std'][product_id]
                
                if mean_volume > 0:
                    volume_ratio = current_volume / mean_volume
                    
                    if volume_ratio > self.thresholds['volume_spike']:
                        anomalies.append({
                            'type': 'volume_spike',
                            'product_id': product_id,
                            'severity': volume_ratio,
                            'current_volume': current_volume,
                            'normal_volume': mean_volume,
                            'timestamp': row.get('timestamp', pd.Timestamp.now())
                        })
                    elif current_volume < mean_volume * 0.2:  # 80% drop
                        anomalies.append({
                            'type': 'volume_drop',
                            'product_id': product_id,
                            'severity': 1 - (current_volume / mean_volume),
                            'current_volume': current_volume,
                            'normal_volume': mean_volume,
                            'timestamp': row.get('timestamp', pd.Timestamp.now())
                        })
        
        return anomalies
    
    def _detect_competitive_anomalies(self, data: pd.DataFrame) -> List[Dict]:
        """Detect competitive pricing anomalies"""
        
        anomalies = []
        
        if 'competitor_price' not in data.columns:
            return anomalies
        
        # Group by timestamp to analyze market
        for timestamp, group in data.groupby('timestamp'):
            market_avg = group['price'].mean()
            
            for _, row in group.iterrows():
                price_deviation = abs(row['price'] - market_avg) / market_avg
                
                if price_deviation > self.thresholds['competitor_deviation']:
                    anomalies.append({
                        'type': 'market_deviation',
                        'product_id': row['product_id'],
                        'deviation_percent': price_deviation * 100,
                        'product_price': row['price'],
                        'market_average': market_avg,
                        'timestamp': timestamp
                    })
        
        return anomalies
    
    def _detect_pattern_anomalies(self, data: pd.DataFrame, category: str) -> List[Dict]:
        """Detect pattern-based anomalies using ML model"""
        
        anomalies = []
        
        model_info = self.models[category]
        model = model_info['model']
        scaler = model_info['scaler']
        
        # Extract features
        features = self._extract_anomaly_features(data)
        
        # Ensure same features as training
        for col in model_info['feature_names']:
            if col not in features.columns:
                features[col] = 0
        features = features[model_info['feature_names']]
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict anomalies (-1 for anomaly, 1 for normal)
        predictions = model.predict(features_scaled)
        anomaly_scores = model.score_samples(features_scaled)
        
        # Process anomalies
        anomaly_indices = np.where(predictions == -1)[0]
        
        for idx in anomaly_indices:
            if idx < len(data):
                row = data.iloc[idx]
                anomalies.append({
                    'type': 'pattern_anomaly',
                    'product_id': row.get('product_id', 'unknown'),
                    'anomaly_score': -anomaly_scores[idx],  # Convert to positive
                    'features': features.iloc[idx].to_dict(),
                    'timestamp': row.get('timestamp', pd.Timestamp.now())
                })
        
        return anomalies
    
    def _calculate_composite_score(self, anomalies: Dict) -> float:
        """Calculate overall anomaly score"""
        
        weights = {
            'price_anomalies': 0.3,
            'volume_anomalies': 0.2,
            'competitive_anomalies': 0.2,
            'pattern_anomalies': 0.3
        }
        
        score = 0
        for anomaly_type, weight in weights.items():
            if anomaly_type in anomalies:
                # Count and severity
                count = len(anomalies[anomaly_type])
                if count > 0:
                    avg_severity = np.mean([a.get('severity', 1) for a in anomalies[anomaly_type]])
                    score += weight * min(1, count / 10) * avg_severity
        
        return min(1, score)
    
    def _determine_alert_level(self, composite_score: float) -> str:
        """Determine alert level based on anomaly score"""
        
        if composite_score > 0.7:
            return 'critical'
        elif composite_score > 0.4:
            return 'warning'
        elif composite_score > 0.2:
            return 'info'
        else:
            return 'normal'


class DataQualityMonitor:
    """Monitor data quality and detect issues"""
    
    def __init__(self):
        self.quality_metrics = {}
        self.quality_thresholds = {
            'completeness': 0.95,
            'accuracy': 0.98,
            'consistency': 0.95,
            'timeliness': 24  # hours
        }
        
    def check_data_quality(self, data: pd.DataFrame, data_type: str) -> Dict:
        """Comprehensive data quality check"""
        
        quality_report = {
            'data_type': data_type,
            'timestamp': pd.Timestamp.now(),
            'issues': [],
            'metrics': {}
        }
        
        # Completeness check
        completeness = self._check_completeness(data)
        quality_report['metrics']['completeness'] = completeness
        
        # Accuracy check
        accuracy_issues = self._check_accuracy(data)
        quality_report['issues'].extend(accuracy_issues)
        
        # Consistency check
        consistency_issues = self._check_consistency(data)
        quality_report['issues'].extend(consistency_issues)
        
        # Timeliness check
        timeliness_issues = self._check_timeliness(data)
        quality_report['issues'].extend(timeliness_issues)
        
        # Overall quality score
        quality_report['quality_score'] = self._calculate_quality_score(quality_report)
        quality_report['quality_status'] = 'good' if quality_report['quality_score'] > 0.9 else 'poor'
        
        return quality_report
    
    def _check_completeness(self, data: pd.DataFrame) -> float:
        """Check data completeness"""
        
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        
        return 1 - (missing_cells / total_cells) if total_cells > 0 else 0
    
    def _check_accuracy(self, data: pd.DataFrame) -> List[Dict]:
        """Check data accuracy and validity"""
        
        issues = []
        
        # Price validation
        if 'price' in data.columns:
            invalid_prices = data[(data['price'] <= 0) | (data['price'] > 10000)]
            if len(invalid_prices) > 0:
                issues.append({
                    'type': 'invalid_price',
                    'count': len(invalid_prices),
                    'examples': invalid_prices.head(5).to_dict('records')
                })
        
        # Quantity validation
        if 'units_sold' in data.columns:
            invalid_quantities = data[data['units_sold'] < 0]
            if len(invalid_quantities) > 0:
                issues.append({
                    'type': 'negative_quantity',
                    'count': len(invalid_quantities),
                    'examples': invalid_quantities.head(5).to_dict('records')
                })
        
        # Date validation
        if 'timestamp' in data.columns:
            future_dates = data[pd.to_datetime(data['timestamp']) > pd.Timestamp.now()]
            if len(future_dates) > 0:
                issues.append({
                    'type': 'future_timestamp',
                    'count': len(future_dates),
                    'examples': future_dates.head(5).to_dict('records')
                })
        
        return issues
    
    def _check_consistency(self, data: pd.DataFrame) -> List[Dict]:
        """Check data consistency"""
        
        issues = []
        
        # Check for duplicate entries
        if 'product_id' in data.columns and 'timestamp' in data.columns:
            duplicates = data.duplicated(subset=['product_id', 'timestamp'])
            if duplicates.sum() > 0:
                issues.append({
                    'type': 'duplicate_entries',
                    'count': duplicates.sum(),
                    'percentage': (duplicates.sum() / len(data)) * 100
                })
        
        # Check for logical inconsistencies
        if 'price' in data.columns and 'cost' in data.columns:
            negative_margin = data[data['price'] < data['cost']]
            if len(negative_margin) > 0:
                issues.append({
                    'type': 'negative_margin',
                    'count': len(negative_margin),
                    'examples': negative_margin.head(5).to_dict('records')
                })
        
        return issues
    
    def _check_timeliness(self, data: pd.DataFrame) -> List[Dict]:
        """Check data timeliness"""
        
        issues = []
        
        if 'timestamp' in data.columns:
            latest_timestamp = pd.to_datetime(data['timestamp']).max()
            hours_old = (pd.Timestamp.now() - latest_timestamp).total_seconds() / 3600
            
            if hours_old > self.quality_thresholds['timeliness']:
                issues.append({
                    'type': 'stale_data',
                    'hours_old': hours_old,
                    'latest_timestamp': latest_timestamp
                })
        
        return issues
    
    def _calculate_quality_score(self, quality_report: Dict) -> float:
        """Calculate overall data quality score"""
        
        # Start with completeness score
        score = quality_report['metrics'].get('completeness', 1.0)
        
        # Deduct for issues
        issue_penalty = len(quality_report['issues']) * 0.1
        score = max(0, score - issue_penalty)
        
        return score


class SystemPerformanceMonitor:
    """Monitor pricing system performance"""
    
    def __init__(self):
        self.performance_metrics = {}
        self.alert_thresholds = {
            'response_time': 1000,  # milliseconds
            'error_rate': 0.01,  # 1%
            'accuracy_deviation': 0.1  # 10%
        }
        
    def monitor_system_health(self, system_logs: pd.DataFrame) -> Dict:
        """Monitor overall system health"""
        
        health_report = {
            'timestamp': pd.Timestamp.now(),
            'metrics': {},
            'anomalies': [],
            'status': 'healthy'
        }
        
        # Response time analysis
        if 'response_time' in system_logs.columns:
            avg_response = system_logs['response_time'].mean()
            health_report['metrics']['avg_response_time'] = avg_response
            
            if avg_response > self.alert_thresholds['response_time']:
                health_report['anomalies'].append({
                    'type': 'slow_response',
                    'avg_ms': avg_response,
                    'threshold_ms': self.alert_thresholds['response_time']
                })
        
        # Error rate analysis
        if 'error' in system_logs.columns:
            error_rate = system_logs['error'].sum() / len(system_logs)
            health_report['metrics']['error_rate'] = error_rate
            
            if error_rate > self.alert_thresholds['error_rate']:
                health_report['anomalies'].append({
                    'type': 'high_error_rate',
                    'rate': error_rate,
                    'threshold': self.alert_thresholds['error_rate']
                })
        
        # Prediction accuracy
        if 'predicted' in system_logs.columns and 'actual' in system_logs.columns:
            accuracy_deviation = np.mean(np.abs(system_logs['predicted'] - system_logs['actual']) / system_logs['actual'])
            health_report['metrics']['accuracy_deviation'] = accuracy_deviation
            
            if accuracy_deviation > self.alert_thresholds['accuracy_deviation']:
                health_report['anomalies'].append({
                    'type': 'poor_accuracy',
                    'deviation': accuracy_deviation,
                    'threshold': self.alert_thresholds['accuracy_deviation']
                })
        
        # Determine overall status
        if len(health_report['anomalies']) > 2:
            health_report['status'] = 'critical'
        elif len(health_report['anomalies']) > 0:
            health_report['status'] = 'warning'
        
        return health_report