"""Audit Logger for Pricing Decisions

Creates comprehensive audit trails for all pricing decisions and changes.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import csv


class PricingAuditLogger:
    """Manages audit logging for pricing decisions"""
    
    def __init__(self, log_directory: str = "pricing_logs"):
        """Initialize audit logger with log directory"""
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.decisions_dir = self.log_directory / "decisions"
        self.changes_dir = self.log_directory / "changes"
        self.alerts_dir = self.log_directory / "alerts"
        
        for dir_path in [self.decisions_dir, self.changes_dir, self.alerts_dir]:
            dir_path.mkdir(exist_ok=True)
            
        # Initialize current log files
        self._initialize_log_files()
        
    def _initialize_log_files(self):
        """Initialize daily log files"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        self.decision_log = self.decisions_dir / f"decisions_{today}.jsonl"
        self.change_log = self.changes_dir / f"changes_{today}.csv"
        self.alert_log = self.alerts_dir / f"alerts_{today}.jsonl"
        
        # Create CSV header if new file
        if not self.change_log.exists():
            with open(self.change_log, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'product_id', 'product_name', 'category',
                    'old_price', 'new_price', 'change_pct', 'confidence',
                    'primary_factor', 'safety_overrides'
                ])
                
    def log_pricing_decision(self, decision_data: Dict) -> str:
        """Log a complete pricing decision"""
        # Add metadata
        log_entry = {
            'log_id': self._generate_log_id(),
            'timestamp': datetime.now().isoformat(),
            'version': '1.0',
            **decision_data
        }
        
        # Write to JSONL file
        with open(self.decision_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
        # Also log price change if applicable
        if decision_data.get('price_change_pct', 0) != 0:
            self._log_price_change(decision_data)
            
        return log_entry['log_id']
        
    def _log_price_change(self, decision_data: Dict):
        """Log price change to CSV for easy analysis"""
        # Determine primary factor
        factors = decision_data.get('factors', {})
        primary_factor = self._get_primary_factor(factors)
        
        # Write to CSV
        with open(self.change_log, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                decision_data.get('product_id', ''),
                decision_data.get('metadata', {}).get('product_name', ''),
                decision_data.get('metadata', {}).get('category', ''),
                decision_data.get('current_price', 0),
                decision_data.get('final_price', 0),
                decision_data.get('price_change_pct', 0),
                decision_data.get('confidence_score', 0),
                primary_factor,
                '|'.join(decision_data.get('safety_overrides', []))
            ])
            
    def _get_primary_factor(self, factors: Dict) -> str:
        """Identify the factor with the greatest impact"""
        if not factors:
            return 'unknown'
            
        max_impact = 0
        primary_factor = 'unknown'
        
        for factor_name, factor_data in factors.items():
            # Calculate impact as deviation from 1.0 times weight
            multiplier = factor_data.get('multiplier', 1.0)
            weight = factor_data.get('weight', 0)
            impact = abs(multiplier - 1.0) * weight
            
            if impact > max_impact:
                max_impact = impact
                primary_factor = factor_name
                
        return primary_factor
        
    def log_alert(self, alert_type: str, severity: str, details: Dict):
        """Log pricing alerts and anomalies"""
        alert_entry = {
            'alert_id': self._generate_log_id('ALERT'),
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'severity': severity,  # 'low', 'medium', 'high', 'critical'
            'details': details
        }
        
        # Write to alert log
        with open(self.alert_log, 'a') as f:
            f.write(json.dumps(alert_entry) + '\n')
            
        # Return alert ID for tracking
        return alert_entry['alert_id']
        
    def log_batch_run(self, run_id: str, summary: Dict):
        """Log summary of batch pricing run"""
        batch_log = self.log_directory / "batch_runs.jsonl"
        
        entry = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'summary': summary
        }
        
        with open(batch_log, 'a') as f:
            f.write(json.dumps(entry) + '\n')
            
    def _generate_log_id(self, prefix: str = 'DEC') -> str:
        """Generate unique log ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
        return f"{prefix}_{timestamp}"
        
    def query_logs(self, 
                  product_id: Optional[str] = None,
                  start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None) -> List[Dict]:
        """Query historical pricing decisions"""
        results = []
        
        # Determine date range
        if not start_date:
            start_date = datetime.now().replace(hour=0, minute=0, second=0)
        if not end_date:
            end_date = datetime.now()
            
        # Iterate through log files in date range
        current_date = start_date
        while current_date <= end_date:
            log_file = self.decisions_dir / f"decisions_{current_date.strftime('%Y-%m-%d')}.jsonl"
            
            if log_file.exists():
                with open(log_file, 'r') as f:
                    for line in f:
                        entry = json.loads(line)
                        
                        # Filter by product_id if specified
                        if product_id and entry.get('product_id') != product_id:
                            continue
                            
                        results.append(entry)
                        
            current_date = current_date.replace(day=current_date.day + 1)
            
        return results
        
    def generate_daily_report(self, date: Optional[datetime] = None) -> Dict:
        """Generate daily pricing activity report"""
        if not date:
            date = datetime.now()
            
        report_date = date.strftime("%Y-%m-%d")
        
        # Read change log
        change_file = self.changes_dir / f"changes_{report_date}.csv"
        if not change_file.exists():
            return {'date': report_date, 'total_changes': 0}
            
        changes = []
        with open(change_file, 'r') as f:
            reader = csv.DictReader(f)
            changes = list(reader)
            
        # Calculate statistics
        total_changes = len(changes)
        avg_change = sum(float(c['change_pct']) for c in changes) / total_changes if total_changes > 0 else 0
        
        # Count by category
        category_counts = {}
        for change in changes:
            cat = change['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
            
        # Count primary factors
        factor_counts = {}
        for change in changes:
            factor = change['primary_factor']
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
            
        return {
            'date': report_date,
            'total_changes': total_changes,
            'average_change_pct': avg_change,
            'increases': sum(1 for c in changes if float(c['change_pct']) > 0),
            'decreases': sum(1 for c in changes if float(c['change_pct']) < 0),
            'by_category': category_counts,
            'by_factor': factor_counts,
            'confidence_avg': sum(float(c['confidence']) for c in changes) / total_changes if total_changes > 0 else 0
        }


class PerformanceTracker:
    """Track pricing performance metrics"""
    
    def __init__(self, metrics_directory: str = "pricing_metrics"):
        """Initialize performance tracker"""
        self.metrics_dir = Path(metrics_directory)
        self.metrics_dir.mkdir(exist_ok=True)
        
    def track_decision_metrics(self, decision_data: Dict):
        """Track metrics for a pricing decision"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'product_id': decision_data.get('product_id'),
            'confidence': decision_data.get('confidence_score'),
            'price_change': decision_data.get('price_change_pct'),
            'factors': {}
        }
        
        # Extract factor metrics
        for factor_name, factor_data in decision_data.get('factors', {}).items():
            metrics['factors'][factor_name] = {
                'multiplier': factor_data.get('multiplier'),
                'confidence': factor_data.get('confidence'),
                'weight': factor_data.get('weight')
            }
            
        # Write to daily metrics file
        today = datetime.now().strftime("%Y-%m-%d")
        metrics_file = self.metrics_dir / f"metrics_{today}.jsonl"
        
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
            
    def calculate_factor_performance(self, days: int = 7) -> Dict:
        """Calculate factor performance over specified days"""
        end_date = datetime.now()
        start_date = end_date.replace(day=end_date.day - days)
        
        factor_stats = {}
        total_decisions = 0
        
        # Read metrics files
        current_date = start_date
        while current_date <= end_date:
            metrics_file = self.metrics_dir / f"metrics_{current_date.strftime('%Y-%m-%d')}.jsonl"
            
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    for line in f:
                        entry = json.loads(line)
                        total_decisions += 1
                        
                        # Aggregate factor data
                        for factor_name, factor_data in entry.get('factors', {}).items():
                            if factor_name not in factor_stats:
                                factor_stats[factor_name] = {
                                    'total_impact': 0,
                                    'avg_confidence': 0,
                                    'count': 0
                                }
                                
                            stats = factor_stats[factor_name]
                            stats['total_impact'] += abs(factor_data.get('multiplier', 1.0) - 1.0)
                            stats['avg_confidence'] += factor_data.get('confidence', 0)
                            stats['count'] += 1
                            
            current_date = current_date.replace(day=current_date.day + 1)
            
        # Calculate averages
        for factor_name, stats in factor_stats.items():
            if stats['count'] > 0:
                stats['avg_impact'] = stats['total_impact'] / stats['count']
                stats['avg_confidence'] = stats['avg_confidence'] / stats['count']
                
        return {
            'period_days': days,
            'total_decisions': total_decisions,
            'factor_performance': factor_stats
        }