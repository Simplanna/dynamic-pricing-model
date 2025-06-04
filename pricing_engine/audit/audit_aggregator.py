"""Aggregates audit data from multiple sources for comprehensive tracking."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable
import logging
from pathlib import Path
import json
from collections import defaultdict

from .regulatory_audit import RegulatoryAuditTrail, AuditEvent, AuditEventType
from ..utils.audit_logger import PricingAuditLogger
from ..core.models import Market, ProductCategory

logger = logging.getLogger(__name__)


@dataclass
class AuditMetrics:
    """Aggregated audit metrics."""
    total_events: int
    events_by_type: Dict[str, int]
    compliance_rate: float
    price_change_count: int
    average_price_change: float
    manual_interventions: int
    system_errors: int
    rollback_count: int
    processing_time_avg: float
    data_quality_score: float


class AuditAggregator:
    """Aggregates and analyzes audit data from multiple sources."""
    
    def __init__(self, regulatory_audit: RegulatoryAuditTrail,
                 pricing_audit: PricingAuditLogger,
                 cache_dir: Optional[Path] = None):
        self.regulatory_audit = regulatory_audit
        self.pricing_audit = pricing_audit
        self.cache_dir = cache_dir or Path("audit_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Aggregation functions
        self._aggregators: Dict[str, Callable] = {
            'hourly': self._aggregate_hourly,
            'daily': self._aggregate_daily,
            'weekly': self._aggregate_weekly,
            'monthly': self._aggregate_monthly
        }
        
        logger.info("Audit aggregator initialized")
    
    def get_unified_view(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get unified view of all audit data."""
        # Query from both sources
        regulatory_events = self.regulatory_audit.query_events(start_date, end_date)
        pricing_logs = self.pricing_audit.query_logs(start_date=start_date, end_date=end_date)
        
        # Combine and organize
        unified = {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'regulatory_events': len(regulatory_events),
            'pricing_decisions': len(pricing_logs),
            'summary': self._calculate_summary_metrics(regulatory_events, pricing_logs),
            'timeline': self._create_unified_timeline(regulatory_events, pricing_logs),
            'anomalies': self._detect_anomalies(regulatory_events, pricing_logs),
            'compliance_status': self._assess_compliance_status(regulatory_events)
        }
        
        return unified
    
    def aggregate_metrics(self, period: str, date: Optional[datetime] = None) -> AuditMetrics:
        """Aggregate metrics for specified period."""
        if period not in self._aggregators:
            raise ValueError(f"Invalid period: {period}. Must be one of {list(self._aggregators.keys())}")
        
        if not date:
            date = datetime.now()
        
        # Check cache first
        cached = self._get_cached_metrics(period, date)
        if cached:
            return cached
        
        # Calculate metrics
        metrics = self._aggregators[period](date)
        
        # Cache results
        self._cache_metrics(period, date, metrics)
        
        return metrics
    
    def generate_audit_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive audit dashboard data."""
        now = datetime.now()
        
        dashboard = {
            'timestamp': now.isoformat(),
            'current_metrics': {
                'hourly': self.aggregate_metrics('hourly', now),
                'daily': self.aggregate_metrics('daily', now),
                'weekly': self.aggregate_metrics('weekly', now)
            },
            'trends': self._calculate_trends(),
            'alerts': self._get_audit_alerts(),
            'data_quality': self._assess_data_quality(),
            'system_health': self._check_system_health()
        }
        
        return dashboard
    
    def reconcile_audit_trails(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Reconcile different audit trails for consistency."""
        # Get events from both systems
        regulatory_events = self.regulatory_audit.query_events(start_date, end_date)
        pricing_logs = self.pricing_audit.query_logs(start_date=start_date, end_date=end_date)
        
        # Create lookup maps
        regulatory_by_sku = defaultdict(list)
        for event in regulatory_events:
            if event.product_sku:
                regulatory_by_sku[event.product_sku].append(event)
        
        pricing_by_sku = defaultdict(list)
        for log in pricing_logs:
            sku = log.get('product_id')
            if sku:
                pricing_by_sku[sku].append(log)
        
        # Find discrepancies
        discrepancies = []
        all_skus = set(regulatory_by_sku.keys()) | set(pricing_by_sku.keys())
        
        for sku in all_skus:
            reg_events = regulatory_by_sku.get(sku, [])
            price_logs = pricing_by_sku.get(sku, [])
            
            # Check for missing records
            if len(reg_events) != len(price_logs):
                discrepancies.append({
                    'sku': sku,
                    'type': 'count_mismatch',
                    'regulatory_count': len(reg_events),
                    'pricing_count': len(price_logs)
                })
            
            # Check for data consistency
            for reg_event in reg_events:
                if reg_event.event_type == AuditEventType.PRICE_CHANGE:
                    # Find corresponding pricing log
                    matching_log = self._find_matching_log(reg_event, price_logs)
                    if not matching_log:
                        discrepancies.append({
                            'sku': sku,
                            'type': 'missing_pricing_log',
                            'regulatory_event': reg_event.event_id,
                            'timestamp': reg_event.timestamp.isoformat()
                        })
        
        reconciliation = {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'total_regulatory_events': len(regulatory_events),
            'total_pricing_logs': len(pricing_logs),
            'discrepancies': discrepancies,
            'discrepancy_rate': len(discrepancies) / max(len(regulatory_events), 1),
            'recommendations': self._generate_reconciliation_recommendations(discrepancies)
        }
        
        return reconciliation
    
    def export_audit_archive(self, start_date: datetime, end_date: datetime,
                           output_path: Path) -> Dict[str, Any]:
        """Export complete audit archive for specified period."""
        # Create archive directory
        archive_dir = output_path / f"audit_archive_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Export regulatory audit
        regulatory_export = self.regulatory_audit.export_for_regulator(
            start_date, end_date,
            archive_dir / "regulatory_audit.json"
        )
        
        # Export pricing audit
        pricing_logs = self.pricing_audit.query_logs(start_date=start_date, end_date=end_date)
        with open(archive_dir / "pricing_audit.json", 'w') as f:
            json.dump(pricing_logs, f, indent=2, default=str)
        
        # Export aggregated metrics
        metrics = {
            'daily': [],
            'weekly': [],
            'monthly': []
        }
        
        current = start_date
        while current <= end_date:
            daily_metrics = self.aggregate_metrics('daily', current)
            metrics['daily'].append({
                'date': current.isoformat(),
                'metrics': self._metrics_to_dict(daily_metrics)
            })
            current += timedelta(days=1)
        
        with open(archive_dir / "aggregated_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create manifest
        manifest = {
            'created_at': datetime.now().isoformat(),
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'contents': {
                'regulatory_audit': str(archive_dir / "regulatory_audit.json"),
                'pricing_audit': str(archive_dir / "pricing_audit.json"),
                'aggregated_metrics': str(archive_dir / "aggregated_metrics.json")
            },
            'summary': regulatory_export
        }
        
        with open(archive_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Audit archive exported to {archive_dir}")
        
        return manifest
    
    def _aggregate_hourly(self, date: datetime) -> AuditMetrics:
        """Aggregate metrics for an hour."""
        start = date.replace(minute=0, second=0, microsecond=0)
        end = start + timedelta(hours=1)
        
        return self._calculate_metrics_for_period(start, end)
    
    def _aggregate_daily(self, date: datetime) -> AuditMetrics:
        """Aggregate metrics for a day."""
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        
        return self._calculate_metrics_for_period(start, end)
    
    def _aggregate_weekly(self, date: datetime) -> AuditMetrics:
        """Aggregate metrics for a week."""
        # Start from Monday
        days_since_monday = date.weekday()
        start = date.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_since_monday)
        end = start + timedelta(days=7)
        
        return self._calculate_metrics_for_period(start, end)
    
    def _aggregate_monthly(self, date: datetime) -> AuditMetrics:
        """Aggregate metrics for a month."""
        start = date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Calculate last day of month
        if date.month == 12:
            end = datetime(date.year + 1, 1, 1) - timedelta(seconds=1)
        else:
            end = datetime(date.year, date.month + 1, 1) - timedelta(seconds=1)
        
        return self._calculate_metrics_for_period(start, end)
    
    def _calculate_metrics_for_period(self, start: datetime, end: datetime) -> AuditMetrics:
        """Calculate metrics for a specific period."""
        # Query events
        events = self.regulatory_audit.query_events(start, end)
        pricing_logs = self.pricing_audit.query_logs(start_date=start, end_date=end)
        
        # Count events by type
        events_by_type = defaultdict(int)
        for event in events:
            events_by_type[event.event_type.value] += 1
        
        # Calculate compliance rate
        compliance_checks = events_by_type.get(AuditEventType.COMPLIANCE_CHECK.value, 0)
        violations = events_by_type.get(AuditEventType.COMPLIANCE_VIOLATION.value, 0)
        compliance_rate = 1 - (violations / max(compliance_checks, 1))
        
        # Price change metrics
        price_changes = [e for e in events if e.event_type == AuditEventType.PRICE_CHANGE]
        avg_price_change = 0
        if price_changes:
            changes = [abs(e.details.get('change_percentage', 0)) for e in price_changes]
            avg_price_change = sum(changes) / len(changes)
        
        # Processing time from pricing logs
        processing_times = []
        for log in pricing_logs:
            if 'processing_time' in log:
                processing_times.append(log['processing_time'])
        
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Data quality score (simple heuristic)
        data_quality = self._calculate_data_quality_score(events, pricing_logs)
        
        return AuditMetrics(
            total_events=len(events),
            events_by_type=dict(events_by_type),
            compliance_rate=compliance_rate,
            price_change_count=len(price_changes),
            average_price_change=avg_price_change,
            manual_interventions=events_by_type.get(AuditEventType.MANUAL_INTERVENTION.value, 0),
            system_errors=events_by_type.get(AuditEventType.SYSTEM_ERROR.value, 0),
            rollback_count=events_by_type.get(AuditEventType.ROLLBACK.value, 0),
            processing_time_avg=avg_processing_time,
            data_quality_score=data_quality
        )
    
    def _calculate_summary_metrics(self, regulatory_events: List[AuditEvent],
                                 pricing_logs: List[Dict]) -> Dict[str, Any]:
        """Calculate summary metrics from combined data."""
        return {
            'total_events': len(regulatory_events) + len(pricing_logs),
            'unique_products': len(set(e.product_sku for e in regulatory_events if e.product_sku)),
            'markets': list(set(e.market.value for e in regulatory_events if e.market)),
            'data_sources': ['regulatory_audit', 'pricing_audit']
        }
    
    def _create_unified_timeline(self, regulatory_events: List[AuditEvent],
                               pricing_logs: List[Dict]) -> List[Dict]:
        """Create unified timeline of events."""
        timeline = []
        
        # Add regulatory events
        for event in regulatory_events:
            timeline.append({
                'timestamp': event.timestamp.isoformat(),
                'source': 'regulatory',
                'type': event.event_type.value,
                'product_sku': event.product_sku,
                'summary': event.action
            })
        
        # Add pricing logs
        for log in pricing_logs:
            timestamp = log.get('timestamp', '')
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            timeline.append({
                'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                'source': 'pricing',
                'type': 'pricing_decision',
                'product_sku': log.get('product_id'),
                'summary': f"Price change: {log.get('price_change_pct', 0)}%"
            })
        
        # Sort by timestamp
        timeline.sort(key=lambda x: x['timestamp'])
        
        # Return last 100 events
        return timeline[-100:]
    
    def _detect_anomalies(self, regulatory_events: List[AuditEvent],
                         pricing_logs: List[Dict]) -> List[Dict]:
        """Detect anomalies in audit data."""
        anomalies = []
        
        # Check for missing regulatory events for pricing decisions
        pricing_timestamps = set()
        for log in pricing_logs:
            if 'timestamp' in log:
                # Round to minute for matching
                ts = datetime.fromisoformat(log['timestamp']) if isinstance(log['timestamp'], str) else log['timestamp']
                pricing_timestamps.add(ts.replace(second=0, microsecond=0))
        
        regulatory_timestamps = set()
        for event in regulatory_events:
            if event.event_type == AuditEventType.PRICE_DECISION:
                regulatory_timestamps.add(event.timestamp.replace(second=0, microsecond=0))
        
        # Missing regulatory records
        missing = pricing_timestamps - regulatory_timestamps
        for ts in missing:
            anomalies.append({
                'type': 'missing_regulatory_record',
                'timestamp': ts.isoformat(),
                'description': 'Pricing decision without regulatory audit'
            })
        
        # Check for rapid price changes
        sku_changes = defaultdict(list)
        for event in regulatory_events:
            if event.event_type == AuditEventType.PRICE_CHANGE and event.product_sku:
                sku_changes[event.product_sku].append(event.timestamp)
        
        for sku, timestamps in sku_changes.items():
            timestamps.sort()
            for i in range(1, len(timestamps)):
                gap = timestamps[i] - timestamps[i-1]
                if gap < timedelta(minutes=30):
                    anomalies.append({
                        'type': 'rapid_price_change',
                        'product_sku': sku,
                        'timestamp': timestamps[i].isoformat(),
                        'gap_minutes': gap.total_seconds() / 60,
                        'description': f'Price changed twice within {gap.total_seconds()/60:.1f} minutes'
                    })
        
        return anomalies
    
    def _assess_compliance_status(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Assess overall compliance status."""
        violations = [e for e in events if e.event_type == AuditEventType.COMPLIANCE_VIOLATION]
        
        severity_counts = defaultdict(int)
        for violation in violations:
            severity = violation.details.get('severity', 'unknown')
            severity_counts[severity] += 1
        
        return {
            'total_violations': len(violations),
            'by_severity': dict(severity_counts),
            'critical_violations': severity_counts.get('critical', 0),
            'status': 'compliant' if severity_counts.get('critical', 0) == 0 else 'non_compliant'
        }
    
    def _calculate_data_quality_score(self, events: List[AuditEvent],
                                    pricing_logs: List[Dict]) -> float:
        """Calculate data quality score (0-1)."""
        score = 1.0
        
        # Check for missing fields
        missing_fields = 0
        for event in events:
            if not event.product_sku and event.event_type in [
                AuditEventType.PRICE_DECISION,
                AuditEventType.PRICE_CHANGE
            ]:
                missing_fields += 1
        
        if events:
            score -= (missing_fields / len(events)) * 0.3
        
        # Check for data consistency
        if len(pricing_logs) > 0:
            logs_with_all_fields = sum(
                1 for log in pricing_logs
                if all(key in log for key in ['product_id', 'timestamp', 'final_price'])
            )
            completeness = logs_with_all_fields / len(pricing_logs)
            score *= completeness
        
        return max(0, min(1, score))
    
    def _find_matching_log(self, event: AuditEvent, logs: List[Dict]) -> Optional[Dict]:
        """Find matching pricing log for regulatory event."""
        event_time = event.timestamp
        
        for log in logs:
            log_time = log.get('timestamp')
            if isinstance(log_time, str):
                log_time = datetime.fromisoformat(log_time)
            
            # Match within 1 minute window
            if abs((event_time - log_time).total_seconds()) < 60:
                return log
        
        return None
    
    def _generate_reconciliation_recommendations(self, discrepancies: List[Dict]) -> List[str]:
        """Generate recommendations based on discrepancies."""
        recommendations = []
        
        if not discrepancies:
            recommendations.append("Audit trails are fully reconciled")
            return recommendations
        
        # Count discrepancy types
        type_counts = defaultdict(int)
        for disc in discrepancies:
            type_counts[disc['type']] += 1
        
        if type_counts.get('count_mismatch', 0) > 0:
            recommendations.append(
                f"Investigate {type_counts['count_mismatch']} products with mismatched audit counts"
            )
        
        if type_counts.get('missing_pricing_log', 0) > 0:
            recommendations.append(
                f"Review logging configuration - {type_counts['missing_pricing_log']} pricing logs missing"
            )
        
        if len(discrepancies) > 10:
            recommendations.append(
                "High number of discrepancies detected - consider system audit and log synchronization"
            )
        
        return recommendations
    
    def _calculate_trends(self) -> Dict[str, List[Dict]]:
        """Calculate audit trends."""
        trends = {}
        
        # Last 7 days trend
        daily_metrics = []
        for i in range(7):
            date = datetime.now() - timedelta(days=i)
            metrics = self.aggregate_metrics('daily', date)
            daily_metrics.append({
                'date': date.strftime('%Y-%m-%d'),
                'total_events': metrics.total_events,
                'compliance_rate': metrics.compliance_rate,
                'price_changes': metrics.price_change_count
            })
        
        trends['daily'] = list(reversed(daily_metrics))
        
        return trends
    
    def _get_audit_alerts(self) -> List[Dict]:
        """Get current audit-related alerts."""
        alerts = []
        
        # Check recent metrics
        current_metrics = self.aggregate_metrics('hourly', datetime.now())
        
        if current_metrics.compliance_rate < 0.95:
            alerts.append({
                'type': 'low_compliance_rate',
                'severity': 'high',
                'message': f'Compliance rate {current_metrics.compliance_rate:.1%} below threshold'
            })
        
        if current_metrics.system_errors > 5:
            alerts.append({
                'type': 'high_error_rate',
                'severity': 'critical',
                'message': f'{current_metrics.system_errors} system errors in last hour'
            })
        
        return alerts
    
    def _check_system_health(self) -> Dict[str, Any]:
        """Check overall system health based on audit data."""
        hourly = self.aggregate_metrics('hourly', datetime.now())
        daily = self.aggregate_metrics('daily', datetime.now())
        
        health_score = 100
        issues = []
        
        # Check error rate
        if hourly.system_errors > 0:
            health_score -= min(20, hourly.system_errors * 5)
            issues.append(f"{hourly.system_errors} errors in last hour")
        
        # Check compliance
        if daily.compliance_rate < 0.95:
            health_score -= 20
            issues.append(f"Compliance rate {daily.compliance_rate:.1%}")
        
        # Check rollbacks
        if daily.rollback_count > 5:
            health_score -= 15
            issues.append(f"{daily.rollback_count} rollbacks today")
        
        return {
            'score': max(0, health_score),
            'status': 'healthy' if health_score >= 80 else 'degraded' if health_score >= 60 else 'unhealthy',
            'issues': issues
        }
    
    def _get_cached_metrics(self, period: str, date: datetime) -> Optional[AuditMetrics]:
        """Get cached metrics if available."""
        cache_key = f"{period}_{date.strftime('%Y%m%d_%H')}"
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            # Check if cache is still valid
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            
            # Cache validity depends on period
            max_age = {
                'hourly': timedelta(minutes=5),
                'daily': timedelta(hours=1),
                'weekly': timedelta(hours=6),
                'monthly': timedelta(days=1)
            }
            
            if cache_age < max_age.get(period, timedelta(hours=1)):
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    return self._dict_to_metrics(data)
        
        return None
    
    def _cache_metrics(self, period: str, date: datetime, metrics: AuditMetrics):
        """Cache metrics to disk."""
        cache_key = f"{period}_{date.strftime('%Y%m%d_%H')}"
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        with open(cache_file, 'w') as f:
            json.dump(self._metrics_to_dict(metrics), f)
    
    def _metrics_to_dict(self, metrics: AuditMetrics) -> Dict:
        """Convert metrics to dictionary."""
        return {
            'total_events': metrics.total_events,
            'events_by_type': metrics.events_by_type,
            'compliance_rate': metrics.compliance_rate,
            'price_change_count': metrics.price_change_count,
            'average_price_change': metrics.average_price_change,
            'manual_interventions': metrics.manual_interventions,
            'system_errors': metrics.system_errors,
            'rollback_count': metrics.rollback_count,
            'processing_time_avg': metrics.processing_time_avg,
            'data_quality_score': metrics.data_quality_score
        }
    
    def _dict_to_metrics(self, data: Dict) -> AuditMetrics:
        """Convert dictionary to metrics."""
        return AuditMetrics(**data)