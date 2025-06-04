"""Real-time monitoring dashboard for pricing operations."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
import logging
from enum import Enum
from collections import defaultdict, deque
import json

from ..core.models import Product, Market, ProductCategory
from ..utils.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics to monitor."""
    # Pricing metrics
    AVG_PRICE_CHANGE = "avg_price_change"
    PRICE_VOLATILITY = "price_volatility"
    MARGIN_HEALTH = "margin_health"
    
    # Compliance metrics
    COMPLIANCE_RATE = "compliance_rate"
    VIOLATION_COUNT = "violation_count"
    TAX_ACCURACY = "tax_accuracy"
    
    # Performance metrics
    MAPE = "mape"
    REVENUE_IMPACT = "revenue_impact"
    CONVERSION_RATE = "conversion_rate"
    
    # Safety metrics
    ROLLBACK_COUNT = "rollback_count"
    CANARY_SUCCESS_RATE = "canary_success_rate"
    SAFETY_VIOLATIONS = "safety_violations"
    
    # System metrics
    PROCESSING_TIME = "processing_time"
    ERROR_RATE = "error_rate"
    API_LATENCY = "api_latency"


@dataclass
class MetricSnapshot:
    """Snapshot of a metric at a point in time."""
    metric_type: MetricType
    value: Decimal
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""
    metric_type: MetricType
    current_value: Decimal
    avg_value: Decimal
    min_value: Decimal
    max_value: Decimal
    trend: str  # 'up', 'down', 'stable'
    change_pct: Decimal
    sample_count: int
    time_window: timedelta


@dataclass
class DashboardConfig:
    """Configuration for monitoring dashboard."""
    update_interval: timedelta = timedelta(minutes=1)
    metric_retention: timedelta = timedelta(hours=24)
    alert_thresholds: Dict[MetricType, Decimal] = field(default_factory=dict)
    trend_window: timedelta = timedelta(hours=1)
    aggregation_intervals: List[timedelta] = field(default_factory=lambda: [
        timedelta(minutes=5),
        timedelta(minutes=15),
        timedelta(hours=1),
        timedelta(hours=24)
    ])


class MonitoringDashboard:
    """Real-time monitoring dashboard for pricing system."""
    
    def __init__(self, config: Optional[DashboardConfig] = None,
                 audit_logger: Optional[AuditLogger] = None):
        self.config = config or DashboardConfig()
        self.audit_logger = audit_logger or AuditLogger()
        
        # Metric storage
        self._metrics: Dict[MetricType, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._aggregated_metrics: Dict[Tuple[MetricType, timedelta], List[MetricSnapshot]] = {}
        
        # State tracking
        self._last_update = datetime.now()
        self._metric_alerts: Dict[MetricType, datetime] = {}
        
        # Initialize default thresholds
        self._init_default_thresholds()
        
        logger.info("Monitoring dashboard initialized")
    
    def record_metric(self, metric_type: MetricType, value: Decimal,
                     metadata: Optional[Dict] = None):
        """Record a metric value."""
        snapshot = MetricSnapshot(
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self._metrics[metric_type].append(snapshot)
        
        # Check thresholds
        self._check_threshold(metric_type, value)
        
        # Cleanup old metrics
        self._cleanup_old_metrics(metric_type)
    
    def get_current_metrics(self) -> Dict[MetricType, MetricSummary]:
        """Get current state of all metrics."""
        summaries = {}
        
        for metric_type in MetricType:
            if metric_type in self._metrics:
                summary = self._calculate_summary(metric_type)
                if summary:
                    summaries[metric_type] = summary
        
        return summaries
    
    def get_metric_history(self, metric_type: MetricType,
                          hours: int = 1) -> List[MetricSnapshot]:
        """Get historical data for a metric."""
        if metric_type not in self._metrics:
            return []
        
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            snapshot for snapshot in self._metrics[metric_type]
            if snapshot.timestamp > cutoff
        ]
    
    def get_aggregated_metrics(self, metric_type: MetricType,
                             interval: timedelta) -> List[Tuple[datetime, Decimal]]:
        """Get aggregated metrics at specified interval."""
        history = self.get_metric_history(metric_type, hours=24)
        
        if not history:
            return []
        
        # Group by interval
        aggregated = []
        current_window_start = history[0].timestamp
        current_values = []
        
        for snapshot in history:
            if snapshot.timestamp >= current_window_start + interval:
                # Calculate aggregate for window
                if current_values:
                    avg_value = sum(current_values) / len(current_values)
                    aggregated.append((current_window_start, avg_value))
                
                # Start new window
                current_window_start = snapshot.timestamp
                current_values = [snapshot.value]
            else:
                current_values.append(snapshot.value)
        
        # Add final window
        if current_values:
            avg_value = sum(current_values) / len(current_values)
            aggregated.append((current_window_start, avg_value))
        
        return aggregated
    
    def get_market_breakdown(self, metric_type: MetricType) -> Dict[Market, MetricSummary]:
        """Get metric breakdown by market."""
        breakdown = {}
        
        for market in Market:
            market_metrics = [
                s for s in self._metrics[metric_type]
                if s.metadata.get('market') == market.value
            ]
            
            if market_metrics:
                # Create temporary metrics dict for calculation
                temp_metrics = defaultdict(deque)
                temp_metrics[metric_type] = deque(market_metrics)
                
                summary = self._calculate_summary(metric_type, temp_metrics)
                if summary:
                    breakdown[market] = summary
        
        return breakdown
    
    def get_category_breakdown(self, metric_type: MetricType) -> Dict[ProductCategory, MetricSummary]:
        """Get metric breakdown by product category."""
        breakdown = {}
        
        for category in ProductCategory:
            category_metrics = [
                s for s in self._metrics[metric_type]
                if s.metadata.get('category') == category.value
            ]
            
            if category_metrics:
                # Create temporary metrics dict for calculation
                temp_metrics = defaultdict(deque)
                temp_metrics[metric_type] = deque(category_metrics)
                
                summary = self._calculate_summary(metric_type, temp_metrics)
                if summary:
                    breakdown[category] = summary
        
        return breakdown
    
    def get_health_score(self) -> Tuple[Decimal, Dict[str, Any]]:
        """Calculate overall system health score (0-100)."""
        scores = {}
        weights = {
            'compliance': Decimal('0.30'),
            'performance': Decimal('0.25'),
            'safety': Decimal('0.25'),
            'system': Decimal('0.20')
        }
        
        # Compliance health
        compliance_rate = self._get_latest_value(MetricType.COMPLIANCE_RATE)
        scores['compliance'] = compliance_rate if compliance_rate else Decimal('100')
        
        # Performance health (inverse of MAPE)
        mape = self._get_latest_value(MetricType.MAPE)
        if mape:
            scores['performance'] = max(Decimal('0'), Decimal('100') - (mape * Decimal('1000')))
        else:
            scores['performance'] = Decimal('100')
        
        # Safety health
        safety_violations = self._get_latest_value(MetricType.SAFETY_VIOLATIONS)
        rollback_count = self._get_latest_value(MetricType.ROLLBACK_COUNT)
        safety_score = Decimal('100')
        if safety_violations:
            safety_score -= safety_violations * Decimal('10')
        if rollback_count:
            safety_score -= rollback_count * Decimal('5')
        scores['safety'] = max(Decimal('0'), safety_score)
        
        # System health
        error_rate = self._get_latest_value(MetricType.ERROR_RATE)
        if error_rate:
            scores['system'] = max(Decimal('0'), Decimal('100') - (error_rate * Decimal('100')))
        else:
            scores['system'] = Decimal('100')
        
        # Calculate weighted score
        total_score = sum(scores[k] * weights[k] for k in scores)
        
        details = {
            'component_scores': {k: float(v) for k, v in scores.items()},
            'weights': {k: float(v) for k, v in weights.items()},
            'timestamp': datetime.now().isoformat()
        }
        
        return total_score, details
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate complete dashboard data for UI."""
        current_metrics = self.get_current_metrics()
        health_score, health_details = self.get_health_score()
        
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'health_score': float(health_score),
            'health_details': health_details,
            'metrics': {},
            'alerts': [],
            'trends': {}
        }
        
        # Add current metrics
        for metric_type, summary in current_metrics.items():
            dashboard_data['metrics'][metric_type.value] = {
                'current': float(summary.current_value),
                'average': float(summary.avg_value),
                'min': float(summary.min_value),
                'max': float(summary.max_value),
                'trend': summary.trend,
                'change_pct': float(summary.change_pct),
                'samples': summary.sample_count
            }
        
        # Add recent alerts
        for metric_type, last_alert in self._metric_alerts.items():
            if datetime.now() - last_alert < timedelta(hours=1):
                dashboard_data['alerts'].append({
                    'metric': metric_type.value,
                    'timestamp': last_alert.isoformat(),
                    'threshold': float(self.config.alert_thresholds.get(metric_type, 0))
                })
        
        # Add trend data for key metrics
        key_metrics = [
            MetricType.AVG_PRICE_CHANGE,
            MetricType.COMPLIANCE_RATE,
            MetricType.MAPE,
            MetricType.REVENUE_IMPACT
        ]
        
        for metric in key_metrics:
            trend_data = self.get_aggregated_metrics(metric, timedelta(minutes=15))
            if trend_data:
                dashboard_data['trends'][metric.value] = [
                    {'time': t.isoformat(), 'value': float(v)}
                    for t, v in trend_data[-20:]  # Last 20 points
                ]
        
        return dashboard_data
    
    def _init_default_thresholds(self):
        """Initialize default alert thresholds."""
        self.config.alert_thresholds.update({
            MetricType.COMPLIANCE_RATE: Decimal('95'),  # Alert if below 95%
            MetricType.MAPE: Decimal('0.05'),  # Alert if above 5%
            MetricType.ERROR_RATE: Decimal('0.01'),  # Alert if above 1%
            MetricType.SAFETY_VIOLATIONS: Decimal('5'),  # Alert if more than 5
            MetricType.ROLLBACK_COUNT: Decimal('3'),  # Alert if more than 3
            MetricType.AVG_PRICE_CHANGE: Decimal('0.15'),  # Alert if above 15%
        })
    
    def _calculate_summary(self, metric_type: MetricType,
                          metrics_dict: Optional[Dict] = None) -> Optional[MetricSummary]:
        """Calculate summary statistics for a metric."""
        if metrics_dict is None:
            metrics_dict = self._metrics
        
        if metric_type not in metrics_dict or not metrics_dict[metric_type]:
            return None
        
        # Get recent values within time window
        cutoff = datetime.now() - self.config.metric_retention
        recent_values = [
            s.value for s in metrics_dict[metric_type]
            if s.timestamp > cutoff
        ]
        
        if not recent_values:
            return None
        
        # Calculate statistics
        current_value = recent_values[-1]
        avg_value = sum(recent_values) / len(recent_values)
        min_value = min(recent_values)
        max_value = max(recent_values)
        
        # Calculate trend
        trend_cutoff = datetime.now() - self.config.trend_window
        trend_values = [
            s.value for s in metrics_dict[metric_type]
            if s.timestamp > trend_cutoff
        ]
        
        if len(trend_values) >= 2:
            first_half_avg = sum(trend_values[:len(trend_values)//2]) / (len(trend_values)//2)
            second_half_avg = sum(trend_values[len(trend_values)//2:]) / (len(trend_values) - len(trend_values)//2)
            
            if second_half_avg > first_half_avg * Decimal('1.05'):
                trend = 'up'
            elif second_half_avg < first_half_avg * Decimal('0.95'):
                trend = 'down'
            else:
                trend = 'stable'
            
            change_pct = (second_half_avg - first_half_avg) / first_half_avg if first_half_avg > 0 else Decimal('0')
        else:
            trend = 'stable'
            change_pct = Decimal('0')
        
        return MetricSummary(
            metric_type=metric_type,
            current_value=current_value,
            avg_value=avg_value,
            min_value=min_value,
            max_value=max_value,
            trend=trend,
            change_pct=change_pct,
            sample_count=len(recent_values),
            time_window=self.config.metric_retention
        )
    
    def _check_threshold(self, metric_type: MetricType, value: Decimal):
        """Check if metric exceeds threshold."""
        threshold = self.config.alert_thresholds.get(metric_type)
        if not threshold:
            return
        
        # Different logic for different metric types
        should_alert = False
        
        if metric_type in [MetricType.COMPLIANCE_RATE]:
            # Alert if below threshold
            should_alert = value < threshold
        else:
            # Alert if above threshold
            should_alert = value > threshold
        
        if should_alert:
            # Check if we've already alerted recently
            last_alert = self._metric_alerts.get(metric_type)
            if not last_alert or datetime.now() - last_alert > timedelta(minutes=15):
                self._metric_alerts[metric_type] = datetime.now()
                
                # Log alert
                self.audit_logger.log_metric_alert(
                    metric_type=metric_type.value,
                    current_value=float(value),
                    threshold=float(threshold),
                    direction='below' if value < threshold else 'above'
                )
    
    def _get_latest_value(self, metric_type: MetricType) -> Optional[Decimal]:
        """Get the latest value for a metric."""
        if metric_type not in self._metrics or not self._metrics[metric_type]:
            return None
        
        return self._metrics[metric_type][-1].value
    
    def _cleanup_old_metrics(self, metric_type: MetricType):
        """Remove metrics older than retention period."""
        if metric_type not in self._metrics:
            return
        
        cutoff = datetime.now() - self.config.metric_retention
        
        # Since we're using deque with maxlen, old items are automatically removed
        # But we can still do explicit cleanup for timestamp-based retention
        current_metrics = list(self._metrics[metric_type])
        self._metrics[metric_type] = deque(
            [s for s in current_metrics if s.timestamp > cutoff],
            maxlen=10000
        )