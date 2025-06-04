"""Performance monitoring for pricing operations."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict, deque
import statistics

from ..core.models import Product, Market, ProductCategory
from .monitoring_dashboard import MonitoringDashboard, MetricType
from .alert_manager import AlertManager, AlertSeverity, AlertCategory

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    mape: Decimal  # Mean Absolute Percentage Error
    rmse: Decimal  # Root Mean Square Error
    bias: Decimal  # Average prediction bias
    hit_rate: Decimal  # Percentage of predictions within threshold
    processing_time: Decimal  # Average processing time in seconds
    throughput: Decimal  # Decisions per second
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceThresholds:
    """Thresholds for performance monitoring."""
    max_mape: Decimal = Decimal('0.02')  # 2%
    max_rmse: Decimal = Decimal('5.00')  # $5
    max_bias: Decimal = Decimal('0.01')  # 1%
    min_hit_rate: Decimal = Decimal('0.95')  # 95%
    max_processing_time: Decimal = Decimal('1.0')  # 1 second
    min_throughput: Decimal = Decimal('10')  # 10 decisions/second


class PerformanceMonitor:
    """Monitors and analyzes pricing system performance."""
    
    def __init__(self, monitoring_dashboard: MonitoringDashboard,
                 alert_manager: AlertManager,
                 thresholds: Optional[PerformanceThresholds] = None):
        self.dashboard = monitoring_dashboard
        self.alert_manager = alert_manager
        self.thresholds = thresholds or PerformanceThresholds()
        
        # Performance tracking
        self._predictions: deque = deque(maxlen=10000)  # (predicted, actual, timestamp)
        self._processing_times: deque = deque(maxlen=1000)
        self._decision_counts: defaultdict = defaultdict(int)
        
        # Aggregated metrics
        self._hourly_metrics: Dict[datetime, PerformanceMetrics] = {}
        self._category_metrics: Dict[ProductCategory, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._market_metrics: Dict[Market, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # State
        self._last_calculation = datetime.now()
        self._degradation_detected = False
        
        logger.info("Performance monitor initialized")
    
    def record_prediction(self, product: Product, predicted_price: Decimal,
                         actual_price: Decimal, processing_time: float):
        """Record a pricing prediction for performance tracking."""
        # Store prediction
        self._predictions.append((
            predicted_price,
            actual_price,
            product,
            datetime.now()
        ))
        
        # Store processing time
        self._processing_times.append(processing_time)
        
        # Update decision count
        self._decision_counts[datetime.now().replace(minute=0, second=0, microsecond=0)] += 1
        
        # Track by category and market
        if product.category:
            self._category_metrics[product.category].append((predicted_price, actual_price))
        if product.market:
            self._market_metrics[product.market].append((predicted_price, actual_price))
        
        # Calculate metrics if needed
        if datetime.now() - self._last_calculation > timedelta(minutes=5):
            self.calculate_metrics()
    
    def calculate_metrics(self) -> PerformanceMetrics:
        """Calculate current performance metrics."""
        if not self._predictions:
            return self._create_empty_metrics()
        
        # Get recent predictions (last hour)
        cutoff = datetime.now() - timedelta(hours=1)
        recent_predictions = [
            (pred, actual) for pred, actual, _, timestamp in self._predictions
            if timestamp > cutoff
        ]
        
        if not recent_predictions:
            return self._create_empty_metrics()
        
        # Calculate MAPE
        mape_values = []
        for predicted, actual in recent_predictions:
            if actual > 0:
                error = abs((predicted - actual) / actual)
                mape_values.append(error)
        
        mape = sum(mape_values) / len(mape_values) if mape_values else Decimal('0')
        
        # Calculate RMSE
        squared_errors = [
            (predicted - actual) ** 2
            for predicted, actual in recent_predictions
        ]
        rmse = Decimal(str(statistics.mean(squared_errors) ** 0.5)) if squared_errors else Decimal('0')
        
        # Calculate bias
        biases = [
            (predicted - actual) / actual
            for predicted, actual in recent_predictions
            if actual > 0
        ]
        bias = sum(biases) / len(biases) if biases else Decimal('0')
        
        # Calculate hit rate (within 2% of actual)
        hits = sum(
            1 for predicted, actual in recent_predictions
            if actual > 0 and abs((predicted - actual) / actual) <= Decimal('0.02')
        )
        hit_rate = Decimal(str(hits)) / Decimal(str(len(recent_predictions)))
        
        # Calculate processing metrics
        avg_processing_time = Decimal(str(statistics.mean(self._processing_times))) if self._processing_times else Decimal('0')
        
        # Calculate throughput
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        throughput = Decimal(str(self._decision_counts[current_hour] / 3600))  # per second
        
        metrics = PerformanceMetrics(
            mape=mape,
            rmse=rmse,
            bias=bias,
            hit_rate=hit_rate,
            processing_time=avg_processing_time,
            throughput=throughput
        )
        
        # Store metrics
        self._hourly_metrics[current_hour] = metrics
        self._last_calculation = datetime.now()
        
        # Update dashboard
        self._update_dashboard(metrics)
        
        # Check thresholds
        self._check_performance_thresholds(metrics)
        
        return metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        current_metrics = self.calculate_metrics()
        
        # Calculate trends
        trend_data = self._calculate_trends()
        
        # Get breakdown by category and market
        category_performance = self._calculate_category_performance()
        market_performance = self._calculate_market_performance()
        
        summary = {
            'current_metrics': {
                'mape': float(current_metrics.mape),
                'rmse': float(current_metrics.rmse),
                'bias': float(current_metrics.bias),
                'hit_rate': float(current_metrics.hit_rate),
                'processing_time': float(current_metrics.processing_time),
                'throughput': float(current_metrics.throughput)
            },
            'trends': trend_data,
            'category_breakdown': category_performance,
            'market_breakdown': market_performance,
            'health_status': self._get_health_status(current_metrics),
            'alerts': self._get_performance_alerts(),
            'timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def detect_degradation(self) -> Tuple[bool, List[str]]:
        """Detect performance degradation."""
        issues = []
        
        # Get recent metrics
        recent_hours = sorted(self._hourly_metrics.keys())[-6:]  # Last 6 hours
        if len(recent_hours) < 3:
            return False, []
        
        recent_metrics = [self._hourly_metrics[h] for h in recent_hours]
        
        # Check for increasing MAPE
        mape_trend = [m.mape for m in recent_metrics]
        if all(mape_trend[i] <= mape_trend[i+1] for i in range(len(mape_trend)-1)):
            issues.append(f"MAPE increasing consistently: {mape_trend[-1]:.1%}")
        
        # Check for decreasing hit rate
        hit_rate_trend = [m.hit_rate for m in recent_metrics]
        if all(hit_rate_trend[i] >= hit_rate_trend[i+1] for i in range(len(hit_rate_trend)-1)):
            issues.append(f"Hit rate decreasing: {hit_rate_trend[-1]:.1%}")
        
        # Check for increasing processing time
        time_trend = [m.processing_time for m in recent_metrics]
        avg_time = sum(time_trend) / len(time_trend)
        if time_trend[-1] > avg_time * Decimal('1.5'):
            issues.append(f"Processing time spike: {time_trend[-1]:.2f}s")
        
        degradation_detected = len(issues) > 0
        
        if degradation_detected and not self._degradation_detected:
            # First detection, create alert
            self.alert_manager.create_alert(
                title="Performance Degradation Detected",
                message="; ".join(issues),
                severity=AlertSeverity.HIGH,
                category=AlertCategory.PERFORMANCE,
                source="performance_monitor",
                metadata={'issues': issues}
            )
        
        self._degradation_detected = degradation_detected
        
        return degradation_detected, issues
    
    def _calculate_trends(self) -> Dict[str, List[Dict]]:
        """Calculate performance trends."""
        trends = {}
        
        # Get hourly data for last 24 hours
        cutoff = datetime.now() - timedelta(hours=24)
        hourly_data = [
            (hour, metrics) for hour, metrics in self._hourly_metrics.items()
            if hour > cutoff
        ]
        
        if not hourly_data:
            return trends
        
        # MAPE trend
        trends['mape'] = [
            {'time': hour.isoformat(), 'value': float(metrics.mape)}
            for hour, metrics in hourly_data
        ]
        
        # Hit rate trend
        trends['hit_rate'] = [
            {'time': hour.isoformat(), 'value': float(metrics.hit_rate)}
            for hour, metrics in hourly_data
        ]
        
        # Processing time trend
        trends['processing_time'] = [
            {'time': hour.isoformat(), 'value': float(metrics.processing_time)}
            for hour, metrics in hourly_data
        ]
        
        return trends
    
    def _calculate_category_performance(self) -> Dict[str, Dict[str, float]]:
        """Calculate performance by product category."""
        performance = {}
        
        for category, predictions in self._category_metrics.items():
            if not predictions:
                continue
            
            # Calculate MAPE for category
            mape_values = []
            for predicted, actual in predictions:
                if actual > 0:
                    error = abs((predicted - actual) / actual)
                    mape_values.append(error)
            
            if mape_values:
                category_mape = sum(mape_values) / len(mape_values)
                performance[category.value] = {
                    'mape': float(category_mape),
                    'sample_count': len(predictions)
                }
        
        return performance
    
    def _calculate_market_performance(self) -> Dict[str, Dict[str, float]]:
        """Calculate performance by market."""
        performance = {}
        
        for market, predictions in self._market_metrics.items():
            if not predictions:
                continue
            
            # Calculate metrics for market
            mape_values = []
            squared_errors = []
            
            for predicted, actual in predictions:
                if actual > 0:
                    error = abs((predicted - actual) / actual)
                    mape_values.append(error)
                    squared_errors.append((predicted - actual) ** 2)
            
            if mape_values:
                market_mape = sum(mape_values) / len(mape_values)
                market_rmse = Decimal(str(statistics.mean(squared_errors) ** 0.5))
                
                performance[market.value] = {
                    'mape': float(market_mape),
                    'rmse': float(market_rmse),
                    'sample_count': len(predictions)
                }
        
        return performance
    
    def _update_dashboard(self, metrics: PerformanceMetrics):
        """Update monitoring dashboard with performance metrics."""
        self.dashboard.record_metric(MetricType.MAPE, metrics.mape)
        self.dashboard.record_metric(MetricType.PROCESSING_TIME, metrics.processing_time)
        
        # Calculate and record error rate (based on hit rate)
        error_rate = Decimal('1') - metrics.hit_rate
        self.dashboard.record_metric(MetricType.ERROR_RATE, error_rate)
    
    def _check_performance_thresholds(self, metrics: PerformanceMetrics):
        """Check if performance metrics exceed thresholds."""
        violations = []
        
        if metrics.mape > self.thresholds.max_mape:
            violations.append(f"MAPE {metrics.mape:.1%} exceeds {self.thresholds.max_mape:.1%}")
        
        if metrics.rmse > self.thresholds.max_rmse:
            violations.append(f"RMSE ${metrics.rmse:.2f} exceeds ${self.thresholds.max_rmse:.2f}")
        
        if abs(metrics.bias) > self.thresholds.max_bias:
            violations.append(f"Bias {metrics.bias:.1%} exceeds {self.thresholds.max_bias:.1%}")
        
        if metrics.hit_rate < self.thresholds.min_hit_rate:
            violations.append(f"Hit rate {metrics.hit_rate:.1%} below {self.thresholds.min_hit_rate:.1%}")
        
        if metrics.processing_time > self.thresholds.max_processing_time:
            violations.append(f"Processing time {metrics.processing_time:.2f}s exceeds {self.thresholds.max_processing_time:.1f}s")
        
        if violations:
            # Check alert rules
            context = {
                'mape': float(metrics.mape),
                'rmse': float(metrics.rmse),
                'bias': float(metrics.bias),
                'hit_rate': float(metrics.hit_rate),
                'processing_time': float(metrics.processing_time),
                'violations': violations
            }
            
            self.alert_manager.check_rules(context)
    
    def _get_health_status(self, metrics: PerformanceMetrics) -> str:
        """Determine overall health status."""
        if (metrics.mape <= self.thresholds.max_mape and
            metrics.hit_rate >= self.thresholds.min_hit_rate and
            metrics.processing_time <= self.thresholds.max_processing_time):
            return "healthy"
        elif (metrics.mape <= self.thresholds.max_mape * Decimal('1.5') and
              metrics.hit_rate >= self.thresholds.min_hit_rate * Decimal('0.9')):
            return "warning"
        else:
            return "critical"
    
    def _get_performance_alerts(self) -> List[str]:
        """Get current performance-related alerts."""
        active_alerts = self.alert_manager.get_active_alerts(category=AlertCategory.PERFORMANCE)
        return [
            f"{alert.severity.value}: {alert.title}"
            for alert in active_alerts[:5]  # Top 5 alerts
        ]
    
    def _create_empty_metrics(self) -> PerformanceMetrics:
        """Create empty metrics when no data available."""
        return PerformanceMetrics(
            mape=Decimal('0'),
            rmse=Decimal('0'),
            bias=Decimal('0'),
            hit_rate=Decimal('1'),
            processing_time=Decimal('0'),
            throughput=Decimal('0')
        )