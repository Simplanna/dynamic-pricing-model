"""Alert management system for pricing operations."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Callable, Set, Any
import logging
from enum import Enum
import json
from collections import defaultdict
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from ..utils.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"  # Immediate action required
    HIGH = "high"         # Urgent attention needed
    MEDIUM = "medium"     # Should be addressed soon
    LOW = "low"          # Informational
    INFO = "info"        # FYI only


class AlertCategory(Enum):
    """Categories of alerts."""
    COMPLIANCE = "compliance"
    SAFETY = "safety"
    PERFORMANCE = "performance"
    SYSTEM = "system"
    BUSINESS = "business"
    SECURITY = "security"


@dataclass
class Alert:
    """Represents an alert."""
    alert_id: str
    title: str
    message: str
    severity: AlertSeverity
    category: AlertCategory
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None


@dataclass
class AlertRule:
    """Rule for generating alerts."""
    rule_id: str
    name: str
    condition: Callable[[Dict], bool]
    severity: AlertSeverity
    category: AlertCategory
    message_template: str
    cooldown_minutes: int = 15
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRecipient:
    """Alert recipient configuration."""
    name: str
    email: Optional[str] = None
    webhook_url: Optional[str] = None
    severity_filter: List[AlertSeverity] = field(default_factory=lambda: list(AlertSeverity))
    category_filter: List[AlertCategory] = field(default_factory=lambda: list(AlertCategory))
    active: bool = True


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, audit_logger: Optional[AuditLogger] = None):
        self.audit_logger = audit_logger or AuditLogger()
        
        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_rules: Dict[str, AlertRule] = {}
        self.recipients: List[AlertRecipient] = []
        
        # State tracking
        self._alert_counter = 0
        self._rule_cooldowns: Dict[str, datetime] = {}
        self._alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Configuration
        self.max_active_alerts = 100
        self.history_retention_days = 30
        self.smtp_config = None  # Set if email alerts needed
        
        # Initialize default rules
        self._init_default_rules()
        
        logger.info("Alert manager initialized")
    
    def create_alert(self, title: str, message: str, severity: AlertSeverity,
                    category: AlertCategory, source: str,
                    metadata: Optional[Dict] = None) -> Alert:
        """Create and dispatch a new alert."""
        self._alert_counter += 1
        alert_id = f"alert_{self._alert_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        alert = Alert(
            alert_id=alert_id,
            title=title,
            message=message,
            severity=severity,
            category=category,
            source=source,
            metadata=metadata or {}
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Dispatch notifications
        self._dispatch_alert(alert)
        
        # Log alert
        self._log_alert_created(alert)
        
        # Cleanup if needed
        self._cleanup_alerts()
        
        return alert
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str,
                         notes: Optional[str] = None) -> bool:
        """Acknowledge an alert."""
        alert = self.active_alerts.get(alert_id)
        if not alert:
            return False
        
        alert.acknowledged = True
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.now()
        
        if notes:
            alert.metadata['acknowledgment_notes'] = notes
        
        # Log acknowledgment
        self.audit_logger.log_event(
            event_type="alert_acknowledged",
            details={
                'alert_id': alert_id,
                'acknowledged_by': acknowledged_by,
                'notes': notes
            }
        )
        
        return True
    
    def resolve_alert(self, alert_id: str, resolution_notes: str) -> bool:
        """Resolve an alert."""
        alert = self.active_alerts.get(alert_id)
        if not alert:
            return False
        
        alert.resolved = True
        alert.resolved_at = datetime.now()
        alert.resolution_notes = resolution_notes
        
        # Remove from active alerts
        del self.active_alerts[alert_id]
        
        # Log resolution
        self.audit_logger.log_event(
            event_type="alert_resolved",
            details={
                'alert_id': alert_id,
                'resolution_notes': resolution_notes,
                'duration_minutes': (
                    alert.resolved_at - alert.timestamp
                ).total_seconds() / 60
            }
        )
        
        return True
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def add_recipient(self, recipient: AlertRecipient):
        """Add an alert recipient."""
        self.recipients.append(recipient)
        logger.info(f"Added alert recipient: {recipient.name}")
    
    def add_callback(self, callback: Callable[[Alert], None]):
        """Add a callback for alert notifications."""
        self._alert_callbacks.append(callback)
    
    def check_rules(self, context: Dict[str, Any]) -> List[Alert]:
        """Check all rules against current context."""
        triggered_alerts = []
        
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            # Check cooldown
            if self._is_rule_in_cooldown(rule.rule_id):
                continue
            
            # Check condition
            try:
                if rule.condition(context):
                    # Create alert from rule
                    alert = self.create_alert(
                        title=rule.name,
                        message=self._format_message(rule.message_template, context),
                        severity=rule.severity,
                        category=rule.category,
                        source=f"rule:{rule.rule_id}",
                        metadata={
                            'rule_id': rule.rule_id,
                            'context': context
                        }
                    )
                    
                    triggered_alerts.append(alert)
                    
                    # Set cooldown
                    self._rule_cooldowns[rule.rule_id] = datetime.now()
                    
            except Exception as e:
                logger.error(f"Error checking rule {rule.rule_id}: {str(e)}")
        
        return triggered_alerts
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None,
                         category: Optional[AlertCategory] = None) -> List[Alert]:
        """Get active alerts with optional filters."""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if category:
            alerts = [a for a in alerts if a.category == category]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        stats = {
            'total_active': len(self.active_alerts),
            'total_history': len(self.alert_history),
            'by_severity': defaultdict(int),
            'by_category': defaultdict(int),
            'acknowledged_count': 0,
            'avg_resolution_time': None,
            'recent_24h': 0
        }
        
        # Count by severity and category
        for alert in self.active_alerts.values():
            stats['by_severity'][alert.severity.value] += 1
            stats['by_category'][alert.category.value] += 1
            if alert.acknowledged:
                stats['acknowledged_count'] += 1
        
        # Calculate resolution times
        resolved_alerts = [a for a in self.alert_history if a.resolved]
        if resolved_alerts:
            resolution_times = [
                (a.resolved_at - a.timestamp).total_seconds() / 60
                for a in resolved_alerts
            ]
            stats['avg_resolution_time'] = sum(resolution_times) / len(resolution_times)
        
        # Count recent alerts
        cutoff = datetime.now() - timedelta(hours=24)
        stats['recent_24h'] = sum(1 for a in self.alert_history if a.timestamp > cutoff)
        
        return stats
    
    def _init_default_rules(self):
        """Initialize default alert rules."""
        
        # Compliance violation rule
        self.add_rule(AlertRule(
            rule_id="compliance_critical",
            name="Critical Compliance Violation",
            condition=lambda ctx: ctx.get('compliance_violations', 0) > 0 and 
                                ctx.get('violation_severity') == 'critical',
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.COMPLIANCE,
            message_template="Critical compliance violation detected: {violation_details}",
            cooldown_minutes=5
        ))
        
        # High MAPE rule
        self.add_rule(AlertRule(
            rule_id="high_mape",
            name="High Pricing Error Rate",
            condition=lambda ctx: ctx.get('mape', 0) > 0.05,  # 5%
            severity=AlertSeverity.HIGH,
            category=AlertCategory.PERFORMANCE,
            message_template="MAPE exceeds 5%: current value {mape:.1%}",
            cooldown_minutes=30
        ))
        
        # Multiple rollbacks rule
        self.add_rule(AlertRule(
            rule_id="multiple_rollbacks",
            name="Multiple Rollbacks Detected",
            condition=lambda ctx: ctx.get('rollback_count_24h', 0) > 3,
            severity=AlertSeverity.HIGH,
            category=AlertCategory.SAFETY,
            message_template="Multiple rollbacks in 24h: {rollback_count_24h} rollbacks",
            cooldown_minutes=60
        ))
        
        # Low compliance rate rule
        self.add_rule(AlertRule(
            rule_id="low_compliance_rate",
            name="Low Compliance Rate",
            condition=lambda ctx: ctx.get('compliance_rate', 100) < 95,
            severity=AlertSeverity.MEDIUM,
            category=AlertCategory.COMPLIANCE,
            message_template="Compliance rate below 95%: {compliance_rate:.1f}%",
            cooldown_minutes=15
        ))
        
        # System error rate rule
        self.add_rule(AlertRule(
            rule_id="high_error_rate",
            name="High System Error Rate",
            condition=lambda ctx: ctx.get('error_rate', 0) > 0.01,  # 1%
            severity=AlertSeverity.HIGH,
            category=AlertCategory.SYSTEM,
            message_template="System error rate exceeds 1%: {error_rate:.1%}",
            cooldown_minutes=10
        ))
        
        # Canary test failure rule
        self.add_rule(AlertRule(
            rule_id="canary_failure",
            name="Canary Test Failed",
            condition=lambda ctx: ctx.get('canary_test_failed', False),
            severity=AlertSeverity.MEDIUM,
            category=AlertCategory.SAFETY,
            message_template="Canary test '{canary_test_name}' failed",
            cooldown_minutes=0  # No cooldown for test failures
        ))
    
    def _dispatch_alert(self, alert: Alert):
        """Dispatch alert to recipients."""
        # Filter recipients
        relevant_recipients = [
            r for r in self.recipients
            if r.active and
            alert.severity in r.severity_filter and
            alert.category in r.category_filter
        ]
        
        # Send to each recipient
        for recipient in relevant_recipients:
            try:
                if recipient.email and self.smtp_config:
                    self._send_email_alert(recipient.email, alert)
                
                if recipient.webhook_url:
                    self._send_webhook_alert(recipient.webhook_url, alert)
                    
            except Exception as e:
                logger.error(f"Failed to send alert to {recipient.name}: {str(e)}")
        
        # Call registered callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {str(e)}")
    
    def _send_email_alert(self, email: str, alert: Alert):
        """Send email alert (placeholder - implement with actual SMTP)."""
        # This would use smtp_config to send actual emails
        logger.info(f"Would send email alert to {email}: {alert.title}")
    
    def _send_webhook_alert(self, webhook_url: str, alert: Alert):
        """Send webhook alert (placeholder - implement with actual HTTP)."""
        # This would POST to webhook_url
        alert_data = {
            'alert_id': alert.alert_id,
            'title': alert.title,
            'message': alert.message,
            'severity': alert.severity.value,
            'category': alert.category.value,
            'timestamp': alert.timestamp.isoformat(),
            'metadata': alert.metadata
        }
        
        logger.info(f"Would send webhook alert to {webhook_url}: {alert.title}")
    
    def _is_rule_in_cooldown(self, rule_id: str) -> bool:
        """Check if rule is in cooldown period."""
        last_triggered = self._rule_cooldowns.get(rule_id)
        if not last_triggered:
            return False
        
        rule = self.alert_rules.get(rule_id)
        if not rule:
            return False
        
        cooldown_end = last_triggered + timedelta(minutes=rule.cooldown_minutes)
        return datetime.now() < cooldown_end
    
    def _format_message(self, template: str, context: Dict[str, Any]) -> str:
        """Format message template with context values."""
        try:
            return template.format(**context)
        except Exception as e:
            logger.error(f"Error formatting message: {str(e)}")
            return template
    
    def _cleanup_alerts(self):
        """Clean up old alerts."""
        # Limit active alerts
        if len(self.active_alerts) > self.max_active_alerts:
            # Remove oldest resolved alerts first
            resolved = [
                (aid, a) for aid, a in self.active_alerts.items()
                if a.resolved
            ]
            resolved.sort(key=lambda x: x[1].timestamp)
            
            for aid, _ in resolved[:len(self.active_alerts) - self.max_active_alerts]:
                del self.active_alerts[aid]
        
        # Clean history
        cutoff = datetime.now() - timedelta(days=self.history_retention_days)
        self.alert_history = [
            a for a in self.alert_history
            if a.timestamp > cutoff
        ]
    
    def _log_alert_created(self, alert: Alert):
        """Log alert creation."""
        self.audit_logger.log_alert(
            alert_type=f"{alert.category.value}_{alert.severity.value}",
            severity=alert.severity.value,
            details={
                'alert_id': alert.alert_id,
                'title': alert.title,
                'message': alert.message,
                'source': alert.source,
                'metadata': alert.metadata
            }
        )