"""Regulatory-compliant audit trail system."""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
import hashlib
import sqlite3
from pathlib import Path
from enum import Enum

from ..core.models import Product, Market, PricingDecision
from ..compliance.compliance_engine import ComplianceCheckResult

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""
    PRICE_DECISION = "price_decision"
    PRICE_CHANGE = "price_change"
    COMPLIANCE_CHECK = "compliance_check"
    COMPLIANCE_VIOLATION = "compliance_violation"
    SAFETY_OVERRIDE = "safety_override"
    ROLLBACK = "rollback"
    CANARY_TEST = "canary_test"
    MANUAL_INTERVENTION = "manual_intervention"
    SYSTEM_ERROR = "system_error"
    CONFIG_CHANGE = "config_change"


@dataclass
class AuditEvent:
    """Immutable audit event record."""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    product_sku: Optional[str]
    market: Optional[Market]
    user_id: Optional[str]
    system_component: str
    action: str
    details: Dict[str, Any]
    compliance_status: Optional[str] = None
    hash_chain: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['timestamp'] = self.timestamp.isoformat()
        if self.market:
            data['market'] = self.market.value
        return data
    
    def calculate_hash(self, previous_hash: str = "") -> str:
        """Calculate cryptographic hash for audit integrity."""
        # Create deterministic string representation
        content = f"{self.event_id}|{self.event_type.value}|{self.timestamp.isoformat()}"
        content += f"|{self.product_sku or ''}|{self.market.value if self.market else ''}"
        content += f"|{self.action}|{json.dumps(self.details, sort_keys=True)}"
        content += f"|{previous_hash}"
        
        # Calculate SHA-256 hash
        return hashlib.sha256(content.encode()).hexdigest()


class RegulatoryAuditTrail:
    """Comprehensive audit trail for regulatory compliance."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("pricing_audit.db")
        self._event_counter = 0
        self._last_hash = ""
        
        # Initialize database
        self._init_database()
        
        # Load last hash for chain continuity
        self._load_last_hash()
        
        logger.info(f"Regulatory audit trail initialized at {self.db_path}")
    
    def log_pricing_decision(self, decision: PricingDecision, 
                           compliance_result: Optional[ComplianceCheckResult] = None,
                           user_id: Optional[str] = None) -> str:
        """Log a pricing decision with full context."""
        details = {
            'product_name': decision.product.name,
            'base_price': float(decision.base_price),
            'recommended_price': float(decision.recommended_price),
            'final_price': float(decision.final_price),
            'factors': decision.factors,
            'explanation': decision.explanation,
            'metadata': decision.metadata
        }
        
        # Add compliance information if available
        if compliance_result:
            details['compliance'] = {
                'is_compliant': compliance_result.is_compliant,
                'violations': len(compliance_result.violations),
                'warnings': len(compliance_result.warnings),
                'tax_calculation': compliance_result.tax_calculation,
                'adjusted_price': float(compliance_result.adjusted_price) if compliance_result.adjusted_price else None
            }
        
        event = self._create_event(
            event_type=AuditEventType.PRICE_DECISION,
            product_sku=decision.product.sku,
            market=decision.product.market,
            user_id=user_id,
            system_component="pricing_engine",
            action="generate_price",
            details=details,
            compliance_status="compliant" if compliance_result and compliance_result.is_compliant else "non_compliant"
        )
        
        return self._store_event(event)
    
    def log_price_change(self, product: Product, old_price: Decimal, 
                        new_price: Decimal, reason: str,
                        approved_by: Optional[str] = None) -> str:
        """Log an actual price change implementation."""
        change_pct = ((new_price - old_price) / old_price * 100) if old_price > 0 else 0
        
        details = {
            'product_name': product.name,
            'old_price': float(old_price),
            'new_price': float(new_price),
            'change_amount': float(new_price - old_price),
            'change_percentage': float(change_pct),
            'reason': reason,
            'approved_by': approved_by,
            'category': product.category.value if product.category else None,
            'cost': float(product.cost) if hasattr(product, 'cost') and product.cost else None
        }
        
        event = self._create_event(
            event_type=AuditEventType.PRICE_CHANGE,
            product_sku=product.sku,
            market=product.market,
            user_id=approved_by,
            system_component="pricing_engine",
            action="apply_price_change",
            details=details
        )
        
        return self._store_event(event)
    
    def log_compliance_violation(self, product: Product, violation_type: str,
                               violation_details: str, severity: str,
                               corrective_action: Optional[str] = None) -> str:
        """Log a compliance violation."""
        details = {
            'violation_type': violation_type,
            'violation_details': violation_details,
            'severity': severity,
            'product_name': product.name,
            'current_price': float(product.current_price),
            'corrective_action': corrective_action,
            'timestamp': datetime.now().isoformat()
        }
        
        event = self._create_event(
            event_type=AuditEventType.COMPLIANCE_VIOLATION,
            product_sku=product.sku,
            market=product.market,
            user_id=None,
            system_component="compliance_engine",
            action="violation_detected",
            details=details,
            compliance_status="violation"
        )
        
        return self._store_event(event)
    
    def log_safety_override(self, product_sku: str, override_reason: str,
                          authorized_by: str, duration_hours: int) -> str:
        """Log a safety control override."""
        details = {
            'override_reason': override_reason,
            'authorized_by': authorized_by,
            'duration_hours': duration_hours,
            'start_time': datetime.now().isoformat(),
            'end_time': (datetime.now() + timedelta(hours=duration_hours)).isoformat()
        }
        
        event = self._create_event(
            event_type=AuditEventType.SAFETY_OVERRIDE,
            product_sku=product_sku,
            market=None,
            user_id=authorized_by,
            system_component="safety_controller",
            action="enable_override",
            details=details
        )
        
        return self._store_event(event)
    
    def log_rollback(self, rollback_plan_id: str, reason: str,
                    affected_products: List[str], success_rate: float) -> str:
        """Log a price rollback operation."""
        details = {
            'rollback_plan_id': rollback_plan_id,
            'reason': reason,
            'affected_products_count': len(affected_products),
            'affected_skus': affected_products[:10],  # First 10 for sample
            'success_rate': success_rate,
            'timestamp': datetime.now().isoformat()
        }
        
        event = self._create_event(
            event_type=AuditEventType.ROLLBACK,
            product_sku=None,
            market=None,
            user_id=None,
            system_component="rollback_manager",
            action="execute_rollback",
            details=details
        )
        
        return self._store_event(event)
    
    def log_manual_intervention(self, product_sku: str, intervention_type: str,
                              details: Dict, user_id: str) -> str:
        """Log manual price intervention."""
        event_details = {
            'intervention_type': intervention_type,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            **details
        }
        
        event = self._create_event(
            event_type=AuditEventType.MANUAL_INTERVENTION,
            product_sku=product_sku,
            market=None,
            user_id=user_id,
            system_component="manual_override",
            action=intervention_type,
            details=event_details
        )
        
        return self._store_event(event)
    
    def query_events(self, start_date: datetime, end_date: datetime,
                    event_types: Optional[List[AuditEventType]] = None,
                    product_sku: Optional[str] = None,
                    market: Optional[Market] = None) -> List[AuditEvent]:
        """Query audit events with filters."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Build query
        query = """
            SELECT * FROM audit_events
            WHERE timestamp >= ? AND timestamp <= ?
        """
        params = [start_date.isoformat(), end_date.isoformat()]
        
        if event_types:
            placeholders = ','.join(['?' for _ in event_types])
            query += f" AND event_type IN ({placeholders})"
            params.extend([et.value for et in event_types])
        
        if product_sku:
            query += " AND product_sku = ?"
            params.append(product_sku)
        
        if market:
            query += " AND market = ?"
            params.append(market.value)
        
        query += " ORDER BY timestamp ASC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Convert to AuditEvent objects
        events = []
        for row in rows:
            event = self._row_to_event(row)
            events.append(event)
        
        conn.close()
        return events
    
    def verify_audit_integrity(self, start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> Tuple[bool, List[str]]:
        """Verify the cryptographic integrity of the audit trail."""
        issues = []
        
        # Get all events in date range
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        events = self.query_events(start_date, end_date)
        
        if not events:
            return True, []
        
        # Verify hash chain
        previous_hash = ""
        for i, event in enumerate(events):
            calculated_hash = event.calculate_hash(previous_hash)
            
            if event.hash_chain != calculated_hash:
                issues.append(
                    f"Hash mismatch at event {event.event_id}: "
                    f"expected {calculated_hash}, got {event.hash_chain}"
                )
            
            previous_hash = event.hash_chain
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def export_for_regulator(self, start_date: datetime, end_date: datetime,
                           output_path: Path) -> Dict[str, Any]:
        """Export audit data in regulator-required format."""
        # Query relevant events
        events = self.query_events(start_date, end_date)
        
        # Organize by type
        by_type = {}
        for event in events:
            event_type = event.event_type.value
            if event_type not in by_type:
                by_type[event_type] = []
            by_type[event_type].append(event.to_dict())
        
        # Calculate summary statistics
        summary = {
            'period_start': start_date.isoformat(),
            'period_end': end_date.isoformat(),
            'total_events': len(events),
            'events_by_type': {k: len(v) for k, v in by_type.items()},
            'compliance_violations': len([e for e in events if e.compliance_status == 'violation']),
            'manual_interventions': len([e for e in events if e.event_type == AuditEventType.MANUAL_INTERVENTION]),
            'rollbacks': len([e for e in events if e.event_type == AuditEventType.ROLLBACK])
        }
        
        # Create export package
        export_data = {
            'export_metadata': {
                'generated_at': datetime.now().isoformat(),
                'system_version': '1.0',
                'export_format': 'regulatory_v1'
            },
            'summary': summary,
            'events': by_type,
            'integrity_verification': {
                'hash_algorithm': 'SHA-256',
                'last_event_hash': events[-1].hash_chain if events else None
            }
        }
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return summary
    
    def _init_database(self):
        """Initialize SQLite database for audit storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                product_sku TEXT,
                market TEXT,
                user_id TEXT,
                system_component TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT NOT NULL,
                compliance_status TEXT,
                hash_chain TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_product_sku ON audit_events(product_sku)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)")
        
        conn.commit()
        conn.close()
    
    def _create_event(self, **kwargs) -> AuditEvent:
        """Create a new audit event."""
        self._event_counter += 1
        event_id = f"EVT_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._event_counter:06d}"
        
        return AuditEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            **kwargs
        )
    
    def _store_event(self, event: AuditEvent) -> str:
        """Store an event in the database."""
        # Calculate hash with chain
        event.hash_chain = event.calculate_hash(self._last_hash)
        self._last_hash = event.hash_chain
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO audit_events (
                event_id, event_type, timestamp, product_sku, market,
                user_id, system_component, action, details,
                compliance_status, hash_chain
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.event_id,
            event.event_type.value,
            event.timestamp.isoformat(),
            event.product_sku,
            event.market.value if event.market else None,
            event.user_id,
            event.system_component,
            event.action,
            json.dumps(event.details),
            event.compliance_status,
            event.hash_chain
        ))
        
        conn.commit()
        conn.close()
        
        return event.event_id
    
    def _load_last_hash(self):
        """Load the last hash from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT hash_chain FROM audit_events
            ORDER BY timestamp DESC, event_id DESC
            LIMIT 1
        """)
        
        result = cursor.fetchone()
        if result:
            self._last_hash = result[0]
        
        conn.close()
    
    def _row_to_event(self, row: sqlite3.Row) -> AuditEvent:
        """Convert database row to AuditEvent."""
        return AuditEvent(
            event_id=row['event_id'],
            event_type=AuditEventType(row['event_type']),
            timestamp=datetime.fromisoformat(row['timestamp']),
            product_sku=row['product_sku'],
            market=Market(row['market']) if row['market'] else None,
            user_id=row['user_id'],
            system_component=row['system_component'],
            action=row['action'],
            details=json.loads(row['details']),
            compliance_status=row['compliance_status'],
            hash_chain=row['hash_chain']
        )