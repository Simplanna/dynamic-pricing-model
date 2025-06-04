"""Rollback management for pricing changes."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple
import logging
from enum import Enum
import json
from pathlib import Path

from ..core.models import Product, PricingDecision, Market
from ..utils.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class RollbackState(Enum):
    """State of a rollback operation."""
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class RollbackReason(Enum):
    """Reasons for rollback."""
    COMPLIANCE_VIOLATION = "compliance_violation"
    SAFETY_LIMIT = "safety_limit"
    CANARY_FAILURE = "canary_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MANUAL_REQUEST = "manual_request"
    SYSTEM_ERROR = "system_error"


@dataclass
class PriceSnapshot:
    """Snapshot of product prices at a point in time."""
    timestamp: datetime
    prices: Dict[str, Decimal]  # SKU -> price
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class RollbackPlan:
    """Plan for executing a rollback."""
    plan_id: str
    reason: RollbackReason
    affected_products: List[Product]
    target_snapshot: PriceSnapshot
    current_prices: Dict[str, Decimal]
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    state: RollbackState = RollbackState.READY
    rollback_decisions: List[PricingDecision] = field(default_factory=list)
    results: Dict[str, bool] = field(default_factory=dict)  # SKU -> success


class RollbackManager:
    """Manages price rollbacks and recovery operations."""
    
    def __init__(self, snapshot_dir: Optional[Path] = None,
                 audit_logger: Optional[AuditLogger] = None):
        self.snapshot_dir = snapshot_dir or Path("./price_snapshots")
        self.snapshot_dir.mkdir(exist_ok=True)
        self.audit_logger = audit_logger or AuditLogger()
        
        # State management
        self.snapshots: List[PriceSnapshot] = []
        self.rollback_plans: Dict[str, RollbackPlan] = {}
        self._plan_counter = 0
        
        # Configuration
        self.max_snapshots = 100
        self.snapshot_retention_days = 30
        self.auto_snapshot_interval = timedelta(hours=1)
        self._last_auto_snapshot = datetime.now()
        
        # Load existing snapshots
        self._load_snapshots()
        
        logger.info(f"Rollback manager initialized with {len(self.snapshots)} snapshots")
    
    def create_snapshot(self, products: List[Product], metadata: Optional[Dict] = None) -> PriceSnapshot:
        """Create a price snapshot."""
        prices = {p.sku: p.current_price for p in products}
        
        snapshot = PriceSnapshot(
            timestamp=datetime.now(),
            prices=prices,
            metadata=metadata or {}
        )
        
        # Add to memory
        self.snapshots.append(snapshot)
        
        # Persist to disk
        self._save_snapshot(snapshot)
        
        # Cleanup old snapshots
        self._cleanup_old_snapshots()
        
        logger.info(f"Created price snapshot with {len(prices)} products")
        
        return snapshot
    
    def auto_snapshot(self, products: List[Product]) -> Optional[PriceSnapshot]:
        """Create automatic snapshot if interval has passed."""
        now = datetime.now()
        
        if now - self._last_auto_snapshot >= self.auto_snapshot_interval:
            snapshot = self.create_snapshot(
                products,
                metadata={
                    'type': 'auto',
                    'trigger': 'interval'
                }
            )
            self._last_auto_snapshot = now
            return snapshot
        
        return None
    
    def create_rollback_plan(self, products: List[Product], reason: RollbackReason,
                           target_time: Optional[datetime] = None,
                           hours_ago: Optional[int] = None) -> RollbackPlan:
        """Create a rollback plan to previous prices."""
        
        # Find target snapshot
        if target_time:
            snapshot = self._find_snapshot_by_time(target_time)
        elif hours_ago:
            target_time = datetime.now() - timedelta(hours=hours_ago)
            snapshot = self._find_snapshot_by_time(target_time)
        else:
            # Use most recent snapshot before current
            snapshot = self._get_previous_snapshot()
        
        if not snapshot:
            raise ValueError("No suitable snapshot found for rollback")
        
        # Create plan
        self._plan_counter += 1
        plan_id = f"rollback_{self._plan_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get current prices
        current_prices = {p.sku: p.current_price for p in products}
        
        # Filter to only products that need rollback
        affected_products = [
            p for p in products
            if p.sku in snapshot.prices and p.current_price != snapshot.prices[p.sku]
        ]
        
        plan = RollbackPlan(
            plan_id=plan_id,
            reason=reason,
            affected_products=affected_products,
            target_snapshot=snapshot,
            current_prices=current_prices
        )
        
        # Generate rollback decisions
        plan.rollback_decisions = self._generate_rollback_decisions(plan)
        
        # Store plan
        self.rollback_plans[plan_id] = plan
        
        # Log plan creation
        self._log_plan_creation(plan)
        
        logger.info(f"Created rollback plan {plan_id} affecting {len(affected_products)} products")
        
        return plan
    
    def execute_rollback(self, plan_id: str, batch_size: Optional[int] = None) -> RollbackPlan:
        """Execute a rollback plan."""
        plan = self.rollback_plans.get(plan_id)
        if not plan:
            raise ValueError(f"Rollback plan {plan_id} not found")
        
        if plan.state != RollbackState.READY:
            raise ValueError(f"Plan {plan_id} is not ready for execution (state: {plan.state})")
        
        # Update state
        plan.state = RollbackState.IN_PROGRESS
        plan.executed_at = datetime.now()
        
        # Log execution start
        self.audit_logger.log_rollback_start(
            plan_id=plan_id,
            reason=plan.reason.value,
            affected_count=len(plan.affected_products),
            target_timestamp=plan.target_snapshot.timestamp
        )
        
        # Execute in batches if specified
        if batch_size:
            success_count = 0
            for i in range(0, len(plan.rollback_decisions), batch_size):
                batch = plan.rollback_decisions[i:i + batch_size]
                batch_results = self._execute_batch(batch, plan)
                
                # Update results
                for sku, success in batch_results.items():
                    plan.results[sku] = success
                    if success:
                        success_count += 1
                
                # Log batch progress
                logger.info(f"Rollback batch {i//batch_size + 1}: "
                          f"{sum(batch_results.values())}/{len(batch_results)} successful")
        else:
            # Execute all at once
            results = self._execute_batch(plan.rollback_decisions, plan)
            plan.results = results
            success_count = sum(results.values())
        
        # Determine final state
        total_products = len(plan.affected_products)
        if success_count == total_products:
            plan.state = RollbackState.COMPLETED
        elif success_count == 0:
            plan.state = RollbackState.FAILED
        else:
            plan.state = RollbackState.PARTIAL
        
        # Log completion
        self._log_plan_completion(plan, success_count)
        
        # Create post-rollback snapshot
        if plan.state in [RollbackState.COMPLETED, RollbackState.PARTIAL]:
            self.create_snapshot(
                plan.affected_products,
                metadata={
                    'type': 'post_rollback',
                    'rollback_plan_id': plan_id,
                    'rollback_reason': plan.reason.value
                }
            )
        
        return plan
    
    def get_rollback_history(self, hours: int = 24) -> List[RollbackPlan]:
        """Get recent rollback history."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent_plans = [
            plan for plan in self.rollback_plans.values()
            if plan.created_at > cutoff
        ]
        
        return sorted(recent_plans, key=lambda p: p.created_at, reverse=True)
    
    def validate_rollback_safety(self, plan: RollbackPlan) -> Tuple[bool, List[str]]:
        """Validate if rollback is safe to execute."""
        issues = []
        
        # Check age of target snapshot
        snapshot_age = datetime.now() - plan.target_snapshot.timestamp
        if snapshot_age > timedelta(days=7):
            issues.append(f"Target snapshot is {snapshot_age.days} days old")
        
        # Check for large price changes
        for product in plan.affected_products:
            current = plan.current_prices.get(product.sku, Decimal('0'))
            target = plan.target_snapshot.prices.get(product.sku, Decimal('0'))
            
            if current > 0 and target > 0:
                change_pct = abs((target - current) / current)
                if change_pct > Decimal('0.25'):  # 25%
                    issues.append(f"{product.sku}: Large price change {change_pct:.1%}")
        
        # Check for recent rollbacks
        recent_rollbacks = [
            p for p in self.rollback_plans.values()
            if p.state == RollbackState.COMPLETED and
            p.executed_at and
            datetime.now() - p.executed_at < timedelta(hours=24)
        ]
        
        if len(recent_rollbacks) > 2:
            issues.append(f"Multiple rollbacks ({len(recent_rollbacks)}) in last 24 hours")
        
        is_safe = len(issues) == 0
        return is_safe, issues
    
    def _generate_rollback_decisions(self, plan: RollbackPlan) -> List[PricingDecision]:
        """Generate pricing decisions for rollback."""
        decisions = []
        
        for product in plan.affected_products:
            target_price = plan.target_snapshot.prices.get(product.sku)
            if not target_price:
                continue
            
            decision = PricingDecision(
                product=product,
                base_price=product.current_price,
                recommended_price=target_price,
                final_price=target_price,
                factors={},
                explanation=f"Rollback: {plan.reason.value}"
            )
            
            decision.add_metadata("rollback_plan_id", plan.plan_id)
            decision.add_metadata("rollback_reason", plan.reason.value)
            decision.add_metadata("target_snapshot_time", plan.target_snapshot.timestamp.isoformat())
            
            decisions.append(decision)
        
        return decisions
    
    def _execute_batch(self, decisions: List[PricingDecision], 
                      plan: RollbackPlan) -> Dict[str, bool]:
        """Execute a batch of rollback decisions."""
        results = {}
        
        for decision in decisions:
            try:
                # In production, this would apply the price change
                # For now, simulating success/failure
                success = True  # Would be actual price update result
                
                if success:
                    # Update product's current price in our records
                    decision.product.current_price = decision.final_price
                
                results[decision.product.sku] = success
                
            except Exception as e:
                logger.error(f"Failed to rollback {decision.product.sku}: {str(e)}")
                results[decision.product.sku] = False
        
        return results
    
    def _find_snapshot_by_time(self, target_time: datetime) -> Optional[PriceSnapshot]:
        """Find the snapshot closest to target time."""
        if not self.snapshots:
            return None
        
        # Find snapshot just before target time
        valid_snapshots = [s for s in self.snapshots if s.timestamp <= target_time]
        
        if not valid_snapshots:
            return self.snapshots[0]  # Return oldest if all are after target
        
        return max(valid_snapshots, key=lambda s: s.timestamp)
    
    def _get_previous_snapshot(self) -> Optional[PriceSnapshot]:
        """Get the most recent snapshot."""
        if not self.snapshots:
            return None
        
        # Skip the very latest (might be current state)
        if len(self.snapshots) > 1:
            return self.snapshots[-2]
        
        return self.snapshots[-1]
    
    def _save_snapshot(self, snapshot: PriceSnapshot):
        """Save snapshot to disk."""
        filename = f"snapshot_{snapshot.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.snapshot_dir / filename
        
        data = {
            'timestamp': snapshot.timestamp.isoformat(),
            'prices': {k: str(v) for k, v in snapshot.prices.items()},
            'metadata': snapshot.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_snapshots(self):
        """Load snapshots from disk."""
        self.snapshots = []
        
        for filepath in sorted(self.snapshot_dir.glob("snapshot_*.json")):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                snapshot = PriceSnapshot(
                    timestamp=datetime.fromisoformat(data['timestamp']),
                    prices={k: Decimal(v) for k, v in data['prices'].items()},
                    metadata=data.get('metadata', {})
                )
                
                self.snapshots.append(snapshot)
                
            except Exception as e:
                logger.error(f"Failed to load snapshot {filepath}: {str(e)}")
        
        # Sort by timestamp
        self.snapshots.sort(key=lambda s: s.timestamp)
    
    def _cleanup_old_snapshots(self):
        """Remove old snapshots beyond retention period."""
        cutoff = datetime.now() - timedelta(days=self.snapshot_retention_days)
        
        # Remove from memory
        self.snapshots = [s for s in self.snapshots if s.timestamp > cutoff]
        
        # Remove from disk
        for filepath in self.snapshot_dir.glob("snapshot_*.json"):
            try:
                # Parse timestamp from filename
                timestamp_str = filepath.stem.replace("snapshot_", "")
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                
                if timestamp < cutoff:
                    filepath.unlink()
                    logger.debug(f"Removed old snapshot: {filepath.name}")
                    
            except Exception as e:
                logger.error(f"Error cleaning up snapshot {filepath}: {str(e)}")
        
        # Limit total snapshots
        if len(self.snapshots) > self.max_snapshots:
            excess = len(self.snapshots) - self.max_snapshots
            self.snapshots = self.snapshots[excess:]
    
    def _log_plan_creation(self, plan: RollbackPlan):
        """Log rollback plan creation."""
        self.audit_logger.log_event(
            event_type="rollback_plan_created",
            details={
                'plan_id': plan.plan_id,
                'reason': plan.reason.value,
                'affected_products': len(plan.affected_products),
                'target_snapshot_time': plan.target_snapshot.timestamp.isoformat(),
                'snapshot_age_hours': (
                    datetime.now() - plan.target_snapshot.timestamp
                ).total_seconds() / 3600
            }
        )
    
    def _log_plan_completion(self, plan: RollbackPlan, success_count: int):
        """Log rollback plan completion."""
        self.audit_logger.log_rollback_complete(
            plan_id=plan.plan_id,
            state=plan.state.value,
            success_count=success_count,
            total_count=len(plan.affected_products),
            duration_seconds=(
                datetime.now() - plan.executed_at
            ).total_seconds() if plan.executed_at else 0
        )