"""Real-time compliance validation for pricing decisions."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Set, Callable
import logging
from enum import Enum

from .compliance_engine import ComplianceEngine, ComplianceCheckResult
from ..core.models import Product, PricingDecision
from ..utils.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class ValidationMode(Enum):
    """Validation enforcement modes."""
    STRICT = "strict"      # Block non-compliant prices
    WARNING = "warning"    # Allow but warn
    MONITOR = "monitor"    # Log only
    BYPASS = "bypass"      # Emergency bypass


@dataclass
class ValidationConfig:
    """Configuration for compliance validation."""
    mode: ValidationMode = ValidationMode.STRICT
    check_taxes: bool = True
    check_promotions: bool = True
    check_municipal: bool = True
    max_validation_time: float = 1.0  # seconds
    cache_ttl: int = 300  # Cache validation results for 5 minutes
    alert_on_violations: bool = True


class ComplianceValidator:
    """Real-time compliance validation with caching and performance optimization."""
    
    def __init__(self, compliance_engine: ComplianceEngine, 
                 config: Optional[ValidationConfig] = None,
                 alert_callback: Optional[Callable] = None):
        self.engine = compliance_engine
        self.config = config or ValidationConfig()
        self.alert_callback = alert_callback
        self._validation_cache: Dict[str, ComplianceCheckResult] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self.audit_logger = AuditLogger()
        
        # Performance metrics
        self._validation_count = 0
        self._violation_count = 0
        self._cache_hits = 0
        
        logger.info(f"Compliance validator initialized in {self.config.mode.value} mode")
    
    def validate_pricing_decision(self, decision: PricingDecision) -> PricingDecision:
        """Validate and potentially modify a pricing decision."""
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(decision.product, decision.final_price)
            cached_result = self._get_cached_result(cache_key)
            
            if cached_result:
                self._cache_hits += 1
                result = cached_result
            else:
                # Perform validation
                result = self.engine.validate_pricing(
                    product=decision.product,
                    proposed_price=decision.final_price,
                    current_price=decision.base_price,
                    is_promotional=decision.is_promotional
                )
                
                # Cache the result
                self._cache_result(cache_key, result)
            
            # Update metrics
            self._validation_count += 1
            if not result.is_compliant:
                self._violation_count += 1
            
            # Handle based on mode
            return self._handle_validation_result(decision, result)
            
        except Exception as e:
            logger.error(f"Validation error for {decision.product.sku}: {str(e)}")
            
            # In case of error, behavior depends on mode
            if self.config.mode == ValidationMode.STRICT:
                # Fail safe - use base price
                decision.final_price = decision.base_price
                decision.add_metadata("validation_error", str(e))
            
            return decision
        
        finally:
            # Check performance
            duration = (datetime.now() - start_time).total_seconds()
            if duration > self.config.max_validation_time:
                logger.warning(f"Validation took {duration:.2f}s, exceeding {self.config.max_validation_time}s limit")
    
    def batch_validate(self, decisions: List[PricingDecision]) -> List[PricingDecision]:
        """Validate multiple pricing decisions efficiently."""
        validated_decisions = []
        
        # Group by state for efficient validation
        by_market = {}
        for decision in decisions:
            market = decision.product.market
            if market not in by_market:
                by_market[market] = []
            by_market[market].append(decision)
        
        # Validate each group
        for market, market_decisions in by_market.items():
            for decision in market_decisions:
                validated = self.validate_pricing_decision(decision)
                validated_decisions.append(validated)
        
        return validated_decisions
    
    def _handle_validation_result(self, decision: PricingDecision, 
                                result: ComplianceCheckResult) -> PricingDecision:
        """Handle validation result based on mode."""
        
        # Add compliance metadata
        decision.add_metadata("compliance_checked", True)
        decision.add_metadata("compliance_result", result.is_compliant)
        decision.add_metadata("compliance_violations", len(result.violations))
        decision.add_metadata("compliance_warnings", len(result.warnings))
        
        if result.is_compliant:
            # Add tax information
            if result.tax_calculation:
                decision.add_metadata("tax_calculation", result.tax_calculation)
                decision.add_metadata("price_with_tax", result.tax_calculation.get('total_with_tax'))
            return decision
        
        # Handle non-compliance based on mode
        if self.config.mode == ValidationMode.STRICT:
            # Use adjusted price or revert to base
            if result.adjusted_price:
                decision.final_price = result.adjusted_price
                decision.add_metadata("price_adjusted_for_compliance", True)
                decision.add_metadata("original_price", decision.final_price)
            else:
                decision.final_price = decision.base_price
                decision.add_metadata("price_reverted_to_base", True)
            
            # Alert if configured
            if self.config.alert_on_violations and self.alert_callback:
                self._send_alert(decision, result)
        
        elif self.config.mode == ValidationMode.WARNING:
            # Keep price but add strong warnings
            decision.add_metadata("compliance_warnings_present", True)
            decision.add_metadata("violation_details", [
                {"rule": v.rule.rule_id, "details": v.violation_details}
                for v in result.violations
            ])
            
            # Alert on critical violations
            if result.has_critical_violations and self.alert_callback:
                self._send_alert(decision, result)
        
        elif self.config.mode == ValidationMode.MONITOR:
            # Just log, don't modify
            logger.info(f"Compliance violations for {decision.product.sku}: {result.violation_summary}")
        
        # Record the validation
        self.audit_logger.log_compliance_validation(
            product_sku=decision.product.sku,
            mode=self.config.mode.value,
            result=result,
            decision_modified=decision.final_price != decision.recommended_price
        )
        
        return decision
    
    def _get_cache_key(self, product: Product, price: Decimal) -> str:
        """Generate cache key for validation result."""
        return f"{product.market.value}:{product.sku}:{price}:{product.location or 'default'}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[ComplianceCheckResult]:
        """Get cached validation result if still valid."""
        if cache_key not in self._validation_cache:
            return None
        
        # Check if cache is still valid
        cache_time = self._cache_timestamps.get(cache_key)
        if not cache_time:
            return None
        
        age = (datetime.now() - cache_time).total_seconds()
        if age > self.config.cache_ttl:
            # Cache expired
            del self._validation_cache[cache_key]
            del self._cache_timestamps[cache_key]
            return None
        
        return self._validation_cache[cache_key]
    
    def _cache_result(self, cache_key: str, result: ComplianceCheckResult):
        """Cache validation result."""
        self._validation_cache[cache_key] = result
        self._cache_timestamps[cache_key] = datetime.now()
        
        # Limit cache size
        if len(self._validation_cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(self._cache_timestamps.items(), 
                               key=lambda x: x[1])[:100]
            for key, _ in oldest_keys:
                del self._validation_cache[key]
                del self._cache_timestamps[key]
    
    def _send_alert(self, decision: PricingDecision, result: ComplianceCheckResult):
        """Send compliance violation alert."""
        try:
            alert_data = {
                'timestamp': datetime.now(),
                'product_sku': decision.product.sku,
                'product_name': decision.product.name,
                'market': decision.product.market.value,
                'proposed_price': float(decision.final_price),
                'violations': result.violation_summary,
                'critical_violations': [
                    v.violation_details for v in result.violations 
                    if v.rule.severity == 'critical'
                ]
            }
            
            if self.alert_callback:
                self.alert_callback(alert_data)
            
            # Also log as alert
            self.audit_logger.log_alert(
                alert_type="compliance_violation",
                severity="high" if result.has_critical_violations else "medium",
                details=alert_data
            )
            
        except Exception as e:
            logger.error(f"Failed to send compliance alert: {str(e)}")
    
    def get_metrics(self) -> Dict:
        """Get validator performance metrics."""
        return {
            'total_validations': self._validation_count,
            'total_violations': self._violation_count,
            'violation_rate': self._violation_count / max(self._validation_count, 1),
            'cache_hits': self._cache_hits,
            'cache_hit_rate': self._cache_hits / max(self._validation_count, 1),
            'cache_size': len(self._validation_cache),
            'mode': self.config.mode.value
        }
    
    def clear_cache(self):
        """Clear validation cache."""
        self._validation_cache.clear()
        self._cache_timestamps.clear()
        logger.info("Compliance validation cache cleared")
    
    def set_mode(self, mode: ValidationMode):
        """Change validation mode."""
        old_mode = self.config.mode
        self.config.mode = mode
        logger.info(f"Validation mode changed from {old_mode.value} to {mode.value}")
        
        # Clear cache when changing modes
        self.clear_cache()
        
        # Log mode change
        self.audit_logger.log_config_change(
            component="compliance_validator",
            setting="validation_mode",
            old_value=old_mode.value,
            new_value=mode.value
        )