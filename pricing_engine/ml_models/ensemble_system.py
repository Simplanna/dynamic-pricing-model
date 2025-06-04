"""
Ensemble Decision System
Combines ML predictions with rule-based factors for interpretable pricing decisions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PricingSignal:
    """Individual pricing signal from a model or rule"""
    source: str
    recommended_price: float
    confidence: float
    reasoning: str
    constraints_respected: bool
    metadata: Dict


class EnsemblePricingSystem:
    """Combine ML and rule-based signals for final pricing decisions"""
    
    def __init__(self):
        self.signal_weights = {
            'rule_based': 0.4,  # Conservative weight for compliance
            'demand_forecast': 0.2,
            'elasticity_optimization': 0.15,
            'competitive_response': 0.15,
            'reinforcement_learning': 0.1
        }
        self.confidence_threshold = 0.7
        self.override_rules = []
        self.decision_history = []
        
    def make_pricing_decision(self, product_id: str, signals: List[PricingSignal], 
                            constraints: Dict) -> Dict:
        """Make final pricing decision combining all signals"""
        
        # Check for rule overrides first
        override = self._check_overrides(product_id, constraints)
        if override:
            return override
        
        # Filter signals by constraint compliance
        valid_signals = [s for s in signals if s.constraints_respected]
        
        if not valid_signals:
            return self._fallback_decision(product_id, constraints)
        
        # Calculate weighted price recommendation
        weighted_price = self._calculate_weighted_price(valid_signals)
        
        # Apply safety bounds
        final_price = self._apply_safety_bounds(weighted_price, constraints)
        
        # Calculate decision confidence
        decision_confidence = self._calculate_decision_confidence(valid_signals)
        
        # Generate explanation
        explanation = self._generate_explanation(valid_signals, final_price)
        
        # Create decision record
        decision = {
            'product_id': product_id,
            'final_price': round(final_price, 2),
            'confidence': decision_confidence,
            'signal_count': len(valid_signals),
            'explanation': explanation,
            'contributing_factors': self._summarize_factors(valid_signals),
            'timestamp': pd.Timestamp.now()
        }
        
        # Store in history
        self.decision_history.append(decision)
        
        return decision
    
    def add_override_rule(self, rule_name: str, condition: callable, action: Dict):
        """Add override rule for specific conditions"""
        
        self.override_rules.append({
            'name': rule_name,
            'condition': condition,
            'action': action,
            'created': pd.Timestamp.now()
        })
    
    def analyze_decision_performance(self, outcomes: pd.DataFrame) -> Dict:
        """Analyze how well ensemble decisions performed"""
        
        if not self.decision_history:
            return {'status': 'no_decisions_to_analyze'}
        
        # Match decisions with outcomes
        performance_data = []
        
        for decision in self.decision_history:
            outcome = outcomes[
                (outcomes['product_id'] == decision['product_id']) &
                (outcomes['timestamp'] >= decision['timestamp'])
            ].iloc[0] if len(outcomes) > 0 else None
            
            if outcome is not None:
                performance_data.append({
                    'decision': decision,
                    'outcome': outcome,
                    'revenue_achieved': outcome.get('revenue', 0),
                    'units_sold': outcome.get('units_sold', 0),
                    'margin': outcome.get('margin', 0)
                })
        
        if not performance_data:
            return {'status': 'no_outcomes_available'}
        
        # Analyze by confidence level
        confidence_analysis = self._analyze_by_confidence(performance_data)
        
        # Analyze by contributing factors
        factor_analysis = self._analyze_by_factors(performance_data)
        
        # Overall performance
        avg_revenue = np.mean([p['revenue_achieved'] for p in performance_data])
        confidence_correlation = self._calculate_confidence_correlation(performance_data)
        
        return {
            'total_decisions': len(performance_data),
            'avg_revenue': avg_revenue,
            'confidence_correlation': confidence_correlation,
            'confidence_analysis': confidence_analysis,
            'factor_analysis': factor_analysis,
            'recommendations': self._generate_weight_recommendations(factor_analysis)
        }
    
    def get_explainable_decision(self, decision_id: str) -> Dict:
        """Get detailed explanation for a specific decision"""
        
        # Find decision in history
        decision = next((d for d in self.decision_history 
                        if str(d['timestamp']) == decision_id), None)
        
        if not decision:
            return {'error': 'Decision not found'}
        
        return {
            'decision_id': decision_id,
            'product_id': decision['product_id'],
            'final_price': decision['final_price'],
            'confidence': decision['confidence'],
            'detailed_explanation': decision['explanation'],
            'factor_breakdown': decision['contributing_factors'],
            'visualization_data': self._prepare_visualization_data(decision)
        }
    
    def _check_overrides(self, product_id: str, constraints: Dict) -> Optional[Dict]:
        """Check if any override rules apply"""
        
        for rule in self.override_rules:
            if rule['condition'](product_id, constraints):
                return {
                    'product_id': product_id,
                    'final_price': rule['action']['price'],
                    'confidence': 1.0,
                    'explanation': f"Override rule '{rule['name']}' applied",
                    'override_applied': True,
                    'timestamp': pd.Timestamp.now()
                }
        
        return None
    
    def _calculate_weighted_price(self, signals: List[PricingSignal]) -> float:
        """Calculate weighted average price from signals"""
        
        weighted_sum = 0
        weight_sum = 0
        
        for signal in signals:
            # Get base weight for signal source
            base_weight = self.signal_weights.get(signal.source, 0.1)
            
            # Adjust weight by confidence
            adjusted_weight = base_weight * signal.confidence
            
            weighted_sum += signal.recommended_price * adjusted_weight
            weight_sum += adjusted_weight
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0
    
    def _apply_safety_bounds(self, price: float, constraints: Dict) -> float:
        """Apply safety bounds to ensure compliance"""
        
        # Apply min/max constraints
        min_price = constraints.get('min_price', 0)
        max_price = constraints.get('max_price', float('inf'))
        
        bounded_price = max(min_price, min(max_price, price))
        
        # Apply maximum change constraint
        current_price = constraints.get('current_price', price)
        max_change = constraints.get('max_change_percent', 0.15)
        
        max_increase = current_price * (1 + max_change)
        max_decrease = current_price * (1 - max_change)
        
        return max(max_decrease, min(max_increase, bounded_price))
    
    def _calculate_decision_confidence(self, signals: List[PricingSignal]) -> float:
        """Calculate overall confidence in decision"""
        
        if not signals:
            return 0.0
        
        # Average confidence weighted by signal importance
        weighted_confidence = 0
        weight_sum = 0
        
        for signal in signals:
            weight = self.signal_weights.get(signal.source, 0.1)
            weighted_confidence += signal.confidence * weight
            weight_sum += weight
        
        avg_confidence = weighted_confidence / weight_sum if weight_sum > 0 else 0
        
        # Penalize if signals disagree significantly
        prices = [s.recommended_price for s in signals]
        price_std = np.std(prices)
        price_mean = np.mean(prices)
        
        disagreement_penalty = min(0.3, price_std / price_mean if price_mean > 0 else 0.3)
        
        return max(0, avg_confidence - disagreement_penalty)
    
    def _generate_explanation(self, signals: List[PricingSignal], final_price: float) -> str:
        """Generate human-readable explanation"""
        
        explanations = []
        
        # Sort signals by influence
        sorted_signals = sorted(signals, 
                               key=lambda s: self.signal_weights.get(s.source, 0.1) * s.confidence, 
                               reverse=True)
        
        # Top factors
        explanations.append(f"Price of ${final_price:.2f} determined by {len(signals)} factors:")
        
        for i, signal in enumerate(sorted_signals[:3]):  # Top 3 factors
            influence = self.signal_weights.get(signal.source, 0.1) * signal.confidence
            explanations.append(f"{i+1}. {signal.source}: {signal.reasoning} (influence: {influence:.1%})")
        
        # Confidence statement
        overall_confidence = self._calculate_decision_confidence(signals)
        if overall_confidence > 0.8:
            explanations.append("High confidence in this pricing decision.")
        elif overall_confidence > 0.6:
            explanations.append("Moderate confidence in this pricing decision.")
        else:
            explanations.append("Lower confidence - consider manual review.")
        
        return " ".join(explanations)
    
    def _summarize_factors(self, signals: List[PricingSignal]) -> Dict:
        """Summarize contributing factors"""
        
        factors = {}
        
        for signal in signals:
            factors[signal.source] = {
                'recommended_price': signal.recommended_price,
                'confidence': signal.confidence,
                'weight': self.signal_weights.get(signal.source, 0.1),
                'influence': self.signal_weights.get(signal.source, 0.1) * signal.confidence
            }
        
        # Sort by influence
        sorted_factors = dict(sorted(factors.items(), 
                                   key=lambda x: x[1]['influence'], 
                                   reverse=True))
        
        return sorted_factors
    
    def _fallback_decision(self, product_id: str, constraints: Dict) -> Dict:
        """Fallback when no valid signals available"""
        
        return {
            'product_id': product_id,
            'final_price': constraints.get('current_price', 0),
            'confidence': 0.3,
            'explanation': 'Fallback to current price - no valid signals available',
            'fallback': True,
            'timestamp': pd.Timestamp.now()
        }
    
    def _analyze_by_confidence(self, performance_data: List[Dict]) -> Dict:
        """Analyze performance by confidence level"""
        
        # Bin by confidence level
        high_conf = [p for p in performance_data if p['decision']['confidence'] > 0.8]
        med_conf = [p for p in performance_data if 0.6 <= p['decision']['confidence'] <= 0.8]
        low_conf = [p for p in performance_data if p['decision']['confidence'] < 0.6]
        
        return {
            'high_confidence': {
                'count': len(high_conf),
                'avg_revenue': np.mean([p['revenue_achieved'] for p in high_conf]) if high_conf else 0
            },
            'medium_confidence': {
                'count': len(med_conf),
                'avg_revenue': np.mean([p['revenue_achieved'] for p in med_conf]) if med_conf else 0
            },
            'low_confidence': {
                'count': len(low_conf),
                'avg_revenue': np.mean([p['revenue_achieved'] for p in low_conf]) if low_conf else 0
            }
        }
    
    def _analyze_by_factors(self, performance_data: List[Dict]) -> Dict:
        """Analyze which factors lead to best performance"""
        
        factor_performance = {}
        
        for source in self.signal_weights.keys():
            # Find decisions where this factor had high influence
            relevant_decisions = []
            
            for p in performance_data:
                factors = p['decision'].get('contributing_factors', {})
                if source in factors and factors[source]['influence'] > 0.1:
                    relevant_decisions.append(p)
            
            if relevant_decisions:
                factor_performance[source] = {
                    'decision_count': len(relevant_decisions),
                    'avg_revenue': np.mean([d['revenue_achieved'] for d in relevant_decisions]),
                    'avg_margin': np.mean([d['margin'] for d in relevant_decisions])
                }
        
        return factor_performance
    
    def _calculate_confidence_correlation(self, performance_data: List[Dict]) -> float:
        """Calculate correlation between confidence and performance"""
        
        if len(performance_data) < 5:
            return 0.0
        
        confidences = [p['decision']['confidence'] for p in performance_data]
        revenues = [p['revenue_achieved'] for p in performance_data]
        
        # Simple correlation coefficient
        return np.corrcoef(confidences, revenues)[0, 1]
    
    def _generate_weight_recommendations(self, factor_analysis: Dict) -> List[str]:
        """Generate recommendations for adjusting weights"""
        
        recommendations = []
        
        # Find best and worst performing factors
        if factor_analysis:
            sorted_factors = sorted(factor_analysis.items(), 
                                  key=lambda x: x[1]['avg_revenue'], 
                                  reverse=True)
            
            best_factor = sorted_factors[0][0]
            worst_factor = sorted_factors[-1][0]
            
            if factor_analysis[best_factor]['avg_revenue'] > factor_analysis[worst_factor]['avg_revenue'] * 1.2:
                recommendations.append(f"Consider increasing weight for {best_factor} - showing strong performance")
                recommendations.append(f"Consider decreasing weight for {worst_factor} - underperforming")
        
        return recommendations
    
    def _prepare_visualization_data(self, decision: Dict) -> Dict:
        """Prepare data for decision visualization"""
        
        factors = decision.get('contributing_factors', {})
        
        return {
            'price_breakdown': [
                {
                    'source': source,
                    'recommended_price': data['recommended_price'],
                    'influence': data['influence']
                }
                for source, data in factors.items()
            ],
            'final_price': decision['final_price'],
            'confidence_score': decision['confidence']
        }


class MLRuleIntegrator:
    """Integrate ML predictions with business rules"""
    
    def __init__(self):
        self.integration_rules = []
        self.ml_trust_scores = {}
        
    def create_integrated_signal(self, ml_prediction: Dict, rule_evaluation: Dict, 
                               product_context: Dict) -> PricingSignal:
        """Create integrated signal combining ML and rules"""
        
        # Determine if ML or rules should dominate
        ml_confidence = ml_prediction.get('confidence', 0.5)
        rule_confidence = rule_evaluation.get('confidence', 0.8)
        
        # Check if ML prediction violates any hard rules
        violations = self._check_rule_violations(ml_prediction, rule_evaluation)
        
        if violations:
            # Rules override ML when violations occur
            return PricingSignal(
                source='integrated_rule_override',
                recommended_price=rule_evaluation['recommended_price'],
                confidence=rule_confidence,
                reasoning=f"ML suggestion violates rules: {violations}",
                constraints_respected=True,
                metadata={'violations': violations}
            )
        
        # Blend ML and rule recommendations
        ml_weight = ml_confidence * self._get_ml_trust_score(product_context)
        rule_weight = rule_confidence * 0.6  # Conservative bias toward rules
        
        total_weight = ml_weight + rule_weight
        blended_price = (
            (ml_prediction['price'] * ml_weight + 
             rule_evaluation['recommended_price'] * rule_weight) / 
            total_weight
        )
        
        return PricingSignal(
            source='integrated_ml_rule',
            recommended_price=blended_price,
            confidence=(ml_confidence + rule_confidence) / 2,
            reasoning=self._generate_integrated_reasoning(ml_prediction, rule_evaluation),
            constraints_respected=True,
            metadata={
                'ml_weight': ml_weight / total_weight,
                'rule_weight': rule_weight / total_weight
            }
        )
    
    def _check_rule_violations(self, ml_prediction: Dict, rule_evaluation: Dict) -> List[str]:
        """Check if ML prediction violates business rules"""
        
        violations = []
        
        # Price bounds
        if ml_prediction['price'] < rule_evaluation.get('min_price', 0):
            violations.append('below_minimum_price')
        if ml_prediction['price'] > rule_evaluation.get('max_price', float('inf')):
            violations.append('above_maximum_price')
        
        # Margin requirements
        if ml_prediction.get('expected_margin', 1) < rule_evaluation.get('min_margin', 0):
            violations.append('insufficient_margin')
        
        # Compliance rules
        if not ml_prediction.get('compliance_checked', True):
            violations.append('compliance_not_verified')
        
        return violations
    
    def _get_ml_trust_score(self, product_context: Dict) -> float:
        """Get trust score for ML predictions based on context"""
        
        category = product_context.get('category', 'unknown')
        
        # Default trust scores by category
        default_scores = {
            'flower': 0.7,  # Good ML performance
            'edibles': 0.6,  # Moderate ML performance
            'concentrates': 0.5,  # Limited data
            'unknown': 0.4
        }
        
        return self.ml_trust_scores.get(category, default_scores.get(category, 0.5))
    
    def _generate_integrated_reasoning(self, ml_prediction: Dict, rule_evaluation: Dict) -> str:
        """Generate reasoning for integrated decision"""
        
        ml_reason = ml_prediction.get('reasoning', 'ML optimization')
        rule_reason = rule_evaluation.get('reasoning', 'Business rules')
        
        return f"Balanced decision: {ml_reason} while respecting {rule_reason}"