"""
ML Integration with Rule-Based System
Seamlessly integrate ML models with existing rule-based pricing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta

# Import rule-based system
import sys
sys.path.append('..')
from pricing_calculator import PricingCalculator

# Import ML models
from forecasting.demand_forecaster import DemandForecaster
from optimization.price_elasticity import PriceElasticityLearner, DynamicPricingOptimizer
from optimization.competitive_response import CompetitiveResponsePredictor
from optimization.reinforcement_learning import PriceOptimizationRL
from anomaly_detection.anomaly_detector import MarketAnomalyDetector
from ensemble_system import EnsemblePricingSystem, MLRuleIntegrator, PricingSignal


class MLEnhancedPricingCalculator(PricingCalculator):
    """Enhanced pricing calculator with ML capabilities"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize ML models
        self.demand_forecaster = DemandForecaster()
        self.elasticity_learner = PriceElasticityLearner()
        self.competitive_predictor = CompetitiveResponsePredictor()
        self.rl_optimizer = PriceOptimizationRL()
        self.anomaly_detector = MarketAnomalyDetector()
        self.ensemble_system = EnsemblePricingSystem()
        self.ml_integrator = MLRuleIntegrator()
        
        # ML feature flags
        self.ml_features = {
            'demand_forecasting': True,
            'price_optimization': True,
            'competitive_response': True,
            'ab_testing': True,
            'anomaly_detection': True
        }
        
        # Override confidence weights to include ML
        self.ensemble_system.signal_weights = {
            'rule_based': 0.35,  # Reduced but still significant
            'demand_forecast': 0.20,
            'elasticity_optimization': 0.20,
            'competitive_response': 0.15,
            'reinforcement_learning': 0.10
        }
    
    def calculate_price_with_ml(self, product_data: Dict, market_data: Dict) -> Dict:
        """Calculate price using both rules and ML"""
        
        # First get rule-based price
        rule_result = self.calculate_price(
            product_data['product_id'],
            product_data['base_price'],
            product_data['days_in_inventory'],
            product_data['stock_level'],
            product_data.get('category', 'flower'),
            market_data
        )
        
        # Collect ML signals
        ml_signals = []
        
        # 1. Demand Forecast Signal
        if self.ml_features['demand_forecasting']:
            demand_signal = self._get_demand_signal(product_data)
            if demand_signal:
                ml_signals.append(demand_signal)
        
        # 2. Price Optimization Signal
        if self.ml_features['price_optimization']:
            elasticity_signal = self._get_elasticity_signal(product_data, market_data)
            if elasticity_signal:
                ml_signals.append(elasticity_signal)
        
        # 3. Competitive Response Signal
        if self.ml_features['competitive_response']:
            competitive_signal = self._get_competitive_signal(product_data, market_data)
            if competitive_signal:
                ml_signals.append(competitive_signal)
        
        # 4. A/B Testing Signal (if active)
        if self.ml_features['ab_testing']:
            ab_signal = self._get_ab_testing_signal(product_data)
            if ab_signal:
                ml_signals.append(ab_signal)
        
        # Create rule-based signal
        rule_signal = PricingSignal(
            source='rule_based',
            recommended_price=rule_result['final_price'],
            confidence=rule_result['price_confidence'],
            reasoning=f"Rule-based: {rule_result['applied_factors']}",
            constraints_respected=True,
            metadata={'factors': rule_result['factor_breakdown']}
        )
        
        # Combine all signals
        all_signals = [rule_signal] + ml_signals
        
        # Check for anomalies
        if self.ml_features['anomaly_detection']:
            anomalies = self._check_anomalies(product_data, market_data, all_signals)
            if anomalies['alert_level'] in ['critical', 'warning']:
                # Add anomaly warning to result
                rule_result['anomaly_alert'] = anomalies
        
        # Make ensemble decision
        constraints = {
            'min_price': product_data['base_price'] * 0.7,
            'max_price': product_data['base_price'] * 1.5,
            'current_price': product_data.get('current_price', rule_result['final_price']),
            'max_change_percent': 0.15
        }
        
        ensemble_decision = self.ensemble_system.make_pricing_decision(
            product_data['product_id'],
            all_signals,
            constraints
        )
        
        # Merge results
        ml_enhanced_result = {
            **rule_result,
            'ml_enhanced_price': ensemble_decision['final_price'],
            'ensemble_confidence': ensemble_decision['confidence'],
            'ml_signals': len(ml_signals),
            'primary_driver': self._identify_primary_driver(ensemble_decision),
            'ml_metadata': {
                'demand_trend': 'increasing' if any(s.source == 'demand_forecast' for s in ml_signals) else 'stable',
                'price_sensitivity': 'high' if any(s.source == 'elasticity_optimization' for s in ml_signals) else 'normal',
                'competitive_pressure': 'active' if any(s.source == 'competitive_response' for s in ml_signals) else 'low'
            }
        }
        
        return ml_enhanced_result
    
    def _get_demand_signal(self, product_data: Dict) -> Optional[PricingSignal]:
        """Generate pricing signal from demand forecast"""
        
        try:
            # Get 7-day forecast
            forecast = self.demand_forecaster.forecast_demand(
                product_data['product_id'], 
                horizon_days=7
            )
            
            # Calculate trend
            trend = (forecast['forecast'].iloc[-1] - forecast['forecast'].iloc[0]) / forecast['forecast'].iloc[0]
            
            # Price recommendation based on demand
            if trend > 0.1:  # Demand increasing
                price_multiplier = 1.05
                reasoning = f"Demand trending up {trend:.1%} over next week"
            elif trend < -0.1:  # Demand decreasing  
                price_multiplier = 0.95
                reasoning = f"Demand trending down {abs(trend):.1%} over next week"
            else:
                price_multiplier = 1.0
                reasoning = "Stable demand forecast"
            
            recommended_price = product_data['base_price'] * price_multiplier
            
            return PricingSignal(
                source='demand_forecast',
                recommended_price=recommended_price,
                confidence=forecast['confidence'].mean(),
                reasoning=reasoning,
                constraints_respected=True,
                metadata={'trend': trend, 'forecast': forecast.to_dict()}
            )
            
        except Exception:
            return None
    
    def _get_elasticity_signal(self, product_data: Dict, market_data: Dict) -> Optional[PricingSignal]:
        """Generate pricing signal from elasticity optimization"""
        
        try:
            # Optimize price based on elasticity
            optimization = self.elasticity_learner.elasticity_models.get(
                product_data['product_id']
            )
            
            if not optimization:
                return None
            
            optimizer = DynamicPricingOptimizer(self.elasticity_learner)
            
            result = optimizer.optimize_price(
                product_data['product_id'],
                {
                    'current_price': product_data.get('current_price', product_data['base_price']),
                    'min_price': product_data['base_price'] * 0.7,
                    'max_price': product_data['base_price'] * 1.5,
                    'objective': 'revenue'
                }
            )
            
            return PricingSignal(
                source='elasticity_optimization',
                recommended_price=result['recommended_price'],
                confidence=result['confidence'],
                reasoning=f"Elasticity optimization: {result['elasticity']:.2f}, expected lift {result['expected_revenue_lift']:.1%}",
                constraints_respected=True,
                metadata=result
            )
            
        except Exception:
            return None
    
    def _get_competitive_signal(self, product_data: Dict, market_data: Dict) -> Optional[PricingSignal]:
        """Generate pricing signal from competitive analysis"""
        
        try:
            # Analyze competitive landscape
            competitor_prices = market_data.get('competitor_prices', {})
            
            if not competitor_prices:
                return None
            
            avg_competitor_price = np.mean(list(competitor_prices.values()))
            current_price = product_data.get('current_price', product_data['base_price'])
            
            # Predict competitor responses
            for comp_id in competitor_prices:
                response = self.competitive_predictor.predict_competitor_response(
                    comp_id,
                    {
                        'current_price': current_price,
                        'new_price': current_price * 1.05  # Test small increase
                    }
                )
                
                if response['response_type'] == 'match_or_beat':
                    # Conservative pricing to avoid price war
                    recommended_price = current_price * 0.98
                    reasoning = "Competitive pressure detected - conservative pricing"
                    confidence = 0.8
                    break
            else:
                # No aggressive competitors
                if current_price < avg_competitor_price * 0.95:
                    recommended_price = current_price * 1.02
                    reasoning = "Below market - opportunity to increase"
                else:
                    recommended_price = avg_competitor_price * 0.98
                    reasoning = "Competitive positioning maintained"
                confidence = 0.7
            
            return PricingSignal(
                source='competitive_response',
                recommended_price=recommended_price,
                confidence=confidence,
                reasoning=reasoning,
                constraints_respected=True,
                metadata={'avg_competitor_price': avg_competitor_price}
            )
            
        except Exception:
            return None
    
    def _get_ab_testing_signal(self, product_data: Dict) -> Optional[PricingSignal]:
        """Get pricing signal from active A/B test"""
        
        try:
            # Check for active experiment
            if product_data['product_id'] not in self.rl_optimizer.active_experiments:
                return None
            
            experiment = self.rl_optimizer.active_experiments[product_data['product_id']]
            
            if experiment['status'] != 'active':
                return None
            
            # Get next test price
            context = {
                'current_price': product_data.get('current_price', product_data['base_price']),
                'day_of_week': datetime.now().weekday(),
                'inventory_level': product_data['stock_level']
            }
            
            test_result = self.rl_optimizer.get_next_price(
                product_data['product_id'],
                context
            )
            
            if test_result['is_experiment']:
                return PricingSignal(
                    source='reinforcement_learning',
                    recommended_price=test_result['price'],
                    confidence=test_result['confidence'],
                    reasoning=f"A/B test exploration - expected improvement {test_result['expected_improvement']:.1%}",
                    constraints_respected=True,
                    metadata={'experiment': True}
                )
            
        except Exception:
            return None
        
        return None
    
    def _check_anomalies(self, product_data: Dict, market_data: Dict, 
                        signals: List[PricingSignal]) -> Dict:
        """Check for anomalies in pricing signals"""
        
        # Prepare current data for anomaly detection
        current_data = pd.DataFrame([{
            'product_id': product_data['product_id'],
            'price': product_data.get('current_price', product_data['base_price']),
            'units_sold': product_data.get('recent_sales', 0),
            'timestamp': datetime.now(),
            'competitor_price': np.mean(list(market_data.get('competitor_prices', {}).values())) 
                               if market_data.get('competitor_prices') else None
        }])
        
        # Detect anomalies
        anomalies = self.anomaly_detector.detect_anomalies(
            current_data,
            product_data.get('category', 'unknown')
        )
        
        # Check signal anomalies
        signal_prices = [s.recommended_price for s in signals]
        if len(signal_prices) > 2:
            price_std = np.std(signal_prices)
            price_mean = np.mean(signal_prices)
            
            if price_std / price_mean > 0.2:  # High disagreement
                anomalies['signal_disagreement'] = {
                    'severity': 'warning',
                    'detail': f"ML signals show high variance: {price_std/price_mean:.1%}"
                }
        
        return anomalies
    
    def _identify_primary_driver(self, ensemble_decision: Dict) -> str:
        """Identify primary driver of pricing decision"""
        
        factors = ensemble_decision.get('contributing_factors', {})
        
        if not factors:
            return 'rule_based'
        
        # Find factor with highest influence
        primary = max(factors.items(), key=lambda x: x[1]['influence'])
        
        driver_names = {
            'rule_based': 'Business Rules',
            'demand_forecast': 'Demand Trends',
            'elasticity_optimization': 'Price Optimization',
            'competitive_response': 'Competition',
            'reinforcement_learning': 'A/B Testing'
        }
        
        return driver_names.get(primary[0], primary[0])
    
    def train_ml_models(self, historical_data: pd.DataFrame):
        """Train all ML models with historical data"""
        
        print("Training ML models...")
        
        # Train demand forecaster
        for product_id in historical_data['product_id'].unique():
            product_history = historical_data[historical_data['product_id'] == product_id]
            if len(product_history) > 30:
                self.demand_forecaster.train_product_model(product_id, product_history)
        
        # Train elasticity models
        for product_id in historical_data['product_id'].unique():
            product_data = historical_data[historical_data['product_id'] == product_id]
            if len(product_data) > 20:
                self.elasticity_learner.calculate_elasticity(product_id, product_data)
        
        # Train anomaly detector by category
        for category in historical_data['category'].unique():
            category_data = historical_data[historical_data['category'] == category]
            self.anomaly_detector.train_detector(category_data, category)
        
        print("ML model training complete!")
    
    def start_price_experiment(self, product_id: str, current_price: float) -> Dict:
        """Start an A/B price testing experiment"""
        
        # Define price range for testing (±10%)
        price_range = (current_price * 0.9, current_price * 1.1)
        
        # Create experiment
        return self.rl_optimizer.create_experiment(
            product_id,
            price_range,
            n_arms=5
        )