"""
Competitive Response Prediction Models
Predict competitor behavior and optimize strategic pricing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


class CompetitiveResponsePredictor:
    """Predict competitor pricing behavior and market dynamics"""
    
    def __init__(self):
        self.response_models = {}
        self.behavior_patterns = {}
        self.market_equilibrium = {}
        self.price_war_threshold = 0.7
        
    def train_competitor_model(self, competitor_id: str, historical_data: pd.DataFrame) -> Dict:
        """Train model to predict competitor responses"""
        
        # Extract competitor behavior patterns
        patterns = self._extract_behavior_patterns(historical_data)
        self.behavior_patterns[competitor_id] = patterns
        
        # Prepare features for response prediction
        X, y = self._prepare_response_features(historical_data)
        
        # Train response time model
        response_time_model = self._train_response_time_model(X, historical_data['response_time'])
        
        # Train response magnitude model
        response_magnitude_model = self._train_response_magnitude_model(X, y)
        
        # Store models
        self.response_models[competitor_id] = {
            'response_time': response_time_model,
            'response_magnitude': response_magnitude_model,
            'patterns': patterns,
            'last_update': pd.Timestamp.now()
        }
        
        return {
            'competitor_id': competitor_id,
            'behavior_type': patterns['primary_strategy'],
            'aggressiveness': patterns['aggressiveness_score'],
            'predictability': patterns['predictability_score']
        }
    
    def predict_competitor_response(self, competitor_id: str, price_action: Dict) -> Dict:
        """Predict how competitor will respond to our price change"""
        
        if competitor_id not in self.response_models:
            return self._default_response_prediction()
        
        models = self.response_models[competitor_id]
        patterns = self.behavior_patterns[competitor_id]
        
        # Prepare features
        features = self._create_action_features(price_action, patterns)
        
        # Predict response time
        response_time = models['response_time']['model'].predict([features])[0]
        
        # Predict response magnitude
        response_magnitude = models['response_magnitude']['model'].predict([features])[0]
        
        # Predict response type
        response_type = self._predict_response_type(price_action, patterns)
        
        # Calculate confidence
        confidence = self._calculate_prediction_confidence(competitor_id, features)
        
        return {
            'competitor_id': competitor_id,
            'expected_response_hours': response_time,
            'expected_price_change': response_magnitude,
            'response_type': response_type,
            'confidence': confidence,
            'recommended_counter_strategy': self._recommend_counter_strategy(response_type, patterns)
        }
    
    def detect_price_war_risk(self, market_state: pd.DataFrame) -> Dict:
        """Detect risk of price war in market"""
        
        # Calculate price volatility
        price_volatility = market_state.groupby('competitor_id')['price'].std()
        
        # Calculate price change frequency
        price_changes = market_state.groupby('competitor_id').apply(
            lambda x: (x['price'].diff() != 0).sum() / len(x)
        )
        
        # Calculate average price level trend
        price_trend = market_state.groupby('date')['price'].mean().pct_change().mean()
        
        # Price war indicators
        indicators = {
            'high_volatility': (price_volatility > price_volatility.quantile(0.75)).sum() / len(price_volatility),
            'frequent_changes': (price_changes > 0.3).sum() / len(price_changes),
            'downward_spiral': price_trend < -0.02,
            'margin_compression': self._calculate_margin_compression(market_state)
        }
        
        # Calculate overall risk
        risk_score = np.mean(list(indicators.values()))
        
        return {
            'price_war_risk': risk_score,
            'risk_level': self._classify_risk_level(risk_score),
            'indicators': indicators,
            'recommended_action': self._recommend_price_war_action(risk_score, indicators)
        }
    
    def predict_market_equilibrium(self, current_prices: Dict, elasticities: Dict) -> Dict:
        """Predict market equilibrium prices"""
        
        # Simple Nash equilibrium approximation
        competitors = list(current_prices.keys())
        n_competitors = len(competitors)
        
        if n_competitors == 0:
            return {}
        
        # Initialize equilibrium prices
        equilibrium_prices = current_prices.copy()
        
        # Iterative adjustment
        max_iterations = 50
        tolerance = 0.01
        
        for iteration in range(max_iterations):
            new_prices = {}
            
            for comp in competitors:
                # Calculate best response given others' prices
                others_avg = np.mean([p for c, p in equilibrium_prices.items() if c != comp])
                elasticity = elasticities.get(comp, -1.0)
                
                # Simplified best response function
                marginal_cost = current_prices[comp] * 0.6  # Assume 40% margin
                best_response = marginal_cost + (others_avg - marginal_cost) / (2 + abs(elasticity))
                
                new_prices[comp] = best_response
            
            # Check convergence
            price_changes = [abs(new_prices[c] - equilibrium_prices[c]) / equilibrium_prices[c] 
                           for c in competitors]
            
            if max(price_changes) < tolerance:
                break
            
            equilibrium_prices = new_prices
        
        self.market_equilibrium = {
            'equilibrium_prices': equilibrium_prices,
            'convergence_iterations': iteration,
            'stability': self._assess_equilibrium_stability(equilibrium_prices, current_prices)
        }
        
        return self.market_equilibrium
    
    def _extract_behavior_patterns(self, historical_data: pd.DataFrame) -> Dict:
        """Extract competitor behavior patterns"""
        
        patterns = {}
        
        # Response aggressiveness
        price_changes = historical_data['competitor_price'].pct_change()
        patterns['aggressiveness_score'] = price_changes.std()
        
        # Response speed
        patterns['avg_response_time'] = historical_data['response_time'].mean()
        patterns['response_consistency'] = 1 / (1 + historical_data['response_time'].std())
        
        # Strategy classification
        if patterns['aggressiveness_score'] > 0.05:
            if patterns['avg_response_time'] < 24:
                patterns['primary_strategy'] = 'aggressive_follower'
            else:
                patterns['primary_strategy'] = 'periodic_adjuster'
        else:
            patterns['primary_strategy'] = 'price_leader'
        
        # Predictability
        patterns['predictability_score'] = patterns['response_consistency']
        
        # Price matching tendency
        price_diffs = historical_data['competitor_price'] - historical_data['our_price']
        patterns['price_matching_tendency'] = (abs(price_diffs) < 0.02).mean()
        
        return patterns
    
    def _prepare_response_features(self, historical_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for response prediction"""
        
        features = []
        targets = []
        
        for i in range(1, len(historical_data)):
            # Features
            feat = [
                historical_data.iloc[i-1]['our_price'],
                historical_data.iloc[i-1]['competitor_price'],
                historical_data.iloc[i-1]['market_share'],
                historical_data.iloc[i-1]['day_of_week'],
                historical_data.iloc[i-1]['inventory_level'],
                historical_data.iloc[i-1]['our_price'] - historical_data.iloc[i-1]['competitor_price']
            ]
            features.append(feat)
            
            # Target (competitor's price change)
            targets.append(historical_data.iloc[i]['competitor_price'] - historical_data.iloc[i-1]['competitor_price'])
        
        return np.array(features), np.array(targets)
    
    def _train_response_time_model(self, X: np.ndarray, response_times: pd.Series) -> Dict:
        """Train model to predict response time"""
        
        # Use RandomForest for response time prediction
        model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        
        # Bin response times into categories
        time_bins = pd.cut(response_times[1:], bins=[0, 6, 24, 72, float('inf')], 
                          labels=['immediate', 'same_day', 'few_days', 'slow'])
        
        model.fit(X, time_bins)
        
        return {
            'model': model,
            'feature_importance': dict(zip(range(X.shape[1]), model.feature_importances_))
        }
    
    def _train_response_magnitude_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train model to predict response magnitude"""
        
        # Use GradientBoosting for response magnitude
        model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X, y)
        
        return {
            'model': model,
            'feature_importance': dict(zip(range(X.shape[1]), model.feature_importances_))
        }
    
    def _create_action_features(self, price_action: Dict, patterns: Dict) -> List[float]:
        """Create features from price action"""
        
        return [
            price_action['current_price'],
            price_action['new_price'],
            price_action.get('market_share', 0.3),
            price_action.get('day_of_week', 3),
            price_action.get('inventory_level', 50),
            price_action['new_price'] - price_action['current_price'],
            patterns['aggressiveness_score'],
            patterns['price_matching_tendency']
        ]
    
    def _predict_response_type(self, price_action: Dict, patterns: Dict) -> str:
        """Predict type of competitive response"""
        
        price_change = (price_action['new_price'] - price_action['current_price']) / price_action['current_price']
        
        if abs(price_change) < 0.02:
            return 'no_response'
        elif patterns['primary_strategy'] == 'aggressive_follower':
            if price_change < 0:
                return 'match_or_beat'
            else:
                return 'partial_follow'
        elif patterns['primary_strategy'] == 'price_leader':
            return 'maintain_premium'
        else:
            return 'delayed_adjustment'
    
    def _calculate_prediction_confidence(self, competitor_id: str, features: List[float]) -> float:
        """Calculate confidence in prediction"""
        
        base_confidence = 0.7
        
        # Adjust based on data recency
        models = self.response_models[competitor_id]
        days_old = (pd.Timestamp.now() - models['last_update']).days
        recency_factor = np.exp(-days_old / 30)
        
        # Adjust based on pattern consistency
        patterns = self.behavior_patterns[competitor_id]
        consistency_factor = patterns['predictability_score']
        
        return base_confidence * recency_factor * consistency_factor
    
    def _recommend_counter_strategy(self, response_type: str, patterns: Dict) -> str:
        """Recommend counter-strategy based on predicted response"""
        
        strategies = {
            'match_or_beat': 'Prepare for margin compression, focus on value differentiation',
            'partial_follow': 'Maintain price leadership, emphasize quality',
            'maintain_premium': 'Consider targeted promotions to capture price-sensitive segment',
            'delayed_adjustment': 'Maximize short-term gains before competitor responds',
            'no_response': 'Monitor for delayed reaction, may indicate different target market'
        }
        
        return strategies.get(response_type, 'Monitor competitor behavior closely')
    
    def _calculate_margin_compression(self, market_state: pd.DataFrame) -> float:
        """Calculate degree of margin compression in market"""
        
        # Compare current prices to historical average
        current_avg = market_state.groupby('date')['price'].mean().iloc[-1]
        historical_avg = market_state['price'].mean()
        
        compression = (historical_avg - current_avg) / historical_avg
        return max(0, compression)
    
    def _classify_risk_level(self, risk_score: float) -> str:
        """Classify price war risk level"""
        
        if risk_score > self.price_war_threshold:
            return 'high'
        elif risk_score > 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _recommend_price_war_action(self, risk_score: float, indicators: Dict) -> str:
        """Recommend action based on price war risk"""
        
        if risk_score > self.price_war_threshold:
            if indicators['margin_compression'] > 0.2:
                return 'Seek price stabilization through signaling or focus on differentiation'
            else:
                return 'Prepare defensive strategy, avoid initiating further price cuts'
        elif risk_score > 0.5:
            return 'Monitor competitor actions closely, maintain pricing discipline'
        else:
            return 'Continue normal pricing strategy with regular monitoring'
    
    def _assess_equilibrium_stability(self, equilibrium_prices: Dict, current_prices: Dict) -> str:
        """Assess stability of market equilibrium"""
        
        # Calculate average deviation from equilibrium
        deviations = [abs(equilibrium_prices[c] - current_prices[c]) / current_prices[c] 
                     for c in current_prices.keys()]
        avg_deviation = np.mean(deviations)
        
        if avg_deviation < 0.05:
            return 'stable'
        elif avg_deviation < 0.15:
            return 'converging'
        else:
            return 'unstable'
    
    def _default_response_prediction(self) -> Dict:
        """Default prediction when no model available"""
        
        return {
            'expected_response_hours': 48,
            'expected_price_change': 0,
            'response_type': 'unknown',
            'confidence': 0.3,
            'recommended_counter_strategy': 'Collect more competitive data before making strategic decisions'
        }