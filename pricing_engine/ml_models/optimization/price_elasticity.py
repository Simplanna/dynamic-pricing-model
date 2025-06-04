"""
Price Elasticity Learning System
Dynamic calculation of price sensitivity by product and customer segment
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class PriceElasticityLearner:
    """Learn and update price elasticity dynamically"""
    
    def __init__(self):
        self.elasticity_models = {}
        self.elasticity_history = {}
        self.segment_elasticities = {}
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        
    def calculate_elasticity(self, product_id: str, price_sales_data: pd.DataFrame) -> Dict:
        """Calculate price elasticity for a product"""
        
        # Prepare data
        data = self._prepare_elasticity_data(price_sales_data)
        
        # Calculate point elasticity
        point_elasticity = self._calculate_point_elasticity(data)
        
        # Calculate arc elasticity for robustness
        arc_elasticity = self._calculate_arc_elasticity(data)
        
        # Fit elasticity curve
        elasticity_model, confidence = self._fit_elasticity_curve(data)
        
        # Store model
        self.elasticity_models[product_id] = {
            'model': elasticity_model,
            'point_elasticity': point_elasticity,
            'arc_elasticity': arc_elasticity,
            'confidence': confidence,
            'last_update': pd.Timestamp.now(),
            'data_points': len(data)
        }
        
        # Classify elasticity
        elasticity_class = self._classify_elasticity(point_elasticity)
        
        return {
            'product_id': product_id,
            'point_elasticity': point_elasticity,
            'arc_elasticity': arc_elasticity,
            'elasticity_class': elasticity_class,
            'confidence': confidence,
            'optimal_price_range': self._calculate_optimal_price_range(elasticity_model, data)
        }
    
    def calculate_segment_elasticity(self, segment: str, segment_data: pd.DataFrame) -> Dict:
        """Calculate elasticity for customer segment"""
        
        segment_elasticities = []
        
        # Group by product and calculate individual elasticities
        for product_id in segment_data['product_id'].unique():
            product_data = segment_data[segment_data['product_id'] == product_id]
            
            if len(product_data) >= 10:  # Minimum data requirement
                elasticity = self._calculate_point_elasticity(product_data)
                segment_elasticities.append(elasticity)
        
        if segment_elasticities:
            avg_elasticity = np.mean(segment_elasticities)
            std_elasticity = np.std(segment_elasticities)
            
            self.segment_elasticities[segment] = {
                'average_elasticity': avg_elasticity,
                'std_dev': std_elasticity,
                'sensitivity_level': self._classify_segment_sensitivity(avg_elasticity),
                'sample_size': len(segment_elasticities)
            }
            
            return self.segment_elasticities[segment]
        
        return None
    
    def update_elasticity_realtime(self, product_id: str, price_change: float, 
                                  quantity_change: float) -> Dict:
        """Update elasticity with new price/quantity observation"""
        
        # Calculate instantaneous elasticity
        instant_elasticity = (quantity_change / price_change) if price_change != 0 else 0
        
        # Initialize history if needed
        if product_id not in self.elasticity_history:
            self.elasticity_history[product_id] = []
        
        # Add to history
        self.elasticity_history[product_id].append({
            'timestamp': pd.Timestamp.now(),
            'price_change': price_change,
            'quantity_change': quantity_change,
            'elasticity': instant_elasticity
        })
        
        # Update model if enough new data
        if len(self.elasticity_history[product_id]) >= 5:
            # Recalculate with exponential weighting for recency
            recent_elasticities = [e['elasticity'] for e in self.elasticity_history[product_id][-20:]]
            weights = np.exp(np.linspace(-1, 0, len(recent_elasticities)))
            weighted_elasticity = np.average(recent_elasticities, weights=weights)
            
            # Update stored elasticity
            if product_id in self.elasticity_models:
                self.elasticity_models[product_id]['point_elasticity'] = weighted_elasticity
                self.elasticity_models[product_id]['last_update'] = pd.Timestamp.now()
        
        return {
            'instant_elasticity': instant_elasticity,
            'updated_elasticity': self.elasticity_models.get(product_id, {}).get('point_elasticity', instant_elasticity),
            'confidence': self._calculate_confidence(product_id)
        }
    
    def get_price_response_curve(self, product_id: str, price_range: Tuple[float, float]) -> pd.DataFrame:
        """Generate price response curve with confidence intervals"""
        
        if product_id not in self.elasticity_models:
            raise ValueError(f"No elasticity model for product {product_id}")
        
        model_info = self.elasticity_models[product_id]
        model = model_info['model']
        
        # Generate price points
        prices = np.linspace(price_range[0], price_range[1], 50)
        
        # Predict quantities
        log_prices = np.log(prices).reshape(-1, 1)
        
        if hasattr(model, 'predict'):
            log_quantities = model.predict(log_prices)
            quantities = np.exp(log_quantities)
        else:
            # Use constant elasticity model
            base_price = (price_range[0] + price_range[1]) / 2
            base_quantity = 100  # Normalized
            elasticity = model_info['point_elasticity']
            quantities = base_quantity * (prices / base_price) ** elasticity
        
        # Calculate revenue
        revenues = prices * quantities
        
        # Calculate confidence intervals
        confidence = model_info['confidence']
        quantity_std = quantities * (1 - confidence) * 0.2
        
        response_curve = pd.DataFrame({
            'price': prices,
            'expected_quantity': quantities,
            'quantity_lower': quantities - 1.96 * quantity_std,
            'quantity_upper': quantities + 1.96 * quantity_std,
            'expected_revenue': revenues,
            'elasticity': model_info['point_elasticity']
        })
        
        return response_curve
    
    def _prepare_elasticity_data(self, price_sales_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare and clean data for elasticity calculation"""
        
        data = price_sales_data.copy()
        
        # Remove outliers
        data = data[(data['price'] > 0) & (data['quantity'] > 0)]
        
        # Calculate log transformations
        data['log_price'] = np.log(data['price'])
        data['log_quantity'] = np.log(data['quantity'])
        
        # Remove extreme outliers (beyond 3 std)
        for col in ['log_price', 'log_quantity']:
            mean = data[col].mean()
            std = data[col].std()
            data = data[np.abs(data[col] - mean) <= 3 * std]
        
        return data
    
    def _calculate_point_elasticity(self, data: pd.DataFrame) -> float:
        """Calculate point elasticity using log-log regression"""
        
        if len(data) < 5:
            return -1.0  # Default elastic assumption
        
        # Log-log regression
        X = data['log_price'].values.reshape(-1, 1)
        y = data['log_quantity'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Elasticity is the coefficient
        elasticity = model.coef_[0]
        
        return elasticity
    
    def _calculate_arc_elasticity(self, data: pd.DataFrame) -> float:
        """Calculate arc elasticity for robustness"""
        
        if len(data) < 2:
            return -1.0
        
        # Sort by price
        data_sorted = data.sort_values('price')
        
        # Calculate between first and last quartile
        q1_idx = int(len(data_sorted) * 0.25)
        q3_idx = int(len(data_sorted) * 0.75)
        
        p1, q1 = data_sorted.iloc[q1_idx][['price', 'quantity']]
        p2, q2 = data_sorted.iloc[q3_idx][['price', 'quantity']]
        
        # Arc elasticity formula
        price_change = (p2 - p1) / ((p1 + p2) / 2)
        quantity_change = (q2 - q1) / ((q1 + q2) / 2)
        
        arc_elasticity = quantity_change / price_change if price_change != 0 else -1.0
        
        return arc_elasticity
    
    def _fit_elasticity_curve(self, data: pd.DataFrame) -> Tuple[object, float]:
        """Fit non-linear elasticity curve"""
        
        X = data['log_price'].values.reshape(-1, 1)
        y = data['log_quantity'].values
        
        # Try polynomial features for non-constant elasticity
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        
        # Use Ridge regression for stability
        model = Ridge(alpha=0.1)
        model.fit(X_poly, y)
        
        # Calculate R-squared for confidence
        y_pred = model.predict(X_poly)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Convert R-squared to confidence (0-1)
        confidence = max(0, min(1, r_squared))
        
        return model, confidence
    
    def _classify_elasticity(self, elasticity: float) -> str:
        """Classify elasticity into categories"""
        
        abs_elasticity = abs(elasticity)
        
        if abs_elasticity > 1.5:
            return 'highly_elastic'
        elif abs_elasticity > 1.0:
            return 'elastic'
        elif abs_elasticity > 0.5:
            return 'unit_elastic'
        elif abs_elasticity > 0.2:
            return 'inelastic'
        else:
            return 'highly_inelastic'
    
    def _classify_segment_sensitivity(self, avg_elasticity: float) -> str:
        """Classify customer segment price sensitivity"""
        
        abs_elasticity = abs(avg_elasticity)
        
        if abs_elasticity > 1.2:
            return 'price_sensitive'
        elif abs_elasticity > 0.8:
            return 'moderately_sensitive'
        else:
            return 'price_insensitive'
    
    def _calculate_optimal_price_range(self, model: object, data: pd.DataFrame) -> Dict:
        """Calculate revenue-maximizing price range"""
        
        # Get current price range
        min_price = data['price'].min()
        max_price = data['price'].max()
        
        # Generate price points
        prices = np.linspace(min_price * 0.8, max_price * 1.2, 100)
        
        # Calculate revenues
        revenues = []
        for price in prices:
            # Simplified - would use full model in production
            elasticity = self._calculate_point_elasticity(data)
            base_price = data['price'].mean()
            base_quantity = data['quantity'].mean()
            
            quantity = base_quantity * (price / base_price) ** elasticity
            revenue = price * quantity
            revenues.append(revenue)
        
        # Find optimal
        optimal_idx = np.argmax(revenues)
        optimal_price = prices[optimal_idx]
        
        # Find 90% revenue range
        threshold = max(revenues) * 0.9
        good_prices = prices[np.array(revenues) >= threshold]
        
        return {
            'optimal_price': optimal_price,
            'min_good_price': good_prices.min(),
            'max_good_price': good_prices.max(),
            'price_flexibility': (good_prices.max() - good_prices.min()) / optimal_price
        }
    
    def _calculate_confidence(self, product_id: str) -> float:
        """Calculate confidence in elasticity estimate"""
        
        if product_id not in self.elasticity_models:
            return 0.0
        
        model_info = self.elasticity_models[product_id]
        
        # Factors affecting confidence
        data_points = model_info.get('data_points', 0)
        days_since_update = (pd.Timestamp.now() - model_info['last_update']).days
        base_confidence = model_info.get('confidence', 0.5)
        
        # Decay confidence over time
        time_decay = np.exp(-days_since_update / 30)
        
        # Boost confidence with more data
        data_boost = min(1.0, data_points / 100)
        
        final_confidence = base_confidence * time_decay * (0.5 + 0.5 * data_boost)
        
        return final_confidence


class DynamicPricingOptimizer:
    """Optimize prices using elasticity learning"""
    
    def __init__(self, elasticity_learner: PriceElasticityLearner):
        self.elasticity_learner = elasticity_learner
        self.optimization_history = {}
        
    def optimize_price(self, product_id: str, constraints: Dict) -> Dict:
        """Optimize price given constraints and elasticity"""
        
        # Get elasticity model
        if product_id not in self.elasticity_learner.elasticity_models:
            return {
                'optimal_price': constraints.get('current_price', 0),
                'confidence': 0,
                'reason': 'No elasticity model available'
            }
        
        # Get price response curve
        price_range = (
            constraints.get('min_price', 0),
            constraints.get('max_price', float('inf'))
        )
        
        response_curve = self.elasticity_learner.get_price_response_curve(product_id, price_range)
        
        # Find optimal considering constraints
        valid_prices = response_curve[
            (response_curve['price'] >= price_range[0]) &
            (response_curve['price'] <= price_range[1])
        ]
        
        if constraints.get('objective') == 'revenue':
            optimal_idx = valid_prices['expected_revenue'].idxmax()
        elif constraints.get('objective') == 'volume':
            optimal_idx = valid_prices['expected_quantity'].idxmax()
        else:  # Default to revenue
            optimal_idx = valid_prices['expected_revenue'].idxmax()
        
        optimal_row = valid_prices.loc[optimal_idx]
        
        # Calculate price change recommendation
        current_price = constraints.get('current_price', optimal_row['price'])
        price_change = (optimal_row['price'] - current_price) / current_price
        
        # Conservative adjustment based on confidence
        confidence = self.elasticity_learner._calculate_confidence(product_id)
        adjusted_price = current_price + (optimal_row['price'] - current_price) * confidence
        
        return {
            'current_price': current_price,
            'optimal_price': optimal_row['price'],
            'recommended_price': adjusted_price,
            'expected_revenue_lift': (optimal_row['expected_revenue'] / (current_price * 100) - 1),
            'confidence': confidence,
            'elasticity': optimal_row['elasticity'],
            'price_change_percent': price_change * 100
        }