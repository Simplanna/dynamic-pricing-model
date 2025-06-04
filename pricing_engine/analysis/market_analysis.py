import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from geopy.distance import geodesic
import logging
from ..core.models import MarketType, CompetitorDensity
from ..data.schemas import SPECIAL_EVENTS


logger = logging.getLogger(__name__)


class MarketAnalyzer:
    """Analyze market patterns and cross-border shopping behavior"""
    
    def __init__(self):
        self.analysis_results = {}
        
    def calculate_market_density(self, dispensaries: List[Dict], 
                               center_lat: float, center_lon: float,
                               radius_miles: float = 15.0) -> Tuple[CompetitorDensity, int]:
        """Calculate competitor density around a location"""
        competitors_in_radius = 0
        
        for disp in dispensaries:
            distance = geodesic(
                (center_lat, center_lon),
                (disp['latitude'], disp['longitude'])
            ).miles
            
            if distance <= radius_miles:
                competitors_in_radius += 1
                
        # Classify density
        if competitors_in_radius < 5:
            density = CompetitorDensity.SPARSE
        elif competitors_in_radius <= 15:
            density = CompetitorDensity.MODERATE
        else:
            density = CompetitorDensity.DENSE
            
        return density, competitors_in_radius
        
    def analyze_cross_border_patterns(self, ma_prices: pd.DataFrame, 
                                    ri_prices: pd.DataFrame,
                                    border_distance_miles: float = 20.0) -> Dict[str, Any]:
        """Analyze cross-border shopping patterns between MA and RI"""
        
        analysis = {
            'price_differential': {},
            'border_zone_activity': {},
            'category_arbitrage': {}
        }
        
        # Calculate average prices by category for each state
        for category in ma_prices['category'].unique():
            ma_avg = ma_prices[ma_prices['category'] == category]['price'].mean()
            ri_avg = ri_prices[ri_prices['category'] == category]['price'].mean()
            
            analysis['price_differential'][category] = {
                'ma_avg_price': ma_avg,
                'ri_avg_price': ri_avg,
                'price_gap': ri_avg - ma_avg,
                'price_gap_pct': ((ri_avg - ma_avg) / ma_avg) * 100 if ma_avg > 0 else 0
            }
            
        # Identify border zone dispensaries
        border_ma = ma_prices[ma_prices['distance_to_border'] <= border_distance_miles]
        border_ri = ri_prices[ri_prices['distance_to_border'] <= border_distance_miles]
        
        analysis['border_zone_activity'] = {
            'ma_border_dispensaries': len(border_ma['dispensary_id'].unique()),
            'ri_border_dispensaries': len(border_ri['dispensary_id'].unique()),
            'ma_border_avg_price': border_ma['price'].mean(),
            'ri_border_avg_price': border_ri['price'].mean(),
            'price_compression': abs(border_ma['price'].mean() - border_ri['price'].mean()) < 
                               abs(ma_prices['price'].mean() - ri_prices['price'].mean())
        }
        
        # Category-specific arbitrage opportunities
        for category in ma_prices['category'].unique():
            ma_cat = border_ma[border_ma['category'] == category]
            ri_cat = border_ri[border_ri['category'] == category]
            
            if len(ma_cat) > 0 and len(ri_cat) > 0:
                analysis['category_arbitrage'][category] = {
                    'max_savings': ri_cat['price'].max() - ma_cat['price'].min(),
                    'avg_savings': ri_cat['price'].mean() - ma_cat['price'].mean(),
                    'worth_travel': (ri_cat['price'].mean() - ma_cat['price'].mean()) > 10  # $10 threshold
                }
                
        return analysis
        
    def detect_seasonal_patterns(self, price_history: pd.DataFrame) -> Dict[str, Any]:
        """Detect seasonal and event-based pricing patterns"""
        
        # Ensure datetime
        price_history['date'] = pd.to_datetime(price_history['recorded_at'])
        
        patterns = {
            'weekly': {},
            'monthly': {},
            'special_events': {},
            'seasonal': {}
        }
        
        # Day of week patterns
        price_history['day_of_week'] = price_history['date'].dt.day_name()
        weekly_avg = price_history.groupby('day_of_week')['price'].agg(['mean', 'std', 'count'])
        patterns['weekly'] = weekly_avg.to_dict('index')
        
        # Monthly patterns
        price_history['month'] = price_history['date'].dt.month_name()
        monthly_avg = price_history.groupby('month')['price'].agg(['mean', 'std', 'count'])
        patterns['monthly'] = monthly_avg.to_dict('index')
        
        # Special event patterns
        for date_str, event_name in SPECIAL_EVENTS.items():
            month, day = map(int, date_str.split('-'))
            
            # Find prices around this event (±3 days)
            event_prices = []
            for year in price_history['date'].dt.year.unique():
                try:
                    event_date = datetime(year, month, day)
                    mask = (price_history['date'] >= event_date - timedelta(days=3)) & \
                           (price_history['date'] <= event_date + timedelta(days=3))
                    event_prices.extend(price_history[mask]['price'].tolist())
                except:
                    continue
                    
            if event_prices:
                patterns['special_events'][event_name] = {
                    'avg_price': np.mean(event_prices),
                    'price_change_pct': ((np.mean(event_prices) / price_history['price'].mean()) - 1) * 100,
                    'sample_size': len(event_prices)
                }
                
        # Seasonal patterns (quarters)
        price_history['quarter'] = price_history['date'].dt.quarter
        price_history['season'] = price_history['quarter'].map({
            1: 'Winter',
            2: 'Spring', 
            3: 'Summer',
            4: 'Fall'
        })
        
        seasonal_avg = price_history.groupby('season')['price'].agg(['mean', 'std', 'count'])
        patterns['seasonal'] = seasonal_avg.to_dict('index')
        
        # Summer tourism impact (for MA/RI coastal areas)
        summer_months = price_history[price_history['date'].dt.month.isin([6, 7, 8])]
        other_months = price_history[~price_history['date'].dt.month.isin([6, 7, 8])]
        
        patterns['seasonal']['summer_tourism_impact'] = {
            'summer_avg': summer_months['price'].mean(),
            'non_summer_avg': other_months['price'].mean(),
            'price_increase_pct': ((summer_months['price'].mean() / other_months['price'].mean()) - 1) * 100
        }
        
        return patterns
        
    def analyze_inventory_age_impact(self, products_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how product age affects pricing"""
        
        if 'days_since_harvest' not in products_df.columns:
            return {'error': 'No age data available'}
            
        # Remove invalid age data
        valid_age = products_df[products_df['days_since_harvest'].notna()]
        
        age_analysis = {
            'age_brackets': {},
            'optimal_pricing_window': {},
            'depreciation_curve': {}
        }
        
        # Define age brackets
        age_brackets = [
            (0, 30, 'Fresh (0-30 days)'),
            (31, 60, 'Prime (31-60 days)'),
            (61, 90, 'Mature (61-90 days)'),
            (91, 120, 'Aged (91-120 days)'),
            (121, 999, 'Old (120+ days)')
        ]
        
        for min_age, max_age, label in age_brackets:
            bracket_data = valid_age[
                (valid_age['days_since_harvest'] >= min_age) & 
                (valid_age['days_since_harvest'] <= max_age)
            ]
            
            if len(bracket_data) > 0:
                age_analysis['age_brackets'][label] = {
                    'avg_price': bracket_data['price'].mean(),
                    'price_std': bracket_data['price'].std(),
                    'count': len(bracket_data),
                    'avg_discount_from_fresh': None
                }
                
        # Calculate discounts relative to fresh product
        if 'Fresh (0-30 days)' in age_analysis['age_brackets']:
            fresh_price = age_analysis['age_brackets']['Fresh (0-30 days)']['avg_price']
            
            for label, data in age_analysis['age_brackets'].items():
                discount = ((data['avg_price'] - fresh_price) / fresh_price) * 100
                data['avg_discount_from_fresh'] = discount
                
        # Find optimal pricing window (highest price with good volume)
        for label, data in age_analysis['age_brackets'].items():
            if data['count'] > 10:  # Minimum sample size
                if 'optimal_window' not in age_analysis['optimal_pricing_window'] or \
                   data['avg_price'] > age_analysis['optimal_pricing_window']['price']:
                    age_analysis['optimal_pricing_window'] = {
                        'window': label,
                        'price': data['avg_price'],
                        'sample_size': data['count']
                    }
                    
        return age_analysis
        
    def calculate_dynamic_radius(self, center_lat: float, center_lon: float,
                               dispensaries: List[Dict]) -> float:
        """Calculate optimal search radius based on competitor density"""
        
        # Start with small radius and expand until we find enough competitors
        min_competitors = 3
        max_radius = 25.0
        
        for radius in range(5, int(max_radius) + 1, 5):
            competitors = 0
            
            for disp in dispensaries:
                distance = geodesic(
                    (center_lat, center_lon),
                    (disp['latitude'], disp['longitude'])
                ).miles
                
                if distance <= radius:
                    competitors += 1
                    
            if competitors >= min_competitors:
                # Add some buffer
                return float(min(radius + 2, max_radius))
                
        return max_radius
        
    def generate_market_summary(self, state: MarketType, 
                              competitor_prices: pd.DataFrame,
                              historical_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Generate comprehensive market summary"""
        
        summary = {
            'state': state.value,
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'insights': []
        }
        
        # Basic metrics
        summary['metrics'] = {
            'avg_price_per_gram': competitor_prices['price'].mean() / competitor_prices.get('weight_grams', 1).mean(),
            'price_range': {
                'min': competitor_prices['price'].min(),
                'max': competitor_prices['price'].max(),
                'std': competitor_prices['price'].std()
            },
            'competitor_count': len(competitor_prices['dispensary_id'].unique()),
            'product_count': len(competitor_prices)
        }
        
        # Category breakdown
        category_stats = competitor_prices.groupby('category').agg({
            'price': ['mean', 'min', 'max', 'count']
        }).round(2)
        
        summary['category_analysis'] = category_stats.to_dict()
        
        # Market insights
        if state == MarketType.MASSACHUSETTS:
            if summary['metrics']['avg_price_per_gram'] < 5.0:
                summary['insights'].append("MA market shows severe price compression due to oversupply")
            summary['insights'].append(f"Average price per gram: ${summary['metrics']['avg_price_per_gram']:.2f}")
            
        elif state == MarketType.RHODE_ISLAND:
            if summary['metrics']['avg_price_per_gram'] > 10.0:
                summary['insights'].append("RI maintains premium pricing due to limited licenses")
            summary['insights'].append("Consider MA border competition for pricing strategy")
            
        # Historical trends if available
        if historical_data is not None and len(historical_data) > 0:
            # 30-day price trend
            thirty_days_ago = datetime.now() - timedelta(days=30)
            recent_data = historical_data[historical_data['recorded_at'] >= thirty_days_ago]
            
            if len(recent_data) > 0:
                price_change = (
                    (competitor_prices['price'].mean() - recent_data['price'].mean()) / 
                    recent_data['price'].mean()
                ) * 100
                
                summary['metrics']['30_day_price_change'] = price_change
                
                if abs(price_change) > 5:
                    trend = "increasing" if price_change > 0 else "decreasing"
                    summary['insights'].append(f"Prices {trend} by {abs(price_change):.1f}% over last 30 days")
                    
        return summary