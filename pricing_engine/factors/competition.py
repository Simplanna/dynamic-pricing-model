"""Competition Factor - 15% Weight

Analyzes competitor pricing with distance weighting and fuzzy SKU matching.
Implements daily change caps and market differentiation.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import re


class CompetitionFactor:
    """Manages competition-based pricing adjustments"""
    
    # Distance-based weights (miles)
    DISTANCE_WEIGHTS = {
        (0, 1): 1.0,      # Direct competition
        (1, 3): 0.8,      # Strong influence
        (3, 5): 0.6,      # Moderate influence
        (5, 10): 0.4,     # Weak influence
        (10, 20): 0.2,    # Minimal influence
        (20, float('inf')): 0.0  # No influence
    }
    
    # Market-specific adjustments
    MARKET_FACTORS = {
        'MA': {
            'tax_rate': 0.20,  # 20% total tax
            'competitive_intensity': 1.1,  # More competitive
            'price_sensitivity': 0.9
        },
        'RI': {
            'tax_rate': 0.17,  # 17% total tax
            'competitive_intensity': 0.9,  # Less competitive
            'price_sensitivity': 1.1
        }
    }
    
    # Daily change cap
    MAX_DAILY_CHANGE = 0.05  # ±5%
    
    def __init__(self, fuzzy_match_threshold: float = 0.80):
        """Initialize with fuzzy matching threshold"""
        self.fuzzy_match_threshold = fuzzy_match_threshold
        
    def calculate_distance_weight(self, distance: float) -> float:
        """Calculate competition weight based on distance"""
        for distance_range, weight in self.DISTANCE_WEIGHTS.items():
            min_dist, max_dist = distance_range
            if min_dist <= distance < max_dist:
                return weight
        return 0.0
        
    def fuzzy_match_products(self, 
                           our_product: Dict,
                           competitor_products: List[Dict]) -> List[Dict]:
        """Find matching competitor products using fuzzy matching"""
        matches = []
        
        our_name = self._normalize_product_name(our_product.get('name', ''))
        our_category = our_product.get('category', '')
        our_brand = our_product.get('brand', '')
        
        for comp_product in competitor_products:
            comp_name = self._normalize_product_name(comp_product.get('name', ''))
            comp_category = comp_product.get('category', '')
            comp_brand = comp_product.get('brand', '')
            
            # Category must match exactly
            if our_category != comp_category:
                continue
                
            # Calculate similarity scores
            name_similarity = SequenceMatcher(None, our_name, comp_name).ratio()
            brand_similarity = SequenceMatcher(None, our_brand, comp_brand).ratio()
            
            # Combined similarity (weighted)
            combined_similarity = (name_similarity * 0.7) + (brand_similarity * 0.3)
            
            # Check for exact SKU match (if available)
            if our_product.get('sku') and comp_product.get('sku'):
                if our_product['sku'] == comp_product['sku']:
                    combined_similarity = 1.0
                    
            if combined_similarity >= self.fuzzy_match_threshold:
                matches.append({
                    'product': comp_product,
                    'similarity': combined_similarity,
                    'name_match': name_similarity,
                    'brand_match': brand_similarity
                })
                
        # Sort by similarity
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches
        
    def _normalize_product_name(self, name: str) -> str:
        """Normalize product name for better matching"""
        # Convert to lowercase
        name = name.lower()
        
        # Remove common cannabis terms and measurements
        remove_terms = [
            r'\d+(\.\d+)?\s*(g|gram|grams|oz|ounce|eighth|quarter|half)',
            r'\d+(\.\d+)?\s*(mg|thc|cbd|cbn)',
            r'(indica|sativa|hybrid)',
            r'[^\w\s]',  # Remove special characters
        ]
        
        for pattern in remove_terms:
            name = re.sub(pattern, ' ', name)
            
        # Remove extra whitespace
        name = ' '.join(name.split())
        
        return name
        
    def calculate_market_position(self,
                                our_price: float,
                                competitor_prices: List[float],
                                weights: List[float]) -> Dict:
        """Calculate our market position relative to competition"""
        if not competitor_prices:
            return {
                'position': 'unknown',
                'percentile': 50,
                'weighted_avg_comp_price': our_price,
                'price_delta': 0,
                'price_delta_pct': 0
            }
            
        # Calculate weighted average competitor price
        weighted_prices = np.array(competitor_prices) * np.array(weights)
        weighted_avg = weighted_prices.sum() / np.array(weights).sum()
        
        # Calculate our position
        all_prices = competitor_prices + [our_price]
        percentile = (sum(p < our_price for p in all_prices) / len(all_prices)) * 100
        
        # Determine position category
        if percentile >= 80:
            position = 'premium'
        elif percentile >= 60:
            position = 'above_market'
        elif percentile >= 40:
            position = 'at_market'
        elif percentile >= 20:
            position = 'below_market'
        else:
            position = 'discount'
            
        price_delta = our_price - weighted_avg
        price_delta_pct = (price_delta / weighted_avg) * 100 if weighted_avg > 0 else 0
        
        return {
            'position': position,
            'percentile': percentile,
            'weighted_avg_comp_price': weighted_avg,
            'price_delta': price_delta,
            'price_delta_pct': price_delta_pct
        }
        
    def calculate_factor_score(self,
                             product_data: Dict,
                             our_price: float,
                             competitor_data: List[Dict],
                             market: str = 'MA') -> Dict:
        """Calculate competition factor score and multiplier"""
        
        # Find matching competitor products
        matches = self.fuzzy_match_products(product_data, competitor_data)
        
        if not matches:
            return self._no_competition_result()
            
        # Extract prices and calculate weights
        comp_prices = []
        weights = []
        
        for match in matches:
            comp_product = match['product']
            price = comp_product.get('price', 0)
            distance = comp_product.get('distance', 0)
            
            if price > 0:
                # Calculate weight based on distance and match quality
                distance_weight = self.calculate_distance_weight(distance)
                match_weight = match['similarity']
                combined_weight = distance_weight * match_weight
                
                comp_prices.append(price)
                weights.append(combined_weight)
                
        if not comp_prices:
            return self._no_competition_result()
            
        # Calculate market position
        market_position = self.calculate_market_position(
            our_price, comp_prices, weights
        )
        
        # Calculate price adjustment
        multiplier = self._calculate_price_adjustment(
            market_position, market, len(matches)
        )
        
        # Apply daily change cap
        multiplier = self._apply_daily_cap(multiplier)
        
        # Calculate confidence
        confidence = self._calculate_confidence(matches, weights)
        
        return {
            'factor_name': 'competition',
            'weight': 0.15,
            'raw_score': market_position['percentile'],
            'position': market_position['position'],
            'multiplier': multiplier,
            'confidence': confidence,
            'details': {
                'competitors_analyzed': len(matches),
                'avg_competitor_price': round(market_position['weighted_avg_comp_price'], 2),
                'our_price': round(our_price, 2),
                'price_delta_pct': round(market_position['price_delta_pct'], 1),
                'market': market,
                'best_match_similarity': matches[0]['similarity'] if matches else 0
            }
        }
        
    def _calculate_price_adjustment(self, 
                                  market_position: Dict,
                                  market: str,
                                  num_competitors: int) -> float:
        """Calculate price multiplier based on market position"""
        position = market_position['position']
        price_delta_pct = market_position['price_delta_pct']
        
        # Get market factors
        market_factors = self.MARKET_FACTORS.get(market, self.MARKET_FACTORS['MA'])
        competitive_intensity = market_factors['competitive_intensity']
        
        # Base adjustments by position
        position_adjustments = {
            'premium': 0.98,       # Slight reduction if too high
            'above_market': 0.99,  # Minor reduction
            'at_market': 1.00,     # No change
            'below_market': 1.01,  # Minor increase
            'discount': 1.02       # Moderate increase
        }
        
        base_adjustment = position_adjustments.get(position, 1.0)
        
        # Adjust based on price delta magnitude
        if abs(price_delta_pct) > 20:
            # Large deviation - stronger correction
            if price_delta_pct > 0:
                base_adjustment *= 0.95
            else:
                base_adjustment *= 1.05
                
        # Adjust for competitive intensity
        base_adjustment = 1 + ((base_adjustment - 1) * competitive_intensity)
        
        # Dampen adjustment based on number of competitors
        if num_competitors < 3:
            # Few competitors - be conservative
            base_adjustment = 1 + ((base_adjustment - 1) * 0.5)
            
        return base_adjustment
        
    def _apply_daily_cap(self, multiplier: float) -> float:
        """Apply daily change cap to prevent extreme adjustments"""
        max_multiplier = 1 + self.MAX_DAILY_CHANGE
        min_multiplier = 1 - self.MAX_DAILY_CHANGE
        
        return np.clip(multiplier, min_multiplier, max_multiplier)
        
    def _calculate_confidence(self, matches: List[Dict], weights: List[float]) -> float:
        """Calculate confidence based on match quality and data availability"""
        if not matches:
            return 0.3
            
        confidence = 1.0
        
        # Factor in match quality
        best_match = matches[0]['similarity'] if matches else 0
        if best_match < 0.9:
            confidence *= (0.5 + best_match * 0.5)
            
        # Factor in number of competitors
        if len(matches) < 3:
            confidence *= 0.8
        elif len(matches) < 5:
            confidence *= 0.9
            
        # Factor in weight distribution (prefer diverse competition)
        if weights:
            weight_std = np.std(weights)
            if weight_std < 0.1:  # Too similar weights
                confidence *= 0.9
                
        return confidence
        
    def _no_competition_result(self) -> Dict:
        """Return default result when no competition data available"""
        return {
            'factor_name': 'competition',
            'weight': 0.15,
            'raw_score': 50,  # Assume middle position
            'position': 'unknown',
            'multiplier': 1.0,  # No adjustment
            'confidence': 0.3,  # Low confidence
            'details': {
                'competitors_analyzed': 0,
                'message': 'No matching competitor products found'
            }
        }