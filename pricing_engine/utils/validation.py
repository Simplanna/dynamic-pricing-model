from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from fuzzywuzzy import fuzz
from ..core.models import ProductCategory


logger = logging.getLogger(__name__)


class DataValidator:
    """Data validation and quality assurance for pricing data"""
    
    PRICE_LIMITS = {
        'min': 1.0,
        'max': 1000.0
    }
    
    THC_LIMITS = {
        'min': 0.0,
        'max': 40.0
    }
    
    CBD_LIMITS = {
        'min': 0.0,
        'max': 30.0
    }
    
    VALID_CATEGORIES = [cat.value for cat in ProductCategory]
    
    def __init__(self):
        self.validation_errors = []
        self.validation_stats = {
            'total_validated': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0
        }
        
    def validate_price(self, price: float, product_name: str = "") -> Tuple[bool, Optional[str]]:
        """Validate price is within acceptable range"""
        if price is None:
            return False, f"Price is missing for {product_name}"
            
        if price < self.PRICE_LIMITS['min']:
            return False, f"Price ${price} below minimum ${self.PRICE_LIMITS['min']} for {product_name}"
            
        if price > self.PRICE_LIMITS['max']:
            return False, f"Price ${price} above maximum ${self.PRICE_LIMITS['max']} for {product_name}"
            
        return True, None
        
    def validate_cannabinoids(self, thc: float, cbd: float, product_name: str = "") -> Tuple[bool, List[str]]:
        """Validate THC and CBD percentages"""
        errors = []
        
        if thc < self.THC_LIMITS['min'] or thc > self.THC_LIMITS['max']:
            errors.append(f"THC {thc}% outside valid range {self.THC_LIMITS['min']}-{self.THC_LIMITS['max']}% for {product_name}")
            
        if cbd < self.CBD_LIMITS['min'] or cbd > self.CBD_LIMITS['max']:
            errors.append(f"CBD {cbd}% outside valid range {self.CBD_LIMITS['min']}-{self.CBD_LIMITS['max']}% for {product_name}")
            
        # Logic check: total cannabinoids shouldn't exceed ~35-40%
        if thc + cbd > 45:
            errors.append(f"Total cannabinoids {thc + cbd}% unrealistically high for {product_name}")
            
        return len(errors) == 0, errors
        
    def validate_category(self, category: str) -> Tuple[bool, Optional[str]]:
        """Validate product category"""
        if not category:
            return False, "Category is missing"
            
        normalized = category.lower().strip()
        
        # Direct match
        if normalized in self.VALID_CATEGORIES:
            return True, None
            
        # Fuzzy match for common variations
        category_mappings = {
            'flowers': 'flower',
            'pre-rolls': 'prerolls',
            'pre rolls': 'prerolls',
            'carts': 'vape',
            'cartridges': 'vape',
            'edible': 'edibles',
            'concentrate': 'concentrates',
            'tincture': 'tinctures',
            'topical': 'topicals'
        }
        
        if normalized in category_mappings:
            return True, None
            
        return False, f"Invalid category: {category}"
        
    def validate_product_data(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive validation of product data"""
        self.validation_stats['total_validated'] += 1
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'data': product.copy()
        }
        
        # Required fields
        required_fields = ['name', 'price', 'category']
        for field in required_fields:
            if field not in product or product[field] is None:
                validation_result['errors'].append(f"Required field '{field}' is missing")
                validation_result['valid'] = False
                
        # Price validation
        if 'price' in product:
            valid, error = self.validate_price(product.get('price'), product.get('name', ''))
            if not valid:
                validation_result['errors'].append(error)
                validation_result['valid'] = False
                
        # Cannabinoid validation
        thc = product.get('thc', 0.0) or 0.0
        cbd = product.get('cbd', 0.0) or 0.0
        valid, errors = self.validate_cannabinoids(thc, cbd, product.get('name', ''))
        if not valid:
            validation_result['errors'].extend(errors)
            validation_result['valid'] = False
            
        # Category validation
        if 'category' in product:
            valid, error = self.validate_category(product['category'])
            if not valid:
                validation_result['warnings'].append(error)
                # Try to fix category
                fixed_category = self.fix_category(product['category'])
                if fixed_category:
                    validation_result['data']['category'] = fixed_category
                    validation_result['warnings'].append(f"Category fixed: {product['category']} -> {fixed_category}")
                    
        # Data consistency checks
        if product.get('weight_grams'):
            if product['weight_grams'] <= 0 or product['weight_grams'] > 1000:
                validation_result['warnings'].append(f"Unusual weight: {product['weight_grams']}g")
                
        # Update stats
        if validation_result['valid']:
            self.validation_stats['passed'] += 1
        else:
            self.validation_stats['failed'] += 1
            
        if validation_result['warnings']:
            self.validation_stats['warnings'] += 1
            
        return validation_result
        
    def fix_category(self, category: str) -> Optional[str]:
        """Attempt to fix common category issues"""
        if not category:
            return None
            
        normalized = category.lower().strip()
        
        # Common mappings
        category_mappings = {
            'flowers': 'flower',
            'pre-rolls': 'prerolls',
            'pre rolls': 'prerolls',
            'carts': 'vape',
            'cartridges': 'vape',
            'vapes': 'vape',
            'edible': 'edibles',
            'concentrate': 'concentrates',
            'concentrates': 'concentrates',
            'tincture': 'tinctures',
            'topical': 'topicals'
        }
        
        if normalized in category_mappings:
            return category_mappings[normalized]
            
        # Fuzzy matching against valid categories
        best_match = None
        best_score = 0
        
        for valid_cat in self.VALID_CATEGORIES:
            score = fuzz.ratio(normalized, valid_cat)
            if score > best_score and score > 80:  # 80% similarity threshold
                best_score = score
                best_match = valid_cat
                
        return best_match
        
    def validate_batch(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate a batch of products"""
        results = []
        
        for product in products:
            result = self.validate_product_data(product)
            results.append(result)
            
        # Summary statistics
        summary = {
            'total': len(products),
            'valid': sum(1 for r in results if r['valid']),
            'invalid': sum(1 for r in results if not r['valid']),
            'warnings': sum(1 for r in results if r['warnings']),
            'validation_results': results,
            'stats': self.validation_stats.copy()
        }
        
        return summary


class ProductMatcher:
    """Match products across different dispensaries"""
    
    def __init__(self, confidence_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold
        
    def calculate_match_score(self, product1: Dict, product2: Dict) -> Tuple[float, Dict[str, bool]]:
        """Calculate match confidence between two products"""
        scores = {}
        criteria = {}
        
        # Category must match exactly
        if product1.get('category') == product2.get('category'):
            scores['category'] = 1.0
            criteria['category_match'] = True
        else:
            return 0.0, {'category_match': False}
            
        # Brand matching (exact or fuzzy)
        brand1 = (product1.get('brand') or '').lower().strip()
        brand2 = (product2.get('brand') or '').lower().strip()
        
        if brand1 and brand2:
            if brand1 == brand2:
                scores['brand'] = 1.0
                criteria['brand_match'] = True
            else:
                brand_score = fuzz.ratio(brand1, brand2) / 100.0
                scores['brand'] = brand_score * 0.8  # Slight penalty for fuzzy match
                criteria['brand_match'] = brand_score > 0.8
        else:
            scores['brand'] = 0.5  # No brand info
            criteria['brand_match'] = False
            
        # THC percentage matching (within 2% is considered a match)
        thc1 = product1.get('thc', 0.0) or 0.0
        thc2 = product2.get('thc', 0.0) or 0.0
        
        if abs(thc1 - thc2) <= 2.0:
            scores['thc'] = 1.0
            criteria['thc_match'] = True
        elif abs(thc1 - thc2) <= 5.0:
            scores['thc'] = 0.7
            criteria['thc_match'] = True
        else:
            scores['thc'] = max(0, 1 - abs(thc1 - thc2) / 20.0)
            criteria['thc_match'] = False
            
        # Name similarity (fuzzy matching)
        name1 = (product1.get('name') or '').lower().strip()
        name2 = (product2.get('name') or '').lower().strip()
        
        if name1 and name2:
            name_score = fuzz.token_sort_ratio(name1, name2) / 100.0
            scores['name'] = name_score
            criteria['name_match'] = name_score > 0.7
        else:
            scores['name'] = 0.0
            criteria['name_match'] = False
            
        # Weight matching (if available)
        weight1 = product1.get('weight_grams')
        weight2 = product2.get('weight_grams')
        
        if weight1 and weight2:
            if abs(weight1 - weight2) < 0.1:
                scores['weight'] = 1.0
                criteria['weight_match'] = True
            else:
                scores['weight'] = max(0, 1 - abs(weight1 - weight2) / max(weight1, weight2))
                criteria['weight_match'] = scores['weight'] > 0.9
        else:
            scores['weight'] = 0.5  # No weight info
            criteria['weight_match'] = False
            
        # Calculate weighted average
        weights = {
            'category': 0.25,
            'brand': 0.25,
            'thc': 0.20,
            'name': 0.20,
            'weight': 0.10
        }
        
        total_score = sum(scores.get(k, 0) * v for k, v in weights.items())
        
        return total_score, criteria
        
    def find_matches(self, target_product: Dict, competitor_products: List[Dict]) -> List[Dict[str, Any]]:
        """Find matching products from competitors"""
        matches = []
        
        for comp_product in competitor_products:
            score, criteria = self.calculate_match_score(target_product, comp_product)
            
            if score >= self.confidence_threshold:
                matches.append({
                    'matched_product': comp_product,
                    'confidence_score': score,
                    'match_criteria': criteria,
                    'price_difference': comp_product.get('price', 0) - target_product.get('price', 0)
                })
                
        # Sort by confidence score
        matches.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        return matches
        
    def deduplicate_matches(self, matches: List[Dict]) -> List[Dict]:
        """Remove duplicate matches, keeping highest confidence"""
        seen = set()
        unique_matches = []
        
        for match in matches:
            # Create unique key based on dispensary and product
            key = (
                match['matched_product'].get('dispensary_id', ''),
                match['matched_product'].get('name', ''),
                match['matched_product'].get('brand', '')
            )
            
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)
                
        return unique_matches