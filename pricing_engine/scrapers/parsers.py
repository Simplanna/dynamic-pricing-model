from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
import re
import json
import logging
from datetime import datetime


logger = logging.getLogger(__name__)


class BaseParser:
    """Base parser class with common functionality"""
    
    def __init__(self):
        self.errors = []
        
    def clean_price(self, price_str: str) -> Optional[float]:
        """Extract numeric price from string"""
        if not price_str:
            return None
            
        # Remove currency symbols and clean
        cleaned = re.sub(r'[^\d.,]', '', price_str)
        cleaned = cleaned.replace(',', '')
        
        try:
            return float(cleaned)
        except ValueError:
            return None
            
    def extract_thc_cbd(self, text: str) -> Dict[str, float]:
        """Extract THC and CBD percentages from text"""
        result = {'thc': 0.0, 'cbd': 0.0}
        
        if not text:
            return result
            
        # THC patterns
        thc_patterns = [
            r'(\d+\.?\d*)\s*%?\s*THC',
            r'THC\s*[:]\s*(\d+\.?\d*)\s*%?',
            r'Δ9-THC\s*[:]\s*(\d+\.?\d*)\s*%?'
        ]
        
        # CBD patterns
        cbd_patterns = [
            r'(\d+\.?\d*)\s*%?\s*CBD',
            r'CBD\s*[:]\s*(\d+\.?\d*)\s*%?'
        ]
        
        for pattern in thc_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result['thc'] = float(match.group(1))
                break
                
        for pattern in cbd_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result['cbd'] = float(match.group(1))
                break
                
        return result
        
    def extract_weight(self, text: str) -> Optional[float]:
        """Extract weight in grams from text"""
        if not text:
            return None
            
        # Common weight patterns
        patterns = [
            (r'(\d+\.?\d*)\s*g(?:ram)?s?\b', 1.0),  # grams
            (r'(\d+\.?\d*)\s*oz\b', 28.35),  # ounces to grams
            (r'(\d+)/(\d+)\s*oz\b', lambda m: (float(m.group(1))/float(m.group(2))) * 28.35),  # fractions
            (r'eighth', 3.5),  # common sizes
            (r'quarter', 7.0),
            (r'half', 14.0)
        ]
        
        for pattern, multiplier in patterns:
            if callable(multiplier):
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return multiplier(match)
            else:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return float(match.group(1)) * multiplier
                    
        return None


class DutchieParser(BaseParser):
    """Parser for Dutchie platform"""
    
    def parse_menu(self, html: str) -> List[Dict[str, Any]]:
        """Parse Dutchie menu HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        products = []
        
        # Find product cards
        product_cards = soup.find_all(['div', 'article'], {'data-testid': 'product-card'})
        if not product_cards:
            # Fallback selectors
            product_cards = soup.find_all('div', class_=re.compile(r'product-card|menu-item'))
            
        for card in product_cards:
            try:
                product = self.parse_product_card(card)
                if product and product.get('price'):
                    products.append(product)
            except Exception as e:
                logger.error(f"Error parsing Dutchie product: {str(e)}")
                self.errors.append(str(e))
                
        return products
        
    def parse_product_card(self, card) -> Dict[str, Any]:
        """Parse individual Dutchie product card"""
        product = {}
        
        # Product name
        name_elem = card.find(['h3', 'h4'], class_=re.compile(r'product-name|title'))
        if name_elem:
            product['name'] = name_elem.get_text(strip=True)
            
        # Brand
        brand_elem = card.find(['span', 'div'], class_=re.compile(r'brand|vendor'))
        if brand_elem:
            product['brand'] = brand_elem.get_text(strip=True)
            
        # Price
        price_elem = card.find(['span', 'div'], class_=re.compile(r'price|cost'))
        if price_elem:
            product['price'] = self.clean_price(price_elem.get_text())
            
        # Category
        category_elem = card.find(['span', 'div'], class_=re.compile(r'category|type'))
        if category_elem:
            product['category'] = category_elem.get_text(strip=True).lower()
            
        # THC/CBD from description or details
        details_elem = card.find(['div', 'p'], class_=re.compile(r'description|details|potency'))
        if details_elem:
            details_text = details_elem.get_text()
            cannabinoids = self.extract_thc_cbd(details_text)
            product.update(cannabinoids)
            
            # Weight
            weight = self.extract_weight(details_text)
            if weight:
                product['weight_grams'] = weight
                
        # Product URL
        link_elem = card.find('a', href=True)
        if link_elem:
            product['url'] = link_elem['href']
            
        return product


class IHeartJaneParser(BaseParser):
    """Parser for iHeartJane platform"""
    
    def parse_menu(self, html: str) -> List[Dict[str, Any]]:
        """Parse iHeartJane menu HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        products = []
        
        # Look for product list items
        product_items = soup.find_all(['div', 'li'], class_=re.compile(r'product-list-item|product-tile'))
        
        for item in product_items:
            try:
                product = self.parse_product_item(item)
                if product and product.get('price'):
                    products.append(product)
            except Exception as e:
                logger.error(f"Error parsing iHeartJane product: {str(e)}")
                self.errors.append(str(e))
                
        return products
        
    def parse_product_item(self, item) -> Dict[str, Any]:
        """Parse individual iHeartJane product"""
        product = {}
        
        # Product name
        name_elem = item.find(['h3', 'h4', 'div'], class_=re.compile(r'product-name|name'))
        if name_elem:
            product['name'] = name_elem.get_text(strip=True)
            
        # Brand
        brand_elem = item.find(['div', 'span'], class_=re.compile(r'brand|manufacturer'))
        if brand_elem:
            product['brand'] = brand_elem.get_text(strip=True)
            
        # Price - iHeartJane often has multiple price options
        price_elem = item.find(['span', 'div'], class_=re.compile(r'price'))
        if price_elem:
            # Get the first/lowest price if multiple
            price_text = price_elem.get_text()
            prices = re.findall(r'\$(\d+\.?\d*)', price_text)
            if prices:
                product['price'] = float(prices[0])
                
        # Category
        category_elem = item.find(['span', 'div'], class_=re.compile(r'category|product-type'))
        if category_elem:
            product['category'] = category_elem.get_text(strip=True).lower()
            
        # Strain and cannabinoids
        strain_elem = item.find(['div', 'span'], class_=re.compile(r'strain'))
        if strain_elem:
            product['strain'] = strain_elem.get_text(strip=True)
            
        # THC/CBD often in a separate element
        potency_elem = item.find(['div', 'span'], class_=re.compile(r'potency|thc|cannabinoid'))
        if potency_elem:
            cannabinoids = self.extract_thc_cbd(potency_elem.get_text())
            product.update(cannabinoids)
            
        # Weight/size
        size_elem = item.find(['span', 'div'], class_=re.compile(r'size|weight|amount'))
        if size_elem:
            weight = self.extract_weight(size_elem.get_text())
            if weight:
                product['weight_grams'] = weight
                
        return product


class MeadowParser(BaseParser):
    """Parser for Meadow platform"""
    
    def parse_menu(self, html: str) -> List[Dict[str, Any]]:
        """Parse Meadow menu HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        products = []
        
        # Meadow often uses React, so look for data attributes
        product_cards = soup.find_all(['div', 'article'], class_=re.compile(r'product-card|product-item'))
        
        # Also check for JSON data in script tags
        script_data = self.extract_json_data(soup)
        if script_data:
            products.extend(self.parse_json_products(script_data))
            
        for card in product_cards:
            try:
                product = self.parse_product_card(card)
                if product and product.get('price'):
                    products.append(product)
            except Exception as e:
                logger.error(f"Error parsing Meadow product: {str(e)}")
                self.errors.append(str(e))
                
        return products
        
    def extract_json_data(self, soup) -> Optional[Dict]:
        """Extract product data from JSON-LD or React props"""
        scripts = soup.find_all('script', type=['application/ld+json', 'application/json'])
        
        for script in scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, dict) and 'products' in data:
                    return data
            except:
                continue
                
        return None
        
    def parse_json_products(self, data: Dict) -> List[Dict[str, Any]]:
        """Parse products from JSON data"""
        products = []
        
        for item in data.get('products', []):
            product = {
                'name': item.get('name'),
                'brand': item.get('brand', {}).get('name'),
                'price': item.get('price'),
                'category': item.get('category', '').lower(),
                'thc': item.get('thc_percentage', 0.0),
                'cbd': item.get('cbd_percentage', 0.0),
                'weight_grams': item.get('weight'),
                'strain': item.get('strain')
            }
            
            # Clean up None values
            product = {k: v for k, v in product.items() if v is not None}
            products.append(product)
            
        return products
        
    def parse_product_card(self, card) -> Dict[str, Any]:
        """Parse individual Meadow product card"""
        product = {}
        
        # Similar structure to other parsers
        name_elem = card.find(['h3', 'h4'], class_=re.compile(r'product-name|title'))
        if name_elem:
            product['name'] = name_elem.get_text(strip=True)
            
        # Price
        price_elem = card.find(['span', 'div'], class_=re.compile(r'price'))
        if price_elem:
            product['price'] = self.clean_price(price_elem.get_text())
            
        # Other details
        details = card.find_all(['span', 'div'], class_=re.compile(r'detail|info'))
        details_text = ' '.join([d.get_text() for d in details])
        
        cannabinoids = self.extract_thc_cbd(details_text)
        product.update(cannabinoids)
        
        weight = self.extract_weight(details_text)
        if weight:
            product['weight_grams'] = weight
            
        return product


class TymberParser(BaseParser):
    """Parser for Tymber platform"""
    
    def parse_menu(self, html: str) -> List[Dict[str, Any]]:
        """Parse Tymber menu HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        products = []
        
        # Tymber menu structure
        menu_items = soup.find_all(['div', 'li'], class_=re.compile(r'menu-item|product'))
        
        for item in menu_items:
            try:
                product = self.parse_menu_item(item)
                if product and product.get('price'):
                    products.append(product)
            except Exception as e:
                logger.error(f"Error parsing Tymber product: {str(e)}")
                self.errors.append(str(e))
                
        return products
        
    def parse_menu_item(self, item) -> Dict[str, Any]:
        """Parse individual Tymber menu item"""
        product = {}
        
        # Product info often in nested structure
        info_elem = item.find(['div'], class_=re.compile(r'info|details'))
        if info_elem:
            # Name
            name_elem = info_elem.find(['h3', 'h4', 'span'], class_=re.compile(r'name|title'))
            if name_elem:
                product['name'] = name_elem.get_text(strip=True)
                
            # Price
            price_elem = info_elem.find(['span', 'div'], class_=re.compile(r'price'))
            if price_elem:
                product['price'] = self.clean_price(price_elem.get_text())
                
        # Extract all text for analysis
        item_text = item.get_text()
        
        # THC/CBD
        cannabinoids = self.extract_thc_cbd(item_text)
        product.update(cannabinoids)
        
        # Weight
        weight = self.extract_weight(item_text)
        if weight:
            product['weight_grams'] = weight
            
        # Category - Tymber often uses badges
        badge_elem = item.find(['span', 'div'], class_=re.compile(r'badge|category|type'))
        if badge_elem:
            product['category'] = badge_elem.get_text(strip=True).lower()
            
        return product


class ParserFactory:
    """Factory for creating appropriate parser based on platform"""
    
    PARSERS = {
        'dutchie': DutchieParser,
        'iheartjane': IHeartJaneParser,
        'meadow': MeadowParser,
        'tymber': TymberParser
    }
    
    @classmethod
    def get_parser(cls, platform: str) -> BaseParser:
        """Get parser instance for platform"""
        parser_class = cls.PARSERS.get(platform.lower())
        if not parser_class:
            raise ValueError(f"No parser available for platform: {platform}")
        return parser_class()
        
    @classmethod
    def parse_menu(cls, html: str, platform: str) -> List[Dict[str, Any]]:
        """Parse menu HTML for given platform"""
        parser = cls.get_parser(platform)
        return parser.parse_menu(html)