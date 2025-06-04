import requests
from typing import Dict, List, Optional, Any
import time
from datetime import datetime
import json
from urllib.parse import urlencode
import logging


logger = logging.getLogger(__name__)


class ScraperBeeClient:
    """ScraperBee API client for dispensary menu scraping"""
    
    API_KEY = "99XUHEZ6W021CVZWRCOO6KZ016HMFL3XFH43FB7AU556GZ94DA0RGA137PBLLYW1QSOOAXCVPPV2W930"
    BASE_URL = "https://app.scraperbee.com/api/v1/"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or self.API_KEY
        self.session = requests.Session()
        self.request_count = 0
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Implement rate limiting to avoid API throttling"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < 1.0:  # Max 1 request per second
            time.sleep(1.0 - time_since_last)
        self.last_request_time = time.time()
        
    def scrape_url(self, url: str, render_js: bool = True, 
                   wait_for_selector: Optional[str] = None,
                   premium_proxy: bool = True) -> Dict[str, Any]:
        """
        Scrape a single URL using ScraperBee
        
        Args:
            url: Target URL to scrape
            render_js: Whether to render JavaScript (needed for most dispensary sites)
            wait_for_selector: CSS selector to wait for before returning
            premium_proxy: Use premium proxies for better success rate
            
        Returns:
            Dict containing HTML content and metadata
        """
        self._rate_limit()
        
        params = {
            'api_key': self.api_key,
            'url': url,
            'render_js': str(render_js).lower(),
            'premium_proxy': str(premium_proxy).lower(),
            'country_code': 'us'
        }
        
        if wait_for_selector:
            params['wait_for_selector'] = wait_for_selector
            
        try:
            self.request_count += 1
            response = self.session.get(
                self.BASE_URL,
                params=params,
                timeout=60
            )
            response.raise_for_status()
            
            return {
                'html': response.text,
                'status_code': response.status_code,
                'url': url,
                'scraped_at': datetime.now(),
                'request_count': self.request_count
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"ScraperBee request failed for {url}: {str(e)}")
            return {
                'html': None,
                'status_code': getattr(e.response, 'status_code', None),
                'error': str(e),
                'url': url,
                'scraped_at': datetime.now()
            }
    
    def batch_scrape(self, urls: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Scrape multiple URLs sequentially with rate limiting"""
        results = []
        for url in urls:
            result = self.scrape_url(url, **kwargs)
            results.append(result)
            if result.get('error'):
                logger.warning(f"Failed to scrape {url}: {result['error']}")
        return results
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        params = {'api_key': self.api_key}
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}usage",
                params=params
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get usage stats: {str(e)}")
            return {'error': str(e)}


class DispensaryMenuScraper:
    """High-level scraper for different dispensary platforms"""
    
    PLATFORM_SELECTORS = {
        'dutchie': {
            'wait_selector': '[data-testid="product-card"]',
            'menu_container': '[data-testid="menu-section"]'
        },
        'iheartjane': {
            'wait_selector': '.product-list-item',
            'menu_container': '.products-grid'
        },
        'meadow': {
            'wait_selector': '.product-card',
            'menu_container': '.products-container'
        },
        'tymber': {
            'wait_selector': '.menu-item',
            'menu_container': '.menu-container'
        }
    }
    
    def __init__(self, client: Optional[ScraperBeeClient] = None):
        self.client = client or ScraperBeeClient()
        
    def identify_platform(self, url: str) -> Optional[str]:
        """Identify the e-commerce platform from URL"""
        url_lower = url.lower()
        
        if 'dutchie' in url_lower or 'dutchie.com' in url_lower:
            return 'dutchie'
        elif 'iheartjane' in url_lower or 'jane.com' in url_lower:
            return 'iheartjane'
        elif 'meadow' in url_lower or 'getmeadow.com' in url_lower:
            return 'meadow'
        elif 'tymber' in url_lower:
            return 'tymber'
        else:
            # Try to detect from embedded iframe patterns
            return None
            
    def scrape_dispensary_menu(self, url: str, platform: Optional[str] = None) -> Dict[str, Any]:
        """
        Scrape a dispensary menu page
        
        Args:
            url: Dispensary menu URL
            platform: Platform name (auto-detected if not provided)
            
        Returns:
            Scraped data including HTML and metadata
        """
        if not platform:
            platform = self.identify_platform(url)
            
        if platform and platform in self.PLATFORM_SELECTORS:
            wait_selector = self.PLATFORM_SELECTORS[platform]['wait_selector']
        else:
            wait_selector = None
            logger.warning(f"Unknown platform for {url}, proceeding without wait selector")
            
        result = self.client.scrape_url(
            url,
            render_js=True,
            wait_for_selector=wait_selector,
            premium_proxy=True
        )
        
        result['platform'] = platform
        return result
    
    def scrape_competitor_dispensaries(self, dispensary_urls: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Scrape multiple competitor dispensaries
        
        Args:
            dispensary_urls: List of dicts with 'url', 'name', 'distance_miles'
            
        Returns:
            List of scraped results with metadata
        """
        results = []
        
        for dispensary in dispensary_urls:
            logger.info(f"Scraping {dispensary['name']} at {dispensary['url']}")
            
            scraped = self.scrape_dispensary_menu(dispensary['url'])
            scraped.update({
                'dispensary_name': dispensary['name'],
                'distance_miles': dispensary.get('distance_miles', 0),
                'dispensary_id': dispensary.get('id', ''),
                'city': dispensary.get('city', ''),
                'state': dispensary.get('state', '')
            })
            
            results.append(scraped)
            
        return results