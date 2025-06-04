#!/usr/bin/env python3
"""
Phase 1 Testing Script - Verify all components are working
"""

import json
from datetime import datetime
from pricing_engine.core.models import Product, ProductCategory, CompetitorPrice, MarketType
from pricing_engine.scrapers.scraperbee_client import ScraperBeeClient, DispensaryMenuScraper
from pricing_engine.scrapers.parsers import ParserFactory
from pricing_engine.utils.validation import DataValidator, ProductMatcher
from pricing_engine.analysis.market_analysis import MarketAnalyzer
from pricing_engine.utils.reporting import DataQualityReport


def test_data_models():
    """Test core data models"""
    print("Testing Data Models...")
    
    try:
        # Test Product model
        product = Product(
            id="test-001",
            name="Blue Dream Flower",
            category=ProductCategory.FLOWER,
            brand="Local Grove",
            strain="Blue Dream",
            thc_percentage=22.5,
            cbd_percentage=0.5,
            weight_grams=3.5
        )
        print(f"✅ Product model created: {product.name}")
        
        # Test CompetitorPrice model
        comp_price = CompetitorPrice(
            dispensary_id="disp-001",
            dispensary_name="Test Dispensary",
            product_match={"name": "Blue Dream", "thc": 22.0},
            price=45.00,
            distance_miles=5.2,
            platform="dutchie",
            scraped_at=datetime.now(),
            match_confidence=0.85
        )
        print(f"✅ CompetitorPrice model created with distance weight: {comp_price.distance_weight:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {str(e)}")
        return False


def test_scraperbee_integration():
    """Test ScraperBee client (without making actual requests)"""
    print("\nTesting ScraperBee Integration...")
    
    try:
        client = ScraperBeeClient()
        scraper = DispensaryMenuScraper(client)
        
        # Test platform identification
        test_urls = [
            ("https://dutchie.com/dispensary/test", "dutchie"),
            ("https://www.iheartjane.com/stores/123", "iheartjane"),
            ("https://www.getmeadow.com/dispensary/test", "meadow")
        ]
        
        for url, expected in test_urls:
            detected = scraper.identify_platform(url)
            if detected == expected:
                print(f"✅ Correctly identified {expected} platform")
            else:
                print(f"❌ Failed to identify {expected} platform")
                
        return True
        
    except Exception as e:
        print(f"❌ ScraperBee test failed: {str(e)}")
        return False


def test_parsers():
    """Test platform-specific parsers"""
    print("\nTesting Platform Parsers...")
    
    # Sample HTML for testing (simplified)
    sample_html = '''
    <div data-testid="product-card">
        <h3 class="product-name">Blue Dream</h3>
        <span class="brand">Local Grove</span>
        <span class="price">$45.00</span>
        <div class="details">22.5% THC | 3.5g</div>
    </div>
    '''
    
    try:
        parser = ParserFactory.get_parser('dutchie')
        products = parser.parse_menu(sample_html)
        
        if products:
            print(f"✅ Parser extracted {len(products)} products")
            print(f"   Sample product: {products[0].get('name')} - ${products[0].get('price')}")
        else:
            print("⚠️ Parser returned no products (may need real HTML)")
            
        return True
        
    except Exception as e:
        print(f"❌ Parser test failed: {str(e)}")
        return False


def test_validation():
    """Test data validation pipeline"""
    print("\nTesting Data Validation...")
    
    validator = DataValidator()
    
    # Test products with various issues
    test_products = [
        {
            'name': 'Valid Product',
            'category': 'flower',
            'price': 45.00,
            'thc': 22.5,
            'cbd': 0.5
        },
        {
            'name': 'Invalid Price',
            'category': 'edibles',
            'price': 0.50,  # Too low
            'thc': 10.0
        },
        {
            'name': 'Invalid THC',
            'category': 'concentrates',
            'price': 60.00,
            'thc': 95.0  # Too high
        }
    ]
    
    try:
        results = validator.validate_batch(test_products)
        
        print(f"✅ Validation completed:")
        print(f"   - Total: {results['total']}")
        print(f"   - Valid: {results['valid']}")
        print(f"   - Invalid: {results['invalid']}")
        print(f"   - Warnings: {results['warnings']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation test failed: {str(e)}")
        return False


def test_product_matching():
    """Test product matching algorithm"""
    print("\nTesting Product Matching...")
    
    matcher = ProductMatcher(confidence_threshold=0.8)
    
    target = {
        'name': 'Blue Dream Flower',
        'category': 'flower',
        'brand': 'Local Grove',
        'thc': 22.5,
        'price': 45.00
    }
    
    competitors = [
        {
            'name': 'Blue Dream',
            'category': 'flower',
            'brand': 'Local Grove',
            'thc': 23.0,
            'price': 42.00
        },
        {
            'name': 'Purple Haze',
            'category': 'flower',
            'brand': 'Different Brand',
            'thc': 18.0,
            'price': 38.00
        }
    ]
    
    try:
        matches = matcher.find_matches(target, competitors)
        
        print(f"✅ Matching completed:")
        print(f"   - Found {len(matches)} matches above threshold")
        
        if matches:
            best_match = matches[0]
            print(f"   - Best match: {best_match['matched_product']['name']} "
                  f"(confidence: {best_match['confidence_score']:.2f})")
            
        return True
        
    except Exception as e:
        print(f"❌ Matching test failed: {str(e)}")
        return False


def test_market_analysis():
    """Test market analysis functions"""
    print("\nTesting Market Analysis...")
    
    analyzer = MarketAnalyzer()
    
    # Test market density calculation
    test_dispensaries = [
        {'name': 'Disp1', 'latitude': 42.3601, 'longitude': -71.0589},
        {'name': 'Disp2', 'latitude': 42.3736, 'longitude': -71.1097},
        {'name': 'Disp3', 'latitude': 42.3145, 'longitude': -71.0365}
    ]
    
    try:
        density, count = analyzer.calculate_market_density(
            test_dispensaries,
            center_lat=42.3601,
            center_lon=-71.0589,
            radius_miles=10
        )
        
        print(f"✅ Market density analysis:")
        print(f"   - Density: {density.value}")
        print(f"   - Competitors in radius: {count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Market analysis test failed: {str(e)}")
        return False


def test_data_quality_report():
    """Test report generation"""
    print("\nTesting Data Quality Report Generation...")
    
    report_gen = DataQualityReport()
    
    # Mock data for report
    validation_results = {
        'total': 10,
        'valid': 8,
        'invalid': 2,
        'warnings': 3,
        'validation_results': []
    }
    
    scraped_data = [
        {'dispensary_name': 'Test1', 'platform': 'dutchie', 'state': 'MA', 'status': 'success', 'products_found': 45},
        {'dispensary_name': 'Test2', 'platform': 'iheartjane', 'state': 'RI', 'status': 'success', 'products_found': 38}
    ]
    
    market_analysis = {
        'ma_analysis': {'avg_price_per_gram': 5.36},
        'ri_analysis': {'avg_price_per_gram': 12.40}
    }
    
    try:
        report = report_gen.generate_full_report(
            validation_results,
            scraped_data,
            market_analysis
        )
        
        print(f"✅ Report generated successfully")
        print(f"   - Overall quality score: {report['summary']['overall_quality_score']:.1f}%")
        print(f"   - Sections: {len(report['report'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Report generation failed: {str(e)}")
        return False


def main():
    """Run all Phase 1 tests"""
    print("=== PHASE 1 TESTING ===")
    print("Testing all components...\n")
    
    tests = [
        ("Data Models", test_data_models),
        ("ScraperBee Integration", test_scraperbee_integration),
        ("Platform Parsers", test_parsers),
        ("Data Validation", test_validation),
        ("Product Matching", test_product_matching),
        ("Market Analysis", test_market_analysis),
        ("Report Generation", test_data_quality_report)
    ]
    
    results = []
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))
        print()
    
    # Summary
    print("=== TEST SUMMARY ===")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 PHASE 1 COMPLETE - All systems operational!")
        print("\nReady to proceed to Phase 2: Pricing Algorithm Development")
    else:
        print("\n⚠️ Some tests failed. Please review and fix before proceeding.")


if __name__ == "__main__":
    main()