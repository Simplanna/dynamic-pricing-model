"""Test script for the nine-factor pricing engine"""

from datetime import datetime, timedelta
from pricing_engine.orchestrator import PricingOrchestrator
from pricing_engine.utils import PricingAuditLogger, PerformanceTracker

def create_test_product():
    """Create a test cannabis product"""
    return {
        'id': 'PROD001',
        'name': 'Blue Dream 3.5g',
        'category': 'flower',
        'brand': 'Premium Gardens',
        'brand_tier': 'premium',
        'current_price': 45.00,
        'base_price': 40.00,
        'cost': 25.00,
        'thc_percentage': 22.5,
        'size': 3.5,
        'size_unit': 'g',
        'harvest_date': (datetime.now() - timedelta(days=30)).isoformat(),
        'taxes_included': True
    }

def create_test_market_data():
    """Create test market data"""
    return {
        'inventory': {
            'quantity': 50,
            'days_until_expiry': 60,
            'last_updated': datetime.now().isoformat()
        },
        'sales_velocity': 2.5,  # units per day
        'sales_history': [
            {'date': (datetime.now() - timedelta(days=i)).isoformat(), 'quantity': 2 + (i % 3)}
            for i in range(14)
        ],
        'competitors': [
            {
                'name': 'Blue Dream 3.5g',
                'brand': 'Standard Farms',
                'category': 'flower',
                'price': 42.00,
                'distance': 2.5,
                'sku': 'COMP001'
            },
            {
                'name': 'Blue Dream Eighth',
                'brand': 'Local Grow',
                'category': 'flower',
                'price': 38.00,
                'distance': 4.0,
                'sku': 'COMP002'
            },
            {
                'name': 'Dream Blue 3.5g',
                'brand': 'Premium Gardens',
                'category': 'flower',
                'price': 44.00,
                'distance': 1.5,
                'sku': 'COMP003'
            }
        ],
        'market': 'MA',
        'store': {
            'location_type': 'suburban',
            'city': 'Worcester',
            'state': 'MA'
        },
        'primary_segment': 'regular',
        'segment_mix': {
            'regular': 0.6,
            'premium': 0.2,
            'value': 0.2
        },
        'pricing_date': datetime.now(),
        'last_updated': datetime.now()
    }

def test_pricing_engine():
    """Test the complete pricing engine"""
    print("=== Cannabis Dynamic Pricing Engine Test ===\n")
    
    # Initialize components
    orchestrator = PricingOrchestrator()
    audit_logger = PricingAuditLogger()
    performance_tracker = PerformanceTracker()
    
    # Create test data
    product = create_test_product()
    market_data = create_test_market_data()
    
    print(f"Product: {product['name']}")
    print(f"Current Price: ${product['current_price']:.2f}")
    print(f"Category: {product['category']}")
    print(f"THC: {product['thc_percentage']}%\n")
    
    # Calculate price
    decision = orchestrator.calculate_price(product, market_data)
    
    # Display results
    print("=== Pricing Decision ===")
    print(f"Recommended Price: ${decision.recommended_price:.2f}")
    print(f"Final Price: ${decision.final_price:.2f}")
    print(f"Price Change: {decision.price_change_pct:+.1f}%")
    print(f"Confidence Score: {decision.confidence_score:.2%}\n")
    
    # Display factor analysis
    print("=== Factor Analysis ===")
    for factor_name, factor_data in decision.factors.items():
        print(f"\n{factor_name.replace('_', ' ').title()}:")
        print(f"  Weight: {factor_data['weight']:.0%}")
        print(f"  Multiplier: {factor_data['multiplier']:.3f}")
        print(f"  Confidence: {factor_data['confidence']:.2f}")
        if 'details' in factor_data:
            for key, value in factor_data['details'].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
    
    # Display safety overrides
    if decision.safety_overrides:
        print("\n=== Safety Overrides Applied ===")
        for override in decision.safety_overrides:
            print(f"- {override}")
    
    # Validate decision
    is_valid, issues = orchestrator.validate_pricing_decision(decision)
    print(f"\n=== Validation ===")
    print(f"Decision Valid: {is_valid}")
    if issues:
        print("Issues:")
        for issue in issues:
            print(f"- {issue}")
    
    # Log the decision
    audit_log = orchestrator.generate_audit_log(decision)
    log_id = audit_logger.log_pricing_decision(audit_log)
    print(f"\n=== Audit Trail ===")
    print(f"Decision logged with ID: {log_id}")
    
    # Track performance metrics
    performance_tracker.track_decision_metrics(audit_log)
    
    # Test special events
    print("\n=== Testing Market Events ===")
    
    # Test 4/20 pricing
    four_twenty = datetime(datetime.now().year, 4, 20)
    market_data['pricing_date'] = four_twenty
    decision_420 = orchestrator.calculate_price(product, market_data)
    print(f"\n4/20 Pricing:")
    print(f"  Regular Price: ${product['current_price']:.2f}")
    print(f"  4/20 Price: ${decision_420.final_price:.2f}")
    print(f"  Event Multiplier: {decision_420.factors['market_events']['multiplier']:.3f}")
    
    # Test inventory pressure scenarios
    print("\n=== Testing Inventory Scenarios ===")
    
    # Low inventory
    market_data_low = market_data.copy()
    market_data_low['inventory']['quantity'] = 5
    market_data_low['pricing_date'] = datetime.now()
    decision_low = orchestrator.calculate_price(product, market_data_low)
    print(f"\nLow Inventory (5 units):")
    print(f"  Price adjustment: {decision_low.price_change_pct:+.1f}%")
    print(f"  Final price: ${decision_low.final_price:.2f}")
    
    # High inventory
    market_data_high = market_data.copy()
    market_data_high['inventory']['quantity'] = 200
    market_data_high['pricing_date'] = datetime.now()
    decision_high = orchestrator.calculate_price(product, market_data_high)
    print(f"\nHigh Inventory (200 units):")
    print(f"  Price adjustment: {decision_high.price_change_pct:+.1f}%")
    print(f"  Final price: ${decision_high.final_price:.2f}")
    
    # Generate daily report
    print("\n=== Daily Report ===")
    report = audit_logger.generate_daily_report()
    print(f"Date: {report['date']}")
    print(f"Total price changes: {report['total_changes']}")
    print(f"Average confidence: {report['confidence_avg']:.2%}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_pricing_engine()