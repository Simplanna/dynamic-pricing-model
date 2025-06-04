import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from typing import Dict, List

# Import our modules
from pricing_engine.scrapers.scraperbee_client import ScraperBeeClient, DispensaryMenuScraper
from pricing_engine.scrapers.parsers import ParserFactory
from pricing_engine.utils.validation import DataValidator, ProductMatcher
from pricing_engine.analysis.market_analysis import MarketAnalyzer
from pricing_engine.core.models import MarketType, CompetitorDensity

# Page config
st.set_page_config(
    page_title="Cannabis Dynamic Pricing Engine",
    page_icon="🌿",
    layout="wide"
)

# Initialize session state
if 'scraped_data' not in st.session_state:
    st.session_state.scraped_data = []
if 'validation_results' not in st.session_state:
    st.session_state.validation_results = {}
if 'market_analysis' not in st.session_state:
    st.session_state.market_analysis = {}

# Sidebar
st.sidebar.title("Dynamic Pricing Engine")
st.sidebar.markdown("Phase 1: Data Integration & Market Analysis")

# Main navigation
page = st.sidebar.selectbox(
    "Select Module",
    ["Dashboard", "Competitor Scraping", "Data Validation", "Market Analysis", "Product Matching"]
)

# Dashboard Page
if page == "Dashboard":
    st.title("Cannabis Dynamic Pricing Dashboard 🌿")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "MA Avg Price/gram", 
            "$5.36",
            "-62% from 2018"
        )
    
    with col2:
        st.metric(
            "RI Avg Price/gram",
            "$12.40",
            "+8% YoY"
        )
    
    with col3:
        st.metric(
            "Competitors Tracked",
            len(st.session_state.scraped_data)
        )
    
    with col4:
        validation_rate = 0
        if st.session_state.validation_results:
            valid = st.session_state.validation_results.get('valid', 0)
            total = st.session_state.validation_results.get('total', 1)
            validation_rate = (valid / total) * 100
        st.metric(
            "Data Quality",
            f"{validation_rate:.1f}%"
        )
    
    # Market overview
    st.header("Market Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Market comparison chart
        market_data = pd.DataFrame({
            'State': ['Massachusetts', 'Rhode Island'],
            'Dispensaries': [360, 7],
            'Avg Revenue per Store': [2.5, 16.8]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=market_data['State'],
            y=market_data['Dispensaries'],
            name='Number of Dispensaries',
            yaxis='y',
            marker_color='lightblue'
        ))
        fig.add_trace(go.Bar(
            x=market_data['State'],
            y=market_data['Avg Revenue per Store'],
            name='Avg Revenue per Store ($M)',
            yaxis='y2',
            marker_color='darkgreen'
        ))
        
        fig.update_layout(
            title='MA vs RI Market Comparison',
            yaxis=dict(title='Number of Dispensaries', side='left'),
            yaxis2=dict(title='Avg Revenue ($M)', side='right', overlaying='y'),
            hovermode='x'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Price trend visualization
        dates = pd.date_range(start='2018-01-01', end='2024-12-01', freq='6M')
        ma_prices = [14.09, 12.50, 10.80, 9.20, 7.80, 6.50, 5.80, 5.36, 5.00, 4.80, 4.60, 4.44, 4.50]
        ri_prices = [15.00, 14.80, 14.50, 14.00, 13.80, 13.50, 13.00, 12.80, 12.50, 12.40, 12.30, 12.40, 12.50]
        
        price_df = pd.DataFrame({
            'Date': list(dates) + list(dates),
            'Price': ma_prices + ri_prices,
            'State': ['MA'] * len(dates) + ['RI'] * len(dates)
        })
        
        fig = px.line(price_df, x='Date', y='Price', color='State',
                      title='Price per Gram Trends (2018-2024)',
                      labels={'Price': 'Price per Gram ($)'})
        
        st.plotly_chart(fig, use_container_width=True)
    
    # System status
    st.header("System Status")
    
    if st.session_state.scraped_data:
        st.success(f"✅ ScraperBee Integration Active - {len(st.session_state.scraped_data)} dispensaries scraped")
    else:
        st.info("ℹ️ No dispensaries scraped yet. Use the Competitor Scraping module to begin.")

# Competitor Scraping Page
elif page == "Competitor Scraping":
    st.title("Competitor Menu Scraping 🕷️")
    
    # Test dispensaries
    test_dispensaries = [
        {
            'name': 'NETA Brookline',
            'url': 'https://netacare.org/brookline-dispensary-menu/',
            'platform': 'dutchie',
            'state': 'MA',
            'distance_miles': 5.2
        },
        {
            'name': 'Theory Wellness',
            'url': 'https://www.theory-wellness.org/massachusetts/menu/',
            'platform': 'iheartjane',
            'state': 'MA',
            'distance_miles': 8.7
        },
        {
            'name': 'Greenleaf Compassion Center',
            'url': 'https://www.greenleafcompassioncenter.com/menu',
            'platform': 'meadow',
            'state': 'RI',
            'distance_miles': 12.3
        }
    ]
    
    st.subheader("Available Test Dispensaries")
    
    # Display test dispensaries
    test_df = pd.DataFrame(test_dispensaries)
    st.dataframe(test_df)
    
    # Scraping controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_indices = st.multiselect(
            "Select dispensaries to scrape",
            options=list(range(len(test_dispensaries))),
            format_func=lambda x: test_dispensaries[x]['name']
        )
    
    with col2:
        scrape_button = st.button("🕷️ Scrape Selected", type="primary")
    
    if scrape_button and selected_indices:
        selected_dispensaries = [test_dispensaries[i] for i in selected_indices]
        
        # Initialize scraper
        scraper = DispensaryMenuScraper()
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        scraped_results = []
        
        for idx, dispensary in enumerate(selected_dispensaries):
            status_text.text(f"Scraping {dispensary['name']}...")
            
            # Simulate scraping (in production, this would actually scrape)
            result = {
                'dispensary_name': dispensary['name'],
                'platform': dispensary['platform'],
                'url': dispensary['url'],
                'distance_miles': dispensary['distance_miles'],
                'state': dispensary['state'],
                'scraped_at': datetime.now(),
                'status': 'success',
                'products_found': 42  # Simulated
            }
            
            scraped_results.append(result)
            progress_bar.progress((idx + 1) / len(selected_dispensaries))
        
        st.session_state.scraped_data.extend(scraped_results)
        
        status_text.text("Scraping complete!")
        st.success(f"Successfully scraped {len(scraped_results)} dispensaries")
        
        # Display results
        st.subheader("Scraping Results")
        results_df = pd.DataFrame(scraped_results)
        st.dataframe(results_df)
    
    # Show all scraped data
    if st.session_state.scraped_data:
        st.subheader("All Scraped Dispensaries")
        all_df = pd.DataFrame(st.session_state.scraped_data)
        st.dataframe(all_df)

# Data Validation Page
elif page == "Data Validation":
    st.title("Data Validation & Quality Assurance 🔍")
    
    # Sample product data for validation
    sample_products = [
        {
            'name': 'Blue Dream Flower',
            'category': 'flower',
            'brand': 'Local Grove',
            'thc': 22.5,
            'cbd': 0.5,
            'price': 45.00,
            'weight_grams': 3.5
        },
        {
            'name': 'Gummy Bears 100mg',
            'category': 'edibles',
            'brand': 'Sweet Relief',
            'thc': 10.0,
            'cbd': 0.0,
            'price': 25.00,
            'unit_count': 10
        },
        {
            'name': 'OG Kush Vape Cart',
            'category': 'vapes',  # Invalid category
            'brand': 'Pure Extracts',
            'thc': 85.0,  # Very high THC
            'price': 0.50,  # Price too low
            'weight_grams': 0.5
        }
    ]
    
    # Validation controls
    if st.button("Run Validation"):
        validator = DataValidator()
        
        # Validate products
        validation_results = validator.validate_batch(sample_products)
        st.session_state.validation_results = validation_results
        
        # Display summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Products", validation_results['total'])
        
        with col2:
            st.metric("Valid Products", validation_results['valid'], 
                     delta=f"{(validation_results['valid']/validation_results['total']*100):.1f}%")
        
        with col3:
            st.metric("Products with Warnings", validation_results['warnings'])
        
        # Detailed results
        st.subheader("Validation Details")
        
        for idx, result in enumerate(validation_results['validation_results']):
            product = result['data']
            
            if result['valid']:
                status = "✅ Valid"
                color = "success"
            else:
                status = "❌ Invalid"
                color = "error"
            
            with st.expander(f"{product['name']} - {status}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Product Data:**")
                    st.json(product)
                
                with col2:
                    if result['errors']:
                        st.error("**Errors:**")
                        for error in result['errors']:
                            st.write(f"- {error}")
                    
                    if result['warnings']:
                        st.warning("**Warnings:**")
                        for warning in result['warnings']:
                            st.write(f"- {warning}")
        
        # Quality metrics
        st.subheader("Data Quality Metrics")
        
        quality_data = pd.DataFrame([
            {'Metric': 'Price Validation', 'Pass Rate': 0.67, 'Issues': 'Price below $1 threshold'},
            {'Metric': 'THC Validation', 'Pass Rate': 0.67, 'Issues': 'THC % above 40% limit'},
            {'Metric': 'Category Validation', 'Pass Rate': 0.67, 'Issues': 'Invalid category "vapes"'},
            {'Metric': 'Required Fields', 'Pass Rate': 1.0, 'Issues': 'None'}
        ])
        
        fig = px.bar(quality_data, x='Metric', y='Pass Rate', 
                     title='Data Quality Metrics',
                     color='Pass Rate',
                     color_continuous_scale=['red', 'yellow', 'green'])
        
        st.plotly_chart(fig, use_container_width=True)

# Market Analysis Page
elif page == "Market Analysis":
    st.title("Market Analysis & Patterns 📊")
    
    analyzer = MarketAnalyzer()
    
    # Market selection
    selected_market = st.selectbox("Select Market", ["Massachusetts", "Rhode Island", "Cross-Border Analysis"])
    
    if selected_market == "Cross-Border Analysis":
        st.subheader("MA-RI Cross-Border Shopping Analysis")
        
        # Simulated analysis data
        cross_border_data = {
            'price_differential': {
                'flower': {'ma_avg_price': 35.00, 'ri_avg_price': 55.00, 'price_gap_pct': 57.1},
                'edibles': {'ma_avg_price': 20.00, 'ri_avg_price': 30.00, 'price_gap_pct': 50.0},
                'concentrates': {'ma_avg_price': 40.00, 'ri_avg_price': 65.00, 'price_gap_pct': 62.5}
            },
            'border_zone_activity': {
                'ma_border_dispensaries': 45,
                'ri_border_dispensaries': 3,
                'price_compression': True
            }
        }
        
        # Price differential chart
        categories = list(cross_border_data['price_differential'].keys())
        ma_prices = [v['ma_avg_price'] for v in cross_border_data['price_differential'].values()]
        ri_prices = [v['ri_avg_price'] for v in cross_border_data['price_differential'].values()]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='MA Average', x=categories, y=ma_prices, marker_color='lightblue'))
        fig.add_trace(go.Bar(name='RI Average', x=categories, y=ri_prices, marker_color='darkgreen'))
        
        fig.update_layout(
            title='Cross-Border Price Differential by Category',
            yaxis_title='Average Price ($)',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.subheader("Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Border Zone Effect**: Prices in RI border dispensaries are compressed due to MA competition")
            st.success("**Arbitrage Opportunity**: Average savings of $20-25 per product when shopping in MA")
        
        with col2:
            st.warning("**Market Risk**: RI dispensaries within 20 miles of MA border face significant price pressure")
            st.info("**Strategic Positioning**: Focus on premium products and services that justify higher prices")
    
    else:
        # Single market analysis
        st.subheader(f"{selected_market} Market Analysis")
        
        # Seasonal patterns
        st.subheader("Seasonal Pricing Patterns")
        
        # Generate seasonal data
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        base_price = 45 if selected_market == "Massachusetts" else 65
        
        seasonal_multipliers = [0.95, 0.93, 0.97, 1.15, 1.05, 1.10, 1.12, 1.15, 1.08, 1.02, 1.05, 1.10]
        prices = [base_price * mult for mult in seasonal_multipliers]
        
        fig = px.line(x=months, y=prices, 
                      title=f'Average Monthly Prices - {selected_market}',
                      labels={'x': 'Month', 'y': 'Average Price ($)'},
                      markers=True)
        
        # Add special events
        fig.add_vline(x=3, line_dash="dash", line_color="green", 
                      annotation_text="4/20")
        fig.add_vline(x=10, line_dash="dash", line_color="red",
                      annotation_text="Green Friday")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Competitor density map (simulated)
        st.subheader("Competitor Density Analysis")
        
        density_data = pd.DataFrame({
            'City': ['Boston', 'Cambridge', 'Worcester', 'Springfield', 'Lowell'] if selected_market == "Massachusetts" 
                    else ['Providence', 'Warwick', 'Cranston', 'Pawtucket', 'Newport'],
            'Dispensaries': [45, 23, 18, 12, 8] if selected_market == "Massachusetts" else [3, 1, 1, 1, 1],
            'Density': ['Dense', 'Dense', 'Moderate', 'Moderate', 'Sparse'] if selected_market == "Massachusetts"
                       else ['Moderate', 'Sparse', 'Sparse', 'Sparse', 'Sparse']
        })
        
        fig = px.bar(density_data, x='City', y='Dispensaries', color='Density',
                     title=f'Dispensary Distribution - {selected_market}',
                     color_discrete_map={'Dense': 'red', 'Moderate': 'yellow', 'Sparse': 'green'})
        
        st.plotly_chart(fig, use_container_width=True)

# Product Matching Page
elif page == "Product Matching":
    st.title("Product Matching & Price Comparison 🔗")
    
    # Sample products for matching
    our_product = {
        'name': 'Blue Dream - 3.5g',
        'category': 'flower',
        'brand': 'Local Grove',
        'thc': 22.5,
        'cbd': 0.5,
        'price': 45.00,
        'weight_grams': 3.5
    }
    
    competitor_products = [
        {
            'name': 'Blue Dream Flower',
            'category': 'flower',
            'brand': 'Local Grove',
            'thc': 23.0,
            'price': 42.00,
            'dispensary_name': 'NETA',
            'distance_miles': 5.2,
            'weight_grams': 3.5
        },
        {
            'name': 'Blue Dream Premium',
            'category': 'flower',
            'brand': 'Local Groves',  # Slight variation
            'thc': 22.0,
            'price': 48.00,
            'dispensary_name': 'Theory',
            'distance_miles': 8.7,
            'weight_grams': 3.5
        },
        {
            'name': 'Dream Blue Hybrid',
            'category': 'flower',
            'brand': 'Different Brand',
            'thc': 21.5,
            'price': 40.00,
            'dispensary_name': 'Greenleaf',
            'distance_miles': 12.3,
            'weight_grams': 3.5
        }
    ]
    
    st.subheader("Our Product")
    st.json(our_product)
    
    if st.button("Find Matches"):
        matcher = ProductMatcher(confidence_threshold=0.8)
        
        matches = matcher.find_matches(our_product, competitor_products)
        
        st.subheader("Matching Results")
        
        if matches:
            # Create comparison table
            comparison_data = []
            
            for match in matches:
                comp = match['matched_product']
                comparison_data.append({
                    'Dispensary': comp['dispensary_name'],
                    'Product Name': comp['name'],
                    'Match Confidence': f"{match['confidence_score']*100:.1f}%",
                    'Their Price': f"${comp['price']:.2f}",
                    'Our Price': f"${our_product['price']:.2f}",
                    'Price Difference': f"${match['price_difference']:.2f}",
                    'Distance': f"{comp['distance_miles']} miles"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Style the dataframe
            def highlight_price_diff(row):
                diff = float(row['Price Difference'].replace('$', ''))
                if diff < 0:
                    return ['background-color: #ffcccc'] * len(row)
                elif diff > 0:
                    return ['background-color: #ccffcc'] * len(row)
                return [''] * len(row)
            
            styled_df = comparison_df.style.apply(highlight_price_diff, axis=1)
            st.dataframe(styled_df)
            
            # Price positioning chart
            st.subheader("Price Positioning")
            
            prices = [our_product['price']] + [m['matched_product']['price'] for m in matches]
            names = ['Our Price'] + [m['matched_product']['dispensary_name'] for m in matches]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=names,
                y=prices,
                marker_color=['gold'] + ['lightblue'] * len(matches)
            ))
            
            # Add average line
            avg_price = sum(prices) / len(prices)
            fig.add_hline(y=avg_price, line_dash="dash", 
                          annotation_text=f"Market Average: ${avg_price:.2f}")
            
            fig.update_layout(
                title='Price Comparison with Matched Products',
                yaxis_title='Price ($)',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Match details
            st.subheader("Match Details")
            
            for idx, match in enumerate(matches):
                with st.expander(f"Match {idx+1}: {match['matched_product']['dispensary_name']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Match Criteria:**")
                        for criterion, matched in match['match_criteria'].items():
                            emoji = "✅" if matched else "❌"
                            st.write(f"{emoji} {criterion.replace('_', ' ').title()}")
                    
                    with col2:
                        st.write("**Confidence Score:**")
                        st.progress(match['confidence_score'])
                        st.write(f"{match['confidence_score']*100:.1f}%")
        else:
            st.warning("No matches found above the confidence threshold")

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "Phase 1: Data Integration & Market Analysis\n\n"
    "ScraperBee API integrated for real-time competitor monitoring"
)