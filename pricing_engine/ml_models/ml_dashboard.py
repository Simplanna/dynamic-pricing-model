"""
ML Model Performance Dashboard
Streamlit dashboard for monitoring ML model performance and predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Import ML models
from forecasting.demand_forecaster import DemandForecaster, CategoryForecaster
from optimization.price_elasticity import PriceElasticityLearner, DynamicPricingOptimizer
from optimization.competitive_response import CompetitiveResponsePredictor
from optimization.reinforcement_learning import PriceOptimizationRL
from anomaly_detection.anomaly_detector import MarketAnomalyDetector, DataQualityMonitor
from ensemble_system import EnsemblePricingSystem, MLRuleIntegrator


def render_ml_dashboard():
    """Render ML model performance dashboard"""
    
    st.title("🤖 Machine Learning Model Dashboard")
    
    # Initialize models (would load from saved state in production)
    if 'ml_models' not in st.session_state:
        st.session_state.ml_models = {
            'demand_forecaster': DemandForecaster(),
            'elasticity_learner': PriceElasticityLearner(),
            'competitive_predictor': CompetitiveResponsePredictor(),
            'rl_optimizer': PriceOptimizationRL(),
            'anomaly_detector': MarketAnomalyDetector(),
            'ensemble_system': EnsemblePricingSystem()
        }
    
    # Dashboard tabs
    tabs = st.tabs([
        "📊 Model Performance",
        "📈 Demand Forecasting", 
        "💰 Price Optimization",
        "🏁 A/B Testing",
        "⚠️ Anomaly Detection",
        "🎯 Ensemble Decisions"
    ])
    
    with tabs[0]:
        render_model_performance()
    
    with tabs[1]:
        render_demand_forecasting()
    
    with tabs[2]:
        render_price_optimization()
    
    with tabs[3]:
        render_ab_testing()
    
    with tabs[4]:
        render_anomaly_detection()
    
    with tabs[5]:
        render_ensemble_decisions()


def render_model_performance():
    """Render overall model performance metrics"""
    
    st.header("Model Performance Overview")
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Demand Forecast Accuracy",
            "85.3%",
            "↑ 2.1%",
            help="MAPE accuracy for 7-day forecasts"
        )
    
    with col2:
        st.metric(
            "Price Optimization Lift",
            "+12.4%",
            "↑ 1.8%",
            help="Average revenue lift from ML pricing"
        )
    
    with col3:
        st.metric(
            "Anomaly Detection Rate",
            "94.2%",
            "↑ 0.5%",
            help="True positive rate for anomalies"
        )
    
    with col4:
        st.metric(
            "Ensemble Confidence",
            "78.5%",
            "↓ 1.2%",
            help="Average decision confidence"
        )
    
    # Model performance over time
    st.subheader("Model Performance Trends")
    
    # Generate sample performance data
    dates = pd.date_range(start='2024-01-01', end='2024-06-04', freq='W')
    performance_data = pd.DataFrame({
        'date': dates,
        'demand_accuracy': 80 + np.random.randn(len(dates)).cumsum() * 0.5,
        'price_optimization': 8 + np.random.randn(len(dates)).cumsum() * 0.3,
        'anomaly_precision': 90 + np.random.randn(len(dates)).cumsum() * 0.2
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=performance_data['date'],
        y=performance_data['demand_accuracy'],
        mode='lines+markers',
        name='Demand Forecast Accuracy',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=performance_data['date'],
        y=performance_data['price_optimization'],
        mode='lines+markers',
        name='Price Optimization Lift %',
        line=dict(color='green', width=2),
        yaxis='y2'
    ))
    
    fig.add_trace(go.Scatter(
        x=performance_data['date'],
        y=performance_data['anomaly_precision'],
        mode='lines+markers',
        name='Anomaly Detection Precision',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title='ML Model Performance Over Time',
        xaxis_title='Date',
        yaxis_title='Accuracy %',
        yaxis2=dict(
            title='Optimization Lift %',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model health status
    st.subheader("Model Health Status")
    
    health_data = {
        'Model': ['Demand Forecaster', 'Price Elasticity', 'Competitive Response', 
                  'Reinforcement Learning', 'Anomaly Detection'],
        'Status': ['🟢 Healthy', '🟢 Healthy', '🟡 Warning', '🟢 Healthy', '🟢 Healthy'],
        'Last Update': ['2 hours ago', '1 hour ago', '12 hours ago', '30 mins ago', '1 hour ago'],
        'Data Points': [50000, 35000, 15000, 8000, 45000],
        'Action': ['No action', 'No action', 'Needs retraining', 'No action', 'No action']
    }
    
    st.dataframe(pd.DataFrame(health_data), use_container_width=True)


def render_demand_forecasting():
    """Render demand forecasting visualizations"""
    
    st.header("Demand Forecasting")
    
    # Product selector
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_product = st.selectbox(
            "Select Product",
            ["Blue Dream - Flower", "Sour Diesel - Flower", "OG Kush - Concentrate", 
             "Gummy Bears - Edibles"]
        )
    
    with col2:
        forecast_horizon = st.slider("Forecast Days", 1, 30, 7)
    
    # Generate sample forecast
    dates = pd.date_range(start=datetime.now(), periods=forecast_horizon, freq='D')
    historical_dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    # Historical data
    historical_demand = 100 + np.random.randn(30).cumsum() * 5
    
    # Forecast with confidence intervals
    base_forecast = historical_demand[-1] + np.random.randn(forecast_horizon).cumsum() * 3
    upper_bound = base_forecast + np.random.rand(forecast_horizon) * 20
    lower_bound = base_forecast - np.random.rand(forecast_horizon) * 20
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_dates,
        y=historical_demand,
        mode='lines',
        name='Historical Demand',
        line=dict(color='blue', width=2)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=dates,
        y=base_forecast,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='green', width=2, dash='dash')
    ))
    
    # Confidence intervals
    fig.add_trace(go.Scatter(
        x=list(dates) + list(dates[::-1]),
        y=list(upper_bound) + list(lower_bound[::-1]),
        fill='toself',
        fillcolor='rgba(0,255,0,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence',
        showlegend=True
    ))
    
    # Events
    fig.add_vline(x=dates[4], line_dash="dot", line_color="red", 
                  annotation_text="Weekend")
    
    fig.update_layout(
        title=f'Demand Forecast: {selected_product}',
        xaxis_title='Date',
        yaxis_title='Units',
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Expected Weekly Demand", f"{int(base_forecast.sum())} units")
    
    with col2:
        st.metric("Peak Day", "Saturday", "↑ 35% vs avg")
    
    with col3:
        st.metric("Trend", "Increasing", "↑ 8.2% week-over-week")
    
    # Category forecast
    st.subheader("Category Demand Patterns")
    
    categories = ['Flower', 'Edibles', 'Concentrates', 'Pre-rolls']
    category_demand = pd.DataFrame({
        'Category': categories,
        'Current': [450, 320, 180, 250],
        'Forecast': [480, 340, 195, 265],
        'Change': ['+6.7%', '+6.3%', '+8.3%', '+6.0%']
    })
    
    fig_cat = px.bar(
        category_demand, 
        x='Category', 
        y=['Current', 'Forecast'],
        title='Category Demand Forecast',
        barmode='group'
    )
    
    st.plotly_chart(fig_cat, use_container_width=True)


def render_price_optimization():
    """Render price optimization analysis"""
    
    st.header("Price Optimization")
    
    # Elasticity analysis
    st.subheader("Price Elasticity Analysis")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_product = st.selectbox(
            "Select Product for Analysis",
            ["Blue Dream - Flower", "Sour Diesel - Flower", "Gummy Bears - Edibles"],
            key="elasticity_product"
        )
    
    with col2:
        current_price = st.number_input("Current Price", value=45.00, step=1.00)
    
    # Generate elasticity curve
    prices = np.linspace(current_price * 0.7, current_price * 1.3, 50)
    elasticity = -1.2  # Sample elasticity
    base_quantity = 100
    quantities = base_quantity * (prices / current_price) ** elasticity
    revenues = prices * quantities
    
    # Find optimal price
    optimal_idx = np.argmax(revenues)
    optimal_price = prices[optimal_idx]
    
    fig = go.Figure()
    
    # Revenue curve
    fig.add_trace(go.Scatter(
        x=prices,
        y=revenues,
        mode='lines',
        name='Revenue',
        line=dict(color='green', width=3)
    ))
    
    # Quantity curve (secondary axis)
    fig.add_trace(go.Scatter(
        x=prices,
        y=quantities,
        mode='lines',
        name='Quantity',
        line=dict(color='blue', width=2),
        yaxis='y2'
    ))
    
    # Mark current and optimal
    fig.add_vline(x=current_price, line_dash="dash", line_color="gray",
                  annotation_text="Current")
    fig.add_vline(x=optimal_price, line_dash="dash", line_color="red",
                  annotation_text="Optimal")
    
    fig.update_layout(
        title=f'Price-Revenue Curve: {selected_product}',
        xaxis_title='Price ($)',
        yaxis_title='Revenue ($)',
        yaxis2=dict(title='Quantity', overlaying='y', side='right'),
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Optimization recommendation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Optimal Price",
            f"${optimal_price:.2f}",
            f"{((optimal_price - current_price) / current_price * 100):+.1f}%"
        )
    
    with col2:
        revenue_lift = (revenues[optimal_idx] - revenues[25]) / revenues[25] * 100
        st.metric("Expected Revenue Lift", f"{revenue_lift:+.1f}%")
    
    with col3:
        st.metric("Price Elasticity", f"{elasticity:.2f}", 
                 help="Negative value indicates normal demand curve")
    
    # Competitive response prediction
    st.subheader("Competitive Response Prediction")
    
    response_data = {
        'Competitor': ['GreenLeaf', 'CannaKing', 'BudHub'],
        'Expected Response': ['Match within 24h', 'No response', 'Partial follow (50%)'],
        'Confidence': ['85%', '72%', '68%'],
        'Impact': ['High', 'Low', 'Medium']
    }
    
    st.dataframe(pd.DataFrame(response_data), use_container_width=True)


def render_ab_testing():
    """Render A/B testing dashboard"""
    
    st.header("A/B Testing & Experimentation")
    
    # Active experiments
    st.subheader("Active Price Experiments")
    
    experiments = {
        'Product': ['Blue Dream', 'Sour Diesel', 'OG Kush'],
        'Status': ['🟢 Active', '🟢 Active', '🟡 Concluding'],
        'Progress': ['45%', '72%', '95%'],
        'Best Price': ['$48.00', '$52.00', '$65.00'],
        'Confidence': ['68%', '82%', '91%'],
        'Revenue Lift': ['+5.2%', '+8.7%', '+12.3%']
    }
    
    exp_df = pd.DataFrame(experiments)
    st.dataframe(exp_df, use_container_width=True)
    
    # Detailed experiment view
    selected_exp = st.selectbox("View Experiment Details", experiments['Product'])
    
    if selected_exp:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Price performance chart
            test_prices = [45, 48, 50, 52, 55]
            test_revenues = [4500, 4800, 4900, 4700, 4400]
            test_samples = [120, 135, 128, 142, 98]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=test_prices,
                y=test_revenues,
                name='Revenue',
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Scatter(
                x=test_prices,
                y=test_samples,
                mode='lines+markers',
                name='Sample Size',
                yaxis='y2',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title=f'Experiment Results: {selected_exp}',
                xaxis_title='Test Price ($)',
                yaxis_title='Revenue ($)',
                yaxis2=dict(title='Samples', overlaying='y', side='right'),
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Recommended Price", "$48.00")
            st.metric("Expected Lift", "+8.7%")
            st.metric("Statistical Significance", "92%")
            
            if st.button("🎯 Implement Winner", type="primary"):
                st.success("Price updated successfully!")
    
    # Experiment history
    st.subheader("Completed Experiments")
    
    history_data = {
        'Product': ['Gummy Bears', 'Vape Pen', 'Pre-rolls'],
        'Completion': ['2024-05-28', '2024-05-20', '2024-05-15'],
        'Winner': ['$25 (+15%)', '$35 (+8%)', '$12 (+5%)'],
        'Implemented': ['✅', '✅', '❌']
    }
    
    st.dataframe(pd.DataFrame(history_data), use_container_width=True)


def render_anomaly_detection():
    """Render anomaly detection alerts"""
    
    st.header("Anomaly Detection")
    
    # Current alerts
    alert_counts = {
        'Critical': 2,
        'Warning': 5,
        'Info': 12
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔴 Critical Alerts", alert_counts['Critical'])
    
    with col2:
        st.metric("🟡 Warnings", alert_counts['Warning'])
    
    with col3:
        st.metric("🔵 Info", alert_counts['Info'])
    
    # Alert details
    st.subheader("Active Anomalies")
    
    alerts = {
        'Time': ['10:32 AM', '10:15 AM', '9:45 AM', '9:30 AM'],
        'Type': ['Price Spike', 'Volume Drop', 'Competitor Deviation', 'Data Quality'],
        'Severity': ['🔴 Critical', '🔴 Critical', '🟡 Warning', '🟡 Warning'],
        'Details': [
            'Blue Dream price 35% above baseline',
            'Edibles volume down 60% vs normal',
            'Competitor pricing 25% below market',
            'Missing inventory data for 3 products'
        ],
        'Action': ['Review', 'Investigate', 'Monitor', 'Fix Data']
    }
    
    alert_df = pd.DataFrame(alerts)
    st.dataframe(alert_df, use_container_width=True)
    
    # Anomaly visualization
    st.subheader("Market Anomaly Patterns")
    
    # Time series with anomalies
    hours = list(range(24))
    normal_pattern = 50 + 30 * np.sin(np.array(hours) * np.pi / 12)
    actual_pattern = normal_pattern.copy()
    
    # Add anomalies
    anomaly_hours = [10, 15, 20]
    for h in anomaly_hours:
        actual_pattern[h] += np.random.choice([-30, 40])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hours,
        y=normal_pattern,
        mode='lines',
        name='Expected Pattern',
        line=dict(color='gray', dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=hours,
        y=actual_pattern,
        mode='lines+markers',
        name='Actual',
        line=dict(color='blue')
    ))
    
    # Mark anomalies
    for h in anomaly_hours:
        fig.add_trace(go.Scatter(
            x=[h],
            y=[actual_pattern[h]],
            mode='markers',
            marker=dict(color='red', size=12, symbol='x'),
            name='Anomaly' if h == anomaly_hours[0] else None,
            showlegend=(h == anomaly_hours[0])
        ))
    
    fig.update_layout(
        title='Hourly Sales Pattern with Anomalies',
        xaxis_title='Hour of Day',
        yaxis_title='Sales Volume',
        height=350
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_ensemble_decisions():
    """Render ensemble decision making"""
    
    st.header("Ensemble Decision System")
    
    # Decision confidence distribution
    st.subheader("Decision Confidence Analysis")
    
    # Sample confidence scores
    confidence_scores = np.random.beta(8, 3, 1000)
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=confidence_scores,
        nbinsx=30,
        name='Decision Confidence',
        marker_color='lightblue'
    ))
    
    fig.add_vline(x=0.7, line_dash="dash", line_color="red",
                  annotation_text="Threshold")
    
    fig.update_layout(
        title='Distribution of Ensemble Decision Confidence',
        xaxis_title='Confidence Score',
        yaxis_title='Number of Decisions',
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent decisions
    st.subheader("Recent Pricing Decisions")
    
    decisions = {
        'Product': ['Blue Dream', 'Sour Diesel', 'Gummy Bears', 'OG Kush'],
        'Rule Price': ['$45.00', '$50.00', '$25.00', '$60.00'],
        'ML Price': ['$48.50', '$52.00', '$24.00', '$65.00'],
        'Final Price': ['$46.50', '$51.00', '$24.50', '$62.00'],
        'Confidence': ['82%', '78%', '91%', '75%'],
        'Primary Factor': ['Demand ↑', 'Competition', 'Elasticity', 'Inventory ↓']
    }
    
    decision_df = pd.DataFrame(decisions)
    st.dataframe(decision_df, use_container_width=True)
    
    # Factor influence
    st.subheader("Decision Factor Influence")
    
    factors = ['Rules', 'Demand', 'Elasticity', 'Competition', 'RL/Testing']
    influence = [40, 20, 15, 15, 10]
    
    fig = px.pie(
        values=influence,
        names=factors,
        title='Average Factor Influence on Pricing Decisions'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Decision explanation
    st.subheader("Decision Explainability")
    
    selected_decision = st.selectbox(
        "Select decision to explain",
        decisions['Product']
    )
    
    if selected_decision:
        st.info(f"""
        **Decision Explanation for {selected_decision}:**
        
        1. **Rule-based factors** (40% influence): Inventory aging suggests slight discount needed
        2. **Demand forecast** (25% influence): Weekend demand expected to increase by 35%
        3. **Price elasticity** (20% influence): Product shows moderate elasticity (-1.2)
        4. **Competition** (15% influence): Competitor prices stable, no immediate threat
        
        **Final Decision**: Price set at $46.50 with 82% confidence
        **Expected Outcome**: +8.5% revenue increase
        """)


if __name__ == "__main__":
    # This would be imported and called from main dashboard
    render_ml_dashboard()