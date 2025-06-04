"""
Demand Forecasting Models for Cannabis Products
Time-series analysis with seasonal patterns and event impact
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class DemandForecaster:
    """Multi-model demand forecasting system"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.seasonal_patterns = {}
        self.event_impacts = {
            'holiday': 1.35,
            'weekend': 1.15,
            'payday': 1.20,
            '420': 2.50,
            'green_wednesday': 1.80,
            'new_year': 1.40
        }
        
    def train_product_model(self, product_id: str, sales_history: pd.DataFrame) -> Dict:
        """Train forecasting model for specific product"""
        
        # Prepare time series data
        ts_data = self._prepare_timeseries(sales_history)
        
        # Detect seasonality
        seasonality = self._detect_seasonality(ts_data)
        self.seasonal_patterns[product_id] = seasonality
        
        # Train ARIMA model for base demand
        arima_model = self._train_arima(ts_data)
        
        # Train RF model for complex patterns
        rf_model, scaler = self._train_random_forest(sales_history)
        
        self.models[product_id] = {
            'arima': arima_model,
            'rf': rf_model,
            'last_update': datetime.now()
        }
        self.scalers[product_id] = scaler
        
        # Calculate model performance
        performance = self._evaluate_model(product_id, sales_history)
        
        return {
            'product_id': product_id,
            'model_type': 'ensemble',
            'seasonality': seasonality,
            'performance': performance
        }
    
    def forecast_demand(self, product_id: str, horizon_days: int = 7) -> pd.DataFrame:
        """Generate demand forecast for product"""
        
        if product_id not in self.models:
            raise ValueError(f"No model trained for product {product_id}")
        
        # Generate base forecast with ARIMA
        arima_forecast = self._arima_forecast(product_id, horizon_days)
        
        # Generate feature-based forecast with RF
        rf_forecast = self._rf_forecast(product_id, horizon_days)
        
        # Ensemble predictions
        forecast = self._ensemble_forecast(arima_forecast, rf_forecast)
        
        # Apply event adjustments
        forecast = self._apply_event_impacts(forecast)
        
        return forecast
    
    def _prepare_timeseries(self, sales_history: pd.DataFrame) -> pd.Series:
        """Prepare time series data from sales history"""
        
        # Aggregate daily sales
        daily_sales = sales_history.groupby(pd.Grouper(key='date', freq='D'))['quantity'].sum()
        
        # Fill missing dates
        date_range = pd.date_range(start=daily_sales.index.min(), 
                                   end=daily_sales.index.max(), 
                                   freq='D')
        daily_sales = daily_sales.reindex(date_range, fill_value=0)
        
        # Apply smoothing
        daily_sales = daily_sales.rolling(window=3, center=True).mean().fillna(daily_sales)
        
        return daily_sales
    
    def _detect_seasonality(self, ts_data: pd.Series) -> Dict:
        """Detect seasonal patterns in time series"""
        
        if len(ts_data) < 60:  # Need at least 2 months of data
            return {'type': 'none', 'period': 0}
        
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(ts_data, model='additive', period=7)
        
        # Calculate seasonality strength
        seasonal_strength = np.std(decomposition.seasonal) / np.std(ts_data)
        
        # Detect weekly patterns
        weekly_pattern = ts_data.groupby(ts_data.index.dayofweek).mean()
        weekly_strength = np.std(weekly_pattern) / np.mean(weekly_pattern)
        
        return {
            'type': 'weekly' if seasonal_strength > 0.1 else 'none',
            'period': 7,
            'strength': seasonal_strength,
            'weekly_pattern': weekly_pattern.to_dict(),
            'peak_days': weekly_pattern.nlargest(2).index.tolist()
        }
    
    def _train_arima(self, ts_data: pd.Series) -> ARIMA:
        """Train ARIMA model for time series"""
        
        # Auto-select best parameters
        best_aic = np.inf
        best_params = None
        
        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):
                    try:
                        model = ARIMA(ts_data, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_params = (p, d, q)
                    except:
                        continue
        
        # Train final model
        final_model = ARIMA(ts_data, order=best_params or (1, 1, 1))
        return final_model.fit()
    
    def _train_random_forest(self, sales_history: pd.DataFrame) -> Tuple[RandomForestRegressor, StandardScaler]:
        """Train Random Forest for complex pattern recognition"""
        
        # Feature engineering
        features = pd.DataFrame()
        features['day_of_week'] = sales_history['date'].dt.dayofweek
        features['day_of_month'] = sales_history['date'].dt.day
        features['month'] = sales_history['date'].dt.month
        features['quarter'] = sales_history['date'].dt.quarter
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        features['is_month_start'] = (features['day_of_month'] <= 5).astype(int)
        features['is_month_end'] = (features['day_of_month'] >= 25).astype(int)
        
        # Add lag features
        for lag in [1, 3, 7, 14]:
            features[f'lag_{lag}'] = sales_history['quantity'].shift(lag)
        
        # Remove NaN rows
        features = features.dropna()
        target = sales_history.loc[features.index, 'quantity']
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Train model
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        rf_model.fit(features_scaled, target)
        
        return rf_model, scaler
    
    def _arima_forecast(self, product_id: str, horizon: int) -> pd.Series:
        """Generate ARIMA forecast"""
        
        model = self.models[product_id]['arima']
        forecast = model.forecast(steps=horizon)
        
        # Ensure non-negative
        forecast = forecast.clip(lower=0)
        
        return forecast
    
    def _rf_forecast(self, product_id: str, horizon: int) -> pd.Series:
        """Generate Random Forest forecast"""
        
        model = self.models[product_id]['rf']
        scaler = self.scalers[product_id]
        
        # Generate future features
        future_dates = pd.date_range(start=datetime.now(), periods=horizon, freq='D')
        features = pd.DataFrame()
        features['day_of_week'] = future_dates.dayofweek
        features['day_of_month'] = future_dates.day
        features['month'] = future_dates.month
        features['quarter'] = future_dates.quarter
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        features['is_month_start'] = (features['day_of_month'] <= 5).astype(int)
        features['is_month_end'] = (features['day_of_month'] >= 25).astype(int)
        
        # For lag features, use recent actuals (simplified)
        for lag in [1, 3, 7, 14]:
            features[f'lag_{lag}'] = 0  # Would use actual recent data in production
        
        # Scale and predict
        features_scaled = scaler.transform(features)
        forecast = model.predict(features_scaled)
        
        return pd.Series(forecast, index=future_dates)
    
    def _ensemble_forecast(self, arima_forecast: pd.Series, rf_forecast: pd.Series) -> pd.DataFrame:
        """Combine forecasts with confidence intervals"""
        
        # Weighted average (ARIMA more reliable for trend, RF for patterns)
        mean_forecast = 0.6 * arima_forecast + 0.4 * rf_forecast
        
        # Calculate confidence intervals
        forecast_std = np.abs(arima_forecast - rf_forecast) * 0.5
        
        forecast_df = pd.DataFrame({
            'date': arima_forecast.index,
            'forecast': mean_forecast,
            'lower_bound': mean_forecast - 1.96 * forecast_std,
            'upper_bound': mean_forecast + 1.96 * forecast_std,
            'confidence': 1 / (1 + forecast_std)
        })
        
        return forecast_df
    
    def _apply_event_impacts(self, forecast: pd.DataFrame) -> pd.DataFrame:
        """Apply known event impacts to forecast"""
        
        forecast = forecast.copy()
        
        for idx, row in forecast.iterrows():
            date = row['date']
            
            # Check for events
            if date.weekday() >= 5:  # Weekend
                forecast.loc[idx, 'forecast'] *= self.event_impacts['weekend']
            
            if date.day in [1, 15]:  # Payday
                forecast.loc[idx, 'forecast'] *= self.event_impacts['payday']
            
            if date.month == 4 and date.day == 20:  # 420
                forecast.loc[idx, 'forecast'] *= self.event_impacts['420']
            
            # Add event flag
            forecast.loc[idx, 'event_adjustment'] = forecast.loc[idx, 'forecast'] / row['forecast']
        
        return forecast
    
    def _evaluate_model(self, product_id: str, test_data: pd.DataFrame) -> Dict:
        """Evaluate model performance"""
        
        # Simple validation (would do proper train/test split in production)
        recent_data = test_data.tail(30)
        
        # Generate predictions for validation period
        predictions = self.forecast_demand(product_id, len(recent_data))
        
        # Calculate metrics
        actual = recent_data['quantity'].values
        predicted = predictions['forecast'].values[:len(actual)]
        
        mape = np.mean(np.abs((actual - predicted) / (actual + 1))) * 100
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        return {
            'mape': mape,
            'rmse': rmse,
            'accuracy': max(0, 100 - mape)
        }


class CategoryForecaster:
    """Aggregate forecasting at category level"""
    
    def __init__(self, demand_forecaster: DemandForecaster):
        self.demand_forecaster = demand_forecaster
        self.category_patterns = {}
        
    def forecast_category(self, category: str, products: List[str], horizon: int = 7) -> pd.DataFrame:
        """Forecast demand for entire category"""
        
        category_forecast = None
        
        for product_id in products:
            try:
                product_forecast = self.demand_forecaster.forecast_demand(product_id, horizon)
                
                if category_forecast is None:
                    category_forecast = product_forecast.copy()
                    category_forecast.rename(columns={'forecast': 'total_forecast'}, inplace=True)
                else:
                    category_forecast['total_forecast'] += product_forecast['forecast']
            except:
                continue
        
        # Add category-level insights
        if category_forecast is not None:
            category_forecast['category'] = category
            category_forecast['product_count'] = len(products)
            
        return category_forecast
    
    def identify_trending_products(self, forecasts: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Identify products with increasing demand trends"""
        
        trending = []
        
        for product_id, forecast in forecasts.items():
            # Calculate trend
            trend = np.polyfit(range(len(forecast)), forecast['forecast'], 1)[0]
            
            if trend > 0:
                trending.append({
                    'product_id': product_id,
                    'trend_strength': trend,
                    'forecast_increase': forecast['forecast'].iloc[-1] / forecast['forecast'].iloc[0] - 1
                })
        
        return sorted(trending, key=lambda x: x['trend_strength'], reverse=True)