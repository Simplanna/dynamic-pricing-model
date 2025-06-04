"""Market Events Factor - 10% Weight

Manages event-based pricing including 4/20, holidays, and seasonal patterns.
Implements surge pricing and demand prediction for special events.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np


class MarketEventsFactor:
    """Manages market event-based pricing adjustments"""
    
    # Major cannabis events and holidays
    CANNABIS_EVENTS = {
        '04-20': {
            'name': '4/20',
            'type': 'major_cannabis',
            'base_multiplier': 1.75,  # 75% base increase
            'category_multipliers': {
                'flower': 1.2,
                'prerolls': 1.3,
                'edibles': 1.15,
                'concentrates': 1.1,
                'vapes': 1.15,
                'accessories': 1.4
            },
            'ramp_days': 7,  # Start ramping 7 days before
            'peak_days': [-1, 0, 1],  # Day before, day of, day after
        },
        '07-10': {
            'name': '710 (Oil Day)',
            'type': 'cannabis',
            'base_multiplier': 1.25,
            'category_multipliers': {
                'concentrates': 1.4,
                'vapes': 1.3,
                'flower': 1.0,
                'prerolls': 1.0,
                'edibles': 1.1,
                'accessories': 1.2
            },
            'ramp_days': 3,
            'peak_days': [0],
        }
    }
    
    # General holidays
    GENERAL_HOLIDAYS = {
        '01-01': {'name': "New Year's Day", 'multiplier': 1.15},
        '07-04': {'name': 'Independence Day', 'multiplier': 1.20},
        '10-31': {'name': 'Halloween', 'multiplier': 1.15, 'categories': ['edibles']},
        '11-25': {'name': 'Thanksgiving Week', 'multiplier': 1.25, 'duration': 7},
        '12-24': {'name': 'Christmas Week', 'multiplier': 1.30, 'duration': 7},
    }
    
    # Seasonal patterns (by month)
    SEASONAL_PATTERNS = {
        1: 0.90,   # January - post-holiday slowdown
        2: 0.92,   # February
        3: 0.95,   # March
        4: 1.10,   # April - 4/20 month
        5: 1.05,   # May
        6: 1.08,   # June - summer begins
        7: 1.10,   # July - peak summer
        8: 1.08,   # August
        9: 1.00,   # September
        10: 1.02,  # October
        11: 1.05,  # November - holiday prep
        12: 1.10   # December - holidays
    }
    
    # Day of week patterns
    DAY_OF_WEEK_PATTERNS = {
        0: 0.95,   # Monday
        1: 0.93,   # Tuesday
        2: 0.95,   # Wednesday
        3: 1.02,   # Thursday
        4: 1.10,   # Friday
        5: 1.08,   # Saturday
        6: 1.00    # Sunday
    }
    
    def __init__(self):
        """Initialize market events factor"""
        self.cached_events = {}
        
    def get_active_events(self, date: datetime) -> List[Dict]:
        """Get all active events for a given date"""
        active_events = []
        date_str = date.strftime('%m-%d')
        
        # Check cannabis events
        for event_date, event_config in self.CANNABIS_EVENTS.items():
            if self._is_event_active(date, event_date, event_config):
                active_events.append({
                    'date': event_date,
                    'config': event_config,
                    'type': 'cannabis_event'
                })
                
        # Check general holidays
        for holiday_date, holiday_config in self.GENERAL_HOLIDAYS.items():
            if self._is_holiday_active(date, holiday_date, holiday_config):
                active_events.append({
                    'date': holiday_date,
                    'config': holiday_config,
                    'type': 'general_holiday'
                })
                
        return active_events
        
    def _is_event_active(self, check_date: datetime, event_date: str, event_config: Dict) -> bool:
        """Check if a cannabis event is active on the given date"""
        event_month, event_day = map(int, event_date.split('-'))
        event_datetime = datetime(check_date.year, event_month, event_day)
        
        # Check if we're in the ramp period
        ramp_start = event_datetime - timedelta(days=event_config.get('ramp_days', 0))
        ramp_end = event_datetime + timedelta(days=1)  # Include day after
        
        # Add peak days
        for peak_offset in event_config.get('peak_days', [0]):
            peak_date = event_datetime + timedelta(days=peak_offset)
            if peak_date.date() == check_date.date():
                return True
                
        return ramp_start <= check_date <= ramp_end
        
    def _is_holiday_active(self, check_date: datetime, holiday_date: str, holiday_config: Dict) -> bool:
        """Check if a general holiday is active on the given date"""
        holiday_month, holiday_day = map(int, holiday_date.split('-'))
        holiday_datetime = datetime(check_date.year, holiday_month, holiday_day)
        
        duration = holiday_config.get('duration', 1)
        holiday_start = holiday_datetime - timedelta(days=duration // 2)
        holiday_end = holiday_datetime + timedelta(days=duration // 2)
        
        return holiday_start <= check_date <= holiday_end
        
    def calculate_event_multiplier(self, 
                                 date: datetime,
                                 category: str,
                                 active_events: List[Dict]) -> float:
        """Calculate combined multiplier for all active events"""
        if not active_events:
            return 1.0
            
        # Start with base multiplier
        combined_multiplier = 1.0
        
        for event in active_events:
            event_config = event['config']
            
            if event['type'] == 'cannabis_event':
                # Get base event multiplier
                base_mult = event_config.get('base_multiplier', 1.0)
                
                # Apply category-specific multiplier
                cat_mult = event_config.get('category_multipliers', {}).get(category, 1.0)
                
                # Calculate ramp factor (gradual increase)
                ramp_factor = self._calculate_ramp_factor(date, event['date'], event_config)
                
                event_multiplier = 1 + ((base_mult - 1) * cat_mult * ramp_factor)
                
            else:  # general holiday
                # Check if category-specific
                if 'categories' in event_config:
                    if category in event_config['categories']:
                        event_multiplier = event_config.get('multiplier', 1.0)
                    else:
                        event_multiplier = 1.0
                else:
                    event_multiplier = event_config.get('multiplier', 1.0)
                    
            # Combine multiplicatively (compound events)
            combined_multiplier *= event_multiplier
            
        # Cap total event multiplier
        return min(combined_multiplier, 2.5)  # Max 150% increase
        
    def _calculate_ramp_factor(self, current_date: datetime, event_date: str, event_config: Dict) -> float:
        """Calculate ramp factor for gradual price increases"""
        event_month, event_day = map(int, event_date.split('-'))
        event_datetime = datetime(current_date.year, event_month, event_day)
        
        days_until = (event_datetime - current_date).days
        ramp_days = event_config.get('ramp_days', 0)
        
        # Check if we're in a peak day
        for peak_offset in event_config.get('peak_days', [0]):
            peak_date = event_datetime + timedelta(days=peak_offset)
            if peak_date.date() == current_date.date():
                return 1.0  # Full multiplier on peak days
                
        # Calculate ramp
        if days_until > ramp_days:
            return 0.0
        elif days_until <= 0:
            return 0.8  # After event, quick ramp down
        else:
            # Linear ramp up
            return (ramp_days - days_until) / ramp_days
            
    def calculate_seasonal_multiplier(self, date: datetime) -> float:
        """Calculate seasonal adjustment multiplier"""
        month_mult = self.SEASONAL_PATTERNS.get(date.month, 1.0)
        dow_mult = self.DAY_OF_WEEK_PATTERNS.get(date.weekday(), 1.0)
        
        # Combine with slight dampening
        return 1 + ((month_mult - 1) * 0.7 + (dow_mult - 1) * 0.3)
        
    def calculate_factor_score(self,
                             product_data: Dict,
                             target_date: Optional[datetime] = None) -> Dict:
        """Calculate market events factor score and multiplier"""
        
        date = target_date or datetime.now()
        category = product_data.get('category', 'flower')
        
        # Get active events
        active_events = self.get_active_events(date)
        
        # Calculate event multiplier
        event_multiplier = self.calculate_event_multiplier(date, category, active_events)
        
        # Calculate seasonal multiplier
        seasonal_multiplier = self.calculate_seasonal_multiplier(date)
        
        # Combine multipliers
        final_multiplier = event_multiplier * seasonal_multiplier
        
        # Calculate confidence
        confidence = self._calculate_confidence(active_events, date)
        
        # Determine event status
        if active_events:
            if any(e['type'] == 'cannabis_event' for e in active_events):
                event_status = 'major_event'
            else:
                event_status = 'holiday'
        else:
            event_status = 'normal'
            
        return {
            'factor_name': 'market_events',
            'weight': 0.10,
            'raw_score': len(active_events),
            'event_status': event_status,
            'multiplier': final_multiplier,
            'confidence': confidence,
            'details': {
                'date': date.strftime('%Y-%m-%d'),
                'active_events': [e['config']['name'] for e in active_events],
                'event_multiplier': round(event_multiplier, 3),
                'seasonal_multiplier': round(seasonal_multiplier, 3),
                'day_of_week': date.strftime('%A'),
                'category': category
            }
        }
        
    def _calculate_confidence(self, active_events: List[Dict], date: datetime) -> float:
        """Calculate confidence based on event predictability"""
        confidence = 0.95  # High base confidence for calendar events
        
        # Slightly lower confidence for compound events
        if len(active_events) > 1:
            confidence *= 0.9
            
        # Lower confidence for far future dates
        days_ahead = (date - datetime.now()).days
        if days_ahead > 30:
            confidence *= 0.8
        elif days_ahead > 60:
            confidence *= 0.6
            
        return confidence
        
    def get_upcoming_events(self, days_ahead: int = 30) -> List[Dict]:
        """Get list of upcoming events for planning"""
        upcoming = []
        start_date = datetime.now()
        
        for i in range(days_ahead):
            check_date = start_date + timedelta(days=i)
            events = self.get_active_events(check_date)
            
            if events:
                upcoming.append({
                    'date': check_date,
                    'events': events,
                    'impact': self._estimate_event_impact(events)
                })
                
        return upcoming
        
    def _estimate_event_impact(self, events: List[Dict]) -> str:
        """Estimate the pricing impact of events"""
        max_multiplier = 1.0
        
        for event in events:
            if event['type'] == 'cannabis_event':
                mult = event['config'].get('base_multiplier', 1.0)
            else:
                mult = event['config'].get('multiplier', 1.0)
            max_multiplier = max(max_multiplier, mult)
            
        if max_multiplier >= 1.5:
            return 'high'
        elif max_multiplier >= 1.2:
            return 'medium'
        elif max_multiplier > 1.0:
            return 'low'
        else:
            return 'none'
            
    def suggest_event_inventory(self, event_date: str, category: str) -> Dict:
        """Suggest inventory levels for upcoming events"""
        event_config = self.CANNABIS_EVENTS.get(event_date, {})
        
        if not event_config:
            return {'suggestion': 'normal', 'multiplier': 1.0}
            
        # Get expected demand multiplier
        base_mult = event_config.get('base_multiplier', 1.0)
        cat_mult = event_config.get('category_multipliers', {}).get(category, 1.0)
        expected_demand = base_mult * cat_mult
        
        # Suggest inventory levels
        if expected_demand >= 2.0:
            suggestion = 'triple_inventory'
            inventory_mult = 3.0
        elif expected_demand >= 1.5:
            suggestion = 'double_inventory'
            inventory_mult = 2.0
        elif expected_demand >= 1.2:
            suggestion = 'increase_50_percent'
            inventory_mult = 1.5
        else:
            suggestion = 'normal_plus'
            inventory_mult = 1.2
            
        return {
            'suggestion': suggestion,
            'inventory_multiplier': inventory_mult,
            'expected_demand_multiplier': expected_demand,
            'lead_time_days': event_config.get('ramp_days', 7) + 7
        }