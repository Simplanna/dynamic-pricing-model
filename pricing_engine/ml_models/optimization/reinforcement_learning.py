"""
Reinforcement Learning for Price Optimization
Conservative multi-armed bandit with safety constraints for cannabis compliance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PricingAction:
    """Represents a pricing action"""
    product_id: str
    current_price: float
    new_price: float
    timestamp: pd.Timestamp
    context: Dict


@dataclass
class PricingReward:
    """Represents reward from pricing action"""
    revenue: float
    units_sold: int
    margin: float
    customer_satisfaction: float
    compliance_maintained: bool


class ConservativeMAB:
    """Conservative Multi-Armed Bandit for price testing"""
    
    def __init__(self, epsilon: float = 0.1, safety_threshold: float = 0.95):
        self.epsilon = epsilon  # Exploration rate
        self.safety_threshold = safety_threshold  # Conservative threshold
        self.arms = {}  # Price points as arms
        self.action_history = []
        self.reward_history = []
        
    def add_arm(self, product_id: str, price_point: float):
        """Add a price point as an arm"""
        
        key = f"{product_id}_{price_point}"
        if key not in self.arms:
            self.arms[key] = {
                'product_id': product_id,
                'price': price_point,
                'pulls': 0,
                'total_reward': 0,
                'avg_reward': 0,
                'ucb_score': float('inf'),  # Upper Confidence Bound
                'safety_violations': 0
            }
    
    def select_action(self, product_id: str, context: Dict) -> float:
        """Select price using epsilon-greedy with safety constraints"""
        
        # Get available arms for product
        product_arms = {k: v for k, v in self.arms.items() 
                       if v['product_id'] == product_id}
        
        if not product_arms:
            return context.get('current_price', 0)
        
        # Safety check - exclude arms with violations
        safe_arms = {k: v for k, v in product_arms.items() 
                    if v['safety_violations'] == 0}
        
        if not safe_arms:
            # All arms violated safety, return current price
            return context.get('current_price', 0)
        
        # Epsilon-greedy selection
        if np.random.random() < self.epsilon:
            # Explore - but conservatively
            selected_key = self._conservative_exploration(safe_arms, context)
        else:
            # Exploit - select best performing arm
            selected_key = max(safe_arms.keys(), 
                             key=lambda k: safe_arms[k]['ucb_score'])
        
        selected_arm = self.arms[selected_key]
        selected_arm['pulls'] += 1
        
        # Record action
        action = PricingAction(
            product_id=product_id,
            current_price=context.get('current_price', selected_arm['price']),
            new_price=selected_arm['price'],
            timestamp=pd.Timestamp.now(),
            context=context
        )
        self.action_history.append(action)
        
        return selected_arm['price']
    
    def update_reward(self, product_id: str, price: float, reward: PricingReward):
        """Update arm statistics with observed reward"""
        
        key = f"{product_id}_{price}"
        if key not in self.arms:
            return
        
        arm = self.arms[key]
        
        # Calculate composite reward
        composite_reward = self._calculate_composite_reward(reward)
        
        # Update statistics
        arm['total_reward'] += composite_reward
        arm['avg_reward'] = arm['total_reward'] / arm['pulls']
        
        # Update UCB score
        total_pulls = sum(a['pulls'] for a in self.arms.values())
        if total_pulls > 0 and arm['pulls'] > 0:
            exploration_bonus = np.sqrt(2 * np.log(total_pulls) / arm['pulls'])
            arm['ucb_score'] = arm['avg_reward'] + exploration_bonus
        
        # Check safety violations
        if not reward.compliance_maintained:
            arm['safety_violations'] += 1
        
        # Store reward
        self.reward_history.append({
            'product_id': product_id,
            'price': price,
            'reward': composite_reward,
            'timestamp': pd.Timestamp.now()
        })
    
    def _conservative_exploration(self, safe_arms: Dict, context: Dict) -> str:
        """Conservative exploration strategy"""
        
        current_price = context.get('current_price', 0)
        
        # Find arms close to current price (within 10%)
        nearby_arms = {k: v for k, v in safe_arms.items() 
                      if abs(v['price'] - current_price) / current_price <= 0.1}
        
        if nearby_arms:
            # Explore among nearby prices
            return np.random.choice(list(nearby_arms.keys()))
        else:
            # If no nearby arms, pick closest one
            return min(safe_arms.keys(), 
                      key=lambda k: abs(safe_arms[k]['price'] - current_price))
    
    def _calculate_composite_reward(self, reward: PricingReward) -> float:
        """Calculate composite reward from multiple objectives"""
        
        # Normalize components
        revenue_score = reward.revenue / 1000  # Normalize to reasonable scale
        margin_score = reward.margin
        satisfaction_score = reward.customer_satisfaction
        
        # Penalize compliance violations heavily
        compliance_penalty = -10.0 if not reward.compliance_maintained else 0.0
        
        # Weighted combination
        composite = (
            0.5 * revenue_score +
            0.3 * margin_score +
            0.2 * satisfaction_score +
            compliance_penalty
        )
        
        return composite


class PriceOptimizationRL:
    """Reinforcement Learning system for price optimization"""
    
    def __init__(self):
        self.bandits = {}  # MAB per product category
        self.experiment_config = {
            'min_experiment_duration': 24,  # hours
            'min_samples_per_arm': 100,
            'max_price_change': 0.15,  # 15% max change
            'confidence_threshold': 0.95
        }
        self.active_experiments = {}
        
    def create_experiment(self, product_id: str, price_range: Tuple[float, float], 
                         n_arms: int = 5) -> Dict:
        """Create price optimization experiment"""
        
        # Generate price points
        prices = np.linspace(price_range[0], price_range[1], n_arms)
        
        # Create bandit for experiment
        bandit = ConservativeMAB(epsilon=0.1, safety_threshold=0.95)
        
        for price in prices:
            bandit.add_arm(product_id, round(price, 2))
        
        self.bandits[product_id] = bandit
        
        # Initialize experiment
        self.active_experiments[product_id] = {
            'start_time': pd.Timestamp.now(),
            'status': 'active',
            'price_range': price_range,
            'n_arms': n_arms,
            'results': []
        }
        
        return {
            'experiment_id': f"exp_{product_id}_{pd.Timestamp.now().strftime('%Y%m%d')}",
            'product_id': product_id,
            'price_points': prices.tolist(),
            'estimated_duration': f"{self.experiment_config['min_experiment_duration']} hours"
        }
    
    def get_next_price(self, product_id: str, context: Dict) -> Dict:
        """Get next price to test"""
        
        if product_id not in self.bandits:
            return {
                'price': context.get('current_price', 0),
                'is_experiment': False,
                'reason': 'No active experiment'
            }
        
        # Check experiment status
        experiment = self.active_experiments[product_id]
        if experiment['status'] != 'active':
            return {
                'price': context.get('current_price', 0),
                'is_experiment': False,
                'reason': f"Experiment {experiment['status']}"
            }
        
        # Get price from bandit
        bandit = self.bandits[product_id]
        recommended_price = bandit.select_action(product_id, context)
        
        # Safety check - ensure price change is within limits
        current_price = context.get('current_price', recommended_price)
        price_change = abs(recommended_price - current_price) / current_price
        
        if price_change > self.experiment_config['max_price_change']:
            # Clip to max change
            if recommended_price > current_price:
                recommended_price = current_price * (1 + self.experiment_config['max_price_change'])
            else:
                recommended_price = current_price * (1 - self.experiment_config['max_price_change'])
        
        return {
            'price': round(recommended_price, 2),
            'is_experiment': True,
            'confidence': self._calculate_confidence(product_id),
            'expected_improvement': self._estimate_improvement(product_id, recommended_price)
        }
    
    def record_outcome(self, product_id: str, price: float, outcome: Dict):
        """Record outcome of pricing decision"""
        
        if product_id not in self.bandits:
            return
        
        # Convert outcome to reward
        reward = PricingReward(
            revenue=outcome.get('revenue', 0),
            units_sold=outcome.get('units_sold', 0),
            margin=outcome.get('margin', 0),
            customer_satisfaction=outcome.get('satisfaction', 0.8),
            compliance_maintained=outcome.get('compliance_ok', True)
        )
        
        # Update bandit
        self.bandits[product_id].update_reward(product_id, price, reward)
        
        # Update experiment results
        if product_id in self.active_experiments:
            self.active_experiments[product_id]['results'].append({
                'timestamp': pd.Timestamp.now(),
                'price': price,
                'outcome': outcome
            })
        
        # Check if experiment should conclude
        self._check_experiment_completion(product_id)
    
    def get_experiment_results(self, product_id: str) -> Dict:
        """Get current experiment results and recommendations"""
        
        if product_id not in self.bandits:
            return {'status': 'no_experiment'}
        
        bandit = self.bandits[product_id]
        experiment = self.active_experiments[product_id]
        
        # Compile results
        arm_performance = []
        for arm_key, arm_data in bandit.arms.items():
            if arm_data['product_id'] == product_id:
                arm_performance.append({
                    'price': arm_data['price'],
                    'tests': arm_data['pulls'],
                    'avg_reward': arm_data['avg_reward'],
                    'confidence': self._calculate_arm_confidence(arm_data),
                    'safety_violations': arm_data['safety_violations']
                })
        
        # Find best price
        safe_arms = [a for a in arm_performance if a['safety_violations'] == 0]
        if safe_arms:
            best_arm = max(safe_arms, key=lambda x: x['avg_reward'])
            recommended_price = best_arm['price']
        else:
            recommended_price = experiment['price_range'][0]  # Default to min price
        
        return {
            'status': experiment['status'],
            'duration_hours': (pd.Timestamp.now() - experiment['start_time']).total_seconds() / 3600,
            'total_tests': len(experiment['results']),
            'arm_performance': sorted(arm_performance, key=lambda x: x['avg_reward'], reverse=True),
            'recommended_price': recommended_price,
            'confidence': self._calculate_confidence(product_id),
            'expected_lift': self._calculate_expected_lift(product_id, recommended_price)
        }
    
    def _calculate_confidence(self, product_id: str) -> float:
        """Calculate confidence in current results"""
        
        if product_id not in self.bandits:
            return 0.0
        
        bandit = self.bandits[product_id]
        
        # Check minimum samples
        total_pulls = sum(arm['pulls'] for arm in bandit.arms.values())
        min_pulls = min(arm['pulls'] for arm in bandit.arms.values() if arm['pulls'] > 0)
        
        if min_pulls < self.experiment_config['min_samples_per_arm']:
            sample_confidence = min_pulls / self.experiment_config['min_samples_per_arm']
        else:
            sample_confidence = 1.0
        
        # Check experiment duration
        experiment = self.active_experiments[product_id]
        duration_hours = (pd.Timestamp.now() - experiment['start_time']).total_seconds() / 3600
        
        if duration_hours < self.experiment_config['min_experiment_duration']:
            duration_confidence = duration_hours / self.experiment_config['min_experiment_duration']
        else:
            duration_confidence = 1.0
        
        return min(sample_confidence, duration_confidence)
    
    def _estimate_improvement(self, product_id: str, price: float) -> float:
        """Estimate expected improvement from price"""
        
        if product_id not in self.bandits:
            return 0.0
        
        bandit = self.bandits[product_id]
        
        # Find current best
        current_best = max(bandit.arms.values(), key=lambda x: x['avg_reward'])
        
        # Find selected arm
        key = f"{product_id}_{price}"
        if key in bandit.arms:
            selected = bandit.arms[key]
            if selected['pulls'] > 0:
                return (selected['avg_reward'] - current_best['avg_reward']) / abs(current_best['avg_reward'])
        
        return 0.0
    
    def _calculate_arm_confidence(self, arm_data: Dict) -> float:
        """Calculate confidence for specific arm"""
        
        if arm_data['pulls'] == 0:
            return 0.0
        
        # Based on number of pulls and consistency
        pulls_confidence = min(1.0, arm_data['pulls'] / 100)
        
        # Penalize safety violations
        safety_confidence = 1.0 if arm_data['safety_violations'] == 0 else 0.5
        
        return pulls_confidence * safety_confidence
    
    def _check_experiment_completion(self, product_id: str):
        """Check if experiment should be concluded"""
        
        experiment = self.active_experiments[product_id]
        bandit = self.bandits[product_id]
        
        # Check duration
        duration_hours = (pd.Timestamp.now() - experiment['start_time']).total_seconds() / 3600
        
        # Check samples
        min_pulls = min(arm['pulls'] for arm in bandit.arms.values())
        
        # Check convergence
        rewards = [arm['avg_reward'] for arm in bandit.arms.values() if arm['pulls'] > 10]
        if rewards:
            reward_std = np.std(rewards)
            has_converged = reward_std < 0.05
        else:
            has_converged = False
        
        # Complete if criteria met
        if (duration_hours >= self.experiment_config['min_experiment_duration'] and
            min_pulls >= self.experiment_config['min_samples_per_arm']) or has_converged:
            experiment['status'] = 'completed'
    
    def _calculate_expected_lift(self, product_id: str, recommended_price: float) -> float:
        """Calculate expected revenue lift from recommended price"""
        
        experiment = self.active_experiments[product_id]
        if not experiment['results']:
            return 0.0
        
        # Calculate baseline (first price tested)
        baseline_results = [r for r in experiment['results'][:20]]
        if baseline_results:
            baseline_revenue = np.mean([r['outcome'].get('revenue', 0) for r in baseline_results])
        else:
            baseline_revenue = 0
        
        # Calculate recommended price performance
        recommended_results = [r for r in experiment['results'] if r['price'] == recommended_price]
        if recommended_results:
            recommended_revenue = np.mean([r['outcome'].get('revenue', 0) for r in recommended_results])
        else:
            recommended_revenue = baseline_revenue
        
        if baseline_revenue > 0:
            return (recommended_revenue - baseline_revenue) / baseline_revenue
        return 0.0