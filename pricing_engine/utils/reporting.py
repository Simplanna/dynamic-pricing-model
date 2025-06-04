from typing import Dict, List, Any
import pandas as pd
from datetime import datetime
import json


class DataQualityReport:
    """Generate comprehensive data quality reports"""
    
    def __init__(self):
        self.report_timestamp = datetime.now()
        self.sections = {}
        
    def add_section(self, name: str, content: Dict[str, Any]):
        """Add a section to the report"""
        self.sections[name] = content
        
    def generate_validation_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation summary section"""
        summary = {
            'overview': {
                'total_products_validated': validation_results['total'],
                'valid_products': validation_results['valid'],
                'invalid_products': validation_results['invalid'],
                'products_with_warnings': validation_results['warnings'],
                'validation_pass_rate': (validation_results['valid'] / validation_results['total'] * 100) if validation_results['total'] > 0 else 0
            },
            'common_errors': {},
            'common_warnings': {}
        }
        
        # Analyze common issues
        error_counts = {}
        warning_counts = {}
        
        for result in validation_results.get('validation_results', []):
            for error in result.get('errors', []):
                error_type = self._categorize_error(error)
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
                
            for warning in result.get('warnings', []):
                warning_type = self._categorize_warning(warning)
                warning_counts[warning_type] = warning_counts.get(warning_type, 0) + 1
                
        summary['common_errors'] = error_counts
        summary['common_warnings'] = warning_counts
        
        return summary
        
    def _categorize_error(self, error: str) -> str:
        """Categorize error messages"""
        if 'price' in error.lower():
            return 'Price Validation Error'
        elif 'thc' in error.lower() or 'cbd' in error.lower():
            return 'Cannabinoid Validation Error'
        elif 'category' in error.lower():
            return 'Category Validation Error'
        elif 'missing' in error.lower() or 'required' in error.lower():
            return 'Missing Required Field'
        else:
            return 'Other Validation Error'
            
    def _categorize_warning(self, warning: str) -> str:
        """Categorize warning messages"""
        if 'category' in warning.lower():
            return 'Category Correction'
        elif 'weight' in warning.lower():
            return 'Unusual Weight'
        elif 'price' in warning.lower():
            return 'Price Anomaly'
        else:
            return 'Other Warning'
            
    def generate_scraping_summary(self, scraped_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate scraping summary section"""
        if not scraped_data:
            return {'status': 'No data scraped yet'}
            
        summary = {
            'total_dispensaries_scraped': len(scraped_data),
            'platforms': {},
            'states': {},
            'success_rate': 0,
            'average_products_per_dispensary': 0
        }
        
        # Analyze by platform
        platform_counts = {}
        state_counts = {}
        successful_scrapes = 0
        total_products = 0
        
        for scrape in scraped_data:
            platform = scrape.get('platform', 'unknown')
            platform_counts[platform] = platform_counts.get(platform, 0) + 1
            
            state = scrape.get('state', 'unknown')
            state_counts[state] = state_counts.get(state, 0) + 1
            
            if scrape.get('status') == 'success':
                successful_scrapes += 1
                total_products += scrape.get('products_found', 0)
                
        summary['platforms'] = platform_counts
        summary['states'] = state_counts
        summary['success_rate'] = (successful_scrapes / len(scraped_data) * 100) if scraped_data else 0
        summary['average_products_per_dispensary'] = total_products / successful_scrapes if successful_scrapes > 0 else 0
        
        return summary
        
    def generate_market_insights(self, market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate market insights section"""
        insights = {
            'price_trends': {},
            'competitive_landscape': {},
            'recommendations': []
        }
        
        # Add insights based on analysis
        if 'ma_analysis' in market_analysis:
            ma_data = market_analysis['ma_analysis']
            insights['price_trends']['massachusetts'] = {
                'current_avg_price_per_gram': ma_data.get('avg_price_per_gram', 0),
                'market_status': 'Oversupplied' if ma_data.get('avg_price_per_gram', 0) < 6 else 'Balanced'
            }
            
        if 'ri_analysis' in market_analysis:
            ri_data = market_analysis['ri_analysis']
            insights['price_trends']['rhode_island'] = {
                'current_avg_price_per_gram': ri_data.get('avg_price_per_gram', 0),
                'market_status': 'Premium Pricing' if ri_data.get('avg_price_per_gram', 0) > 10 else 'Competitive'
            }
            
        # Generate recommendations
        if insights['price_trends'].get('massachusetts', {}).get('market_status') == 'Oversupplied':
            insights['recommendations'].append(
                "MA Market: Focus on differentiation through quality and service rather than price competition"
            )
            
        if insights['price_trends'].get('rhode_island', {}).get('market_status') == 'Premium Pricing':
            insights['recommendations'].append(
                "RI Market: Monitor MA border competition closely - price pressure likely to increase"
            )
            
        return insights
        
    def generate_full_report(self, validation_results: Dict[str, Any],
                           scraped_data: List[Dict[str, Any]],
                           market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete data quality report"""
        
        # Add all sections
        self.add_section('Report Metadata', {
            'generated_at': self.report_timestamp.isoformat(),
            'report_version': '1.0',
            'phase': 'Phase 1 - Data Integration & Market Analysis'
        })
        
        self.add_section('Data Validation Summary', 
                        self.generate_validation_summary(validation_results))
        
        self.add_section('Scraping Summary',
                        self.generate_scraping_summary(scraped_data))
        
        self.add_section('Market Insights',
                        self.generate_market_insights(market_analysis))
        
        # Add data quality metrics
        quality_metrics = {
            'overall_data_quality_score': 0,
            'component_scores': {}
        }
        
        # Calculate component scores
        if validation_results:
            validation_score = validation_results.get('valid', 0) / validation_results.get('total', 1) * 100
            quality_metrics['component_scores']['validation'] = validation_score
        else:
            quality_metrics['component_scores']['validation'] = 0
            
        if scraped_data:
            scraping_score = sum(1 for s in scraped_data if s.get('status') == 'success') / len(scraped_data) * 100
            quality_metrics['component_scores']['scraping'] = scraping_score
        else:
            quality_metrics['component_scores']['scraping'] = 0
            
        # Calculate overall score
        if quality_metrics['component_scores']:
            quality_metrics['overall_data_quality_score'] = sum(
                quality_metrics['component_scores'].values()
            ) / len(quality_metrics['component_scores'])
            
        self.add_section('Data Quality Metrics', quality_metrics)
        
        # Add recommendations
        recommendations = {
            'immediate_actions': [],
            'future_improvements': []
        }
        
        # Generate recommendations based on metrics
        if quality_metrics['overall_data_quality_score'] < 80:
            recommendations['immediate_actions'].append(
                "Improve data validation rules to increase quality score above 80%"
            )
            
        if quality_metrics['component_scores'].get('scraping', 0) < 90:
            recommendations['immediate_actions'].append(
                "Investigate and fix scraping failures to achieve 90%+ success rate"
            )
            
        recommendations['future_improvements'].extend([
            "Implement automated daily scraping schedule",
            "Add more dispensary sources to improve market coverage",
            "Enhance product matching algorithm with ML-based approach",
            "Build historical price database for trend analysis"
        ])
        
        self.add_section('Recommendations', recommendations)
        
        return {
            'report': self.sections,
            'summary': {
                'overall_quality_score': quality_metrics['overall_data_quality_score'],
                'total_products_validated': validation_results.get('total', 0),
                'dispensaries_scraped': len(scraped_data),
                'report_generated': self.report_timestamp.isoformat()
            }
        }
        
    def export_report(self, report: Dict[str, Any], format: str = 'json') -> str:
        """Export report in specified format"""
        if format == 'json':
            return json.dumps(report, indent=2, default=str)
        elif format == 'markdown':
            return self._generate_markdown_report(report)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def _generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """Generate markdown formatted report"""
        md_lines = []
        
        md_lines.append("# Cannabis Dynamic Pricing - Data Quality Report")
        md_lines.append(f"\nGenerated: {report['summary']['report_generated']}")
        md_lines.append(f"\n## Executive Summary")
        md_lines.append(f"\n- **Overall Data Quality Score**: {report['summary']['overall_quality_score']:.1f}%")
        md_lines.append(f"- **Products Validated**: {report['summary']['total_products_validated']}")
        md_lines.append(f"- **Dispensaries Scraped**: {report['summary']['dispensaries_scraped']}")
        
        for section_name, section_data in report['report'].items():
            md_lines.append(f"\n## {section_name}")
            md_lines.append(self._dict_to_markdown(section_data, level=0))
            
        return '\n'.join(md_lines)
        
    def _dict_to_markdown(self, data: Any, level: int = 0) -> str:
        """Convert dictionary to markdown format"""
        lines = []
        indent = '  ' * level
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    lines.append(f"\n{indent}**{key}**:")
                    lines.append(self._dict_to_markdown(value, level + 1))
                elif isinstance(value, list):
                    lines.append(f"\n{indent}**{key}**:")
                    for item in value:
                        lines.append(f"{indent}- {item}")
                else:
                    lines.append(f"{indent}- **{key}**: {value}")
        else:
            lines.append(f"{indent}{data}")
            
        return '\n'.join(lines)