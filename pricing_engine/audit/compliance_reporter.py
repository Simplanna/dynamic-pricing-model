"""Automated compliance reporting for regulatory requirements."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
import logging
from enum import Enum
from pathlib import Path
import json
import csv
from collections import defaultdict
import pandas as pd

from .regulatory_audit import RegulatoryAuditTrail, AuditEventType
from ..core.models import Market, ProductCategory
from ..compliance.state_compliance import ComplianceRule

logger = logging.getLogger(__name__)


class ReportType(Enum):
    """Types of compliance reports."""
    DAILY_SUMMARY = "daily_summary"
    WEEKLY_COMPLIANCE = "weekly_compliance"
    MONTHLY_REGULATORY = "monthly_regulatory"
    TAX_REPORT = "tax_report"
    VIOLATION_SUMMARY = "violation_summary"
    MANUAL_OVERRIDE = "manual_override"
    AUDIT_TRAIL = "audit_trail"
    PRICE_CHANGE_LOG = "price_change_log"


@dataclass
class ReportSection:
    """Section of a compliance report."""
    title: str
    content: Dict[str, Any]
    tables: List[pd.DataFrame] = field(default_factory=list)
    charts: List[Dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class ComplianceReport:
    """Complete compliance report."""
    report_id: str
    report_type: ReportType
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    sections: List[ReportSection]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert report to dictionary."""
        return {
            'report_id': self.report_id,
            'report_type': self.report_type.value,
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'generated_at': self.generated_at.isoformat(),
            'sections': [
                {
                    'title': section.title,
                    'content': section.content,
                    'notes': section.notes
                }
                for section in self.sections
            ],
            'metadata': self.metadata
        }


class ComplianceReporter:
    """Generates regulatory compliance reports."""
    
    def __init__(self, audit_trail: RegulatoryAuditTrail,
                 report_directory: Optional[Path] = None):
        self.audit_trail = audit_trail
        self.report_dir = report_directory or Path("compliance_reports")
        self.report_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        for report_type in ReportType:
            (self.report_dir / report_type.value).mkdir(exist_ok=True)
        
        logger.info("Compliance reporter initialized")
    
    def generate_daily_summary(self, date: Optional[date] = None) -> ComplianceReport:
        """Generate daily compliance summary report."""
        if not date:
            date = datetime.now().date()
        
        start = datetime.combine(date, datetime.min.time())
        end = datetime.combine(date, datetime.max.time())
        
        # Query audit events
        events = self.audit_trail.query_events(start, end)
        
        # Create report sections
        sections = []
        
        # Overview section
        overview = self._create_overview_section(events, start, end)
        sections.append(overview)
        
        # Pricing activity section
        pricing_section = self._create_pricing_section(events)
        sections.append(pricing_section)
        
        # Compliance section
        compliance_section = self._create_compliance_section(events)
        sections.append(compliance_section)
        
        # Alerts and interventions
        alerts_section = self._create_alerts_section(events)
        sections.append(alerts_section)
        
        # Create report
        report = ComplianceReport(
            report_id=f"DAILY_{date.strftime('%Y%m%d')}",
            report_type=ReportType.DAILY_SUMMARY,
            period_start=start,
            period_end=end,
            generated_at=datetime.now(),
            sections=sections,
            metadata={
                'total_events': len(events),
                'report_date': date.isoformat()
            }
        )
        
        # Save report
        self._save_report(report)
        
        return report
    
    def generate_weekly_compliance(self, week_start: Optional[date] = None) -> ComplianceReport:
        """Generate weekly compliance report."""
        if not week_start:
            # Default to last Monday
            today = datetime.now().date()
            week_start = today - timedelta(days=today.weekday())
        
        week_end = week_start + timedelta(days=6)
        
        start = datetime.combine(week_start, datetime.min.time())
        end = datetime.combine(week_end, datetime.max.time())
        
        # Query events
        events = self.audit_trail.query_events(start, end)
        
        sections = []
        
        # Weekly summary
        summary_section = self._create_weekly_summary(events, start, end)
        sections.append(summary_section)
        
        # Compliance violations analysis
        violations_section = self._analyze_violations(events)
        sections.append(violations_section)
        
        # Market comparison
        market_section = self._create_market_comparison(events)
        sections.append(market_section)
        
        # Trends analysis
        trends_section = self._analyze_weekly_trends(events, start)
        sections.append(trends_section)
        
        report = ComplianceReport(
            report_id=f"WEEKLY_{week_start.strftime('%Y%m%d')}",
            report_type=ReportType.WEEKLY_COMPLIANCE,
            period_start=start,
            period_end=end,
            generated_at=datetime.now(),
            sections=sections,
            metadata={
                'week_number': week_start.isocalendar()[1],
                'year': week_start.year
            }
        )
        
        self._save_report(report)
        return report
    
    def generate_monthly_regulatory(self, year: int, month: int) -> ComplianceReport:
        """Generate comprehensive monthly regulatory report."""
        # Calculate date range
        start = datetime(year, month, 1)
        if month == 12:
            end = datetime(year + 1, 1, 1) - timedelta(seconds=1)
        else:
            end = datetime(year, month + 1, 1) - timedelta(seconds=1)
        
        # Query all events for the month
        events = self.audit_trail.query_events(start, end)
        
        sections = []
        
        # Executive summary
        exec_summary = self._create_executive_summary(events, start, end)
        sections.append(exec_summary)
        
        # Detailed compliance analysis
        compliance_analysis = self._detailed_compliance_analysis(events)
        sections.append(compliance_analysis)
        
        # Tax compliance
        tax_section = self._create_tax_compliance_section(events)
        sections.append(tax_section)
        
        # Manual interventions and overrides
        manual_section = self._analyze_manual_interventions(events)
        sections.append(manual_section)
        
        # System performance
        performance_section = self._analyze_system_performance(events)
        sections.append(performance_section)
        
        # Regulatory attestation
        attestation = self._create_attestation_section(events, start, end)
        sections.append(attestation)
        
        report = ComplianceReport(
            report_id=f"MONTHLY_{year}{month:02d}",
            report_type=ReportType.MONTHLY_REGULATORY,
            period_start=start,
            period_end=end,
            generated_at=datetime.now(),
            sections=sections,
            metadata={
                'month': month,
                'year': year,
                'total_pricing_decisions': len([e for e in events if e.event_type == AuditEventType.PRICE_DECISION]),
                'regulatory_format': 'MA_RI_Cannabis_v1'
            }
        )
        
        self._save_report(report)
        return report
    
    def generate_tax_report(self, start_date: datetime, end_date: datetime,
                          market: Market) -> ComplianceReport:
        """Generate tax compliance report."""
        # Query price change events
        events = self.audit_trail.query_events(
            start_date, end_date,
            event_types=[AuditEventType.PRICE_CHANGE, AuditEventType.PRICE_DECISION],
            market=market
        )
        
        sections = []
        
        # Tax summary
        tax_summary = self._calculate_tax_summary(events, market)
        sections.append(tax_summary)
        
        # Category breakdown
        category_tax = self._tax_by_category(events, market)
        sections.append(category_tax)
        
        # Municipal breakdown (if applicable)
        if market == Market.MASSACHUSETTS:
            municipal_tax = self._municipal_tax_breakdown(events)
            sections.append(municipal_tax)
        
        report = ComplianceReport(
            report_id=f"TAX_{market.value}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}",
            report_type=ReportType.TAX_REPORT,
            period_start=start_date,
            period_end=end_date,
            generated_at=datetime.now(),
            sections=sections,
            metadata={
                'market': market.value,
                'currency': 'USD'
            }
        )
        
        self._save_report(report)
        return report
    
    def export_for_regulator(self, report: ComplianceReport,
                           format: str = "pdf") -> Path:
        """Export report in regulator-required format."""
        export_path = self.report_dir / "exports" / f"{report.report_id}.{format}"
        export_path.parent.mkdir(exist_ok=True)
        
        if format == "json":
            # JSON export
            with open(export_path, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
        
        elif format == "csv":
            # CSV export - flatten data
            self._export_to_csv(report, export_path)
        
        elif format == "pdf":
            # PDF export would require additional library
            # For now, create a detailed text report
            self._export_to_text(report, export_path.with_suffix('.txt'))
            logger.info(f"PDF export not implemented, created text report: {export_path.with_suffix('.txt')}")
        
        return export_path
    
    def _create_overview_section(self, events: List, start: datetime, 
                               end: datetime) -> ReportSection:
        """Create overview section."""
        # Count events by type
        event_counts = defaultdict(int)
        for event in events:
            event_counts[event.event_type.value] += 1
        
        # Calculate key metrics
        total_decisions = event_counts.get(AuditEventType.PRICE_DECISION.value, 0)
        total_changes = event_counts.get(AuditEventType.PRICE_CHANGE.value, 0)
        violations = event_counts.get(AuditEventType.COMPLIANCE_VIOLATION.value, 0)
        
        content = {
            'period': f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}",
            'total_events': len(events),
            'event_breakdown': dict(event_counts),
            'key_metrics': {
                'pricing_decisions': total_decisions,
                'price_changes_applied': total_changes,
                'compliance_violations': violations,
                'violation_rate': violations / max(total_decisions, 1)
            }
        }
        
        return ReportSection(
            title="Daily Overview",
            content=content,
            notes=[
                f"Report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"All times in system timezone"
            ]
        )
    
    def _create_pricing_section(self, events: List) -> ReportSection:
        """Create pricing activity section."""
        price_changes = [e for e in events if e.event_type == AuditEventType.PRICE_CHANGE]
        
        if not price_changes:
            return ReportSection(
                title="Pricing Activity",
                content={'message': 'No price changes recorded'},
                notes=[]
            )
        
        # Analyze price changes
        increases = 0
        decreases = 0
        total_change_pct = Decimal('0')
        
        for event in price_changes:
            details = event.details
            change_pct = Decimal(str(details.get('change_percentage', 0)))
            total_change_pct += abs(change_pct)
            
            if change_pct > 0:
                increases += 1
            elif change_pct < 0:
                decreases += 1
        
        avg_change = total_change_pct / len(price_changes) if price_changes else Decimal('0')
        
        content = {
            'total_changes': len(price_changes),
            'increases': increases,
            'decreases': decreases,
            'no_change': len(price_changes) - increases - decreases,
            'average_change_magnitude': float(avg_change),
            'largest_increase': self._find_largest_change(price_changes, 'increase'),
            'largest_decrease': self._find_largest_change(price_changes, 'decrease')
        }
        
        return ReportSection(
            title="Pricing Activity",
            content=content,
            notes=[
                "Price changes are recorded when applied to the system",
                "Percentage changes are calculated from previous price"
            ]
        )
    
    def _create_compliance_section(self, events: List) -> ReportSection:
        """Create compliance section."""
        violations = [e for e in events if e.event_type == AuditEventType.COMPLIANCE_VIOLATION]
        compliance_checks = [e for e in events if e.event_type == AuditEventType.COMPLIANCE_CHECK]
        
        # Analyze violations by type
        violation_types = defaultdict(int)
        violation_severity = defaultdict(int)
        
        for violation in violations:
            details = violation.details
            violation_types[details.get('violation_type', 'unknown')] += 1
            violation_severity[details.get('severity', 'unknown')] += 1
        
        content = {
            'total_compliance_checks': len(compliance_checks),
            'total_violations': len(violations),
            'compliance_rate': 1 - (len(violations) / max(len(compliance_checks), 1)),
            'violations_by_type': dict(violation_types),
            'violations_by_severity': dict(violation_severity),
            'critical_violations': violation_severity.get('critical', 0)
        }
        
        notes = []
        if violation_severity.get('critical', 0) > 0:
            notes.append("CRITICAL: Critical compliance violations detected requiring immediate attention")
        
        return ReportSection(
            title="Compliance Status",
            content=content,
            notes=notes
        )
    
    def _create_alerts_section(self, events: List) -> ReportSection:
        """Create alerts and interventions section."""
        manual_interventions = [e for e in events if e.event_type == AuditEventType.MANUAL_INTERVENTION]
        safety_overrides = [e for e in events if e.event_type == AuditEventType.SAFETY_OVERRIDE]
        rollbacks = [e for e in events if e.event_type == AuditEventType.ROLLBACK]
        
        content = {
            'manual_interventions': len(manual_interventions),
            'safety_overrides': len(safety_overrides),
            'rollbacks': len(rollbacks),
            'intervention_details': [
                {
                    'time': e.timestamp.strftime('%H:%M:%S'),
                    'type': e.details.get('intervention_type', 'unknown'),
                    'user': e.user_id,
                    'product': e.product_sku
                }
                for e in manual_interventions[:5]  # First 5
            ]
        }
        
        notes = []
        if rollbacks:
            notes.append(f"System performed {len(rollbacks)} automatic rollback(s)")
        
        return ReportSection(
            title="Alerts and Interventions",
            content=content,
            notes=notes
        )
    
    def _find_largest_change(self, price_changes: List, direction: str) -> Dict:
        """Find the largest price change in a direction."""
        if not price_changes:
            return {}
        
        if direction == 'increase':
            changes = [(e, e.details.get('change_percentage', 0)) 
                      for e in price_changes if e.details.get('change_percentage', 0) > 0]
        else:
            changes = [(e, abs(e.details.get('change_percentage', 0))) 
                      for e in price_changes if e.details.get('change_percentage', 0) < 0]
        
        if not changes:
            return {}
        
        largest = max(changes, key=lambda x: x[1])
        event = largest[0]
        
        return {
            'product_sku': event.product_sku,
            'product_name': event.details.get('product_name', 'Unknown'),
            'change_percentage': event.details.get('change_percentage', 0),
            'old_price': event.details.get('old_price', 0),
            'new_price': event.details.get('new_price', 0)
        }
    
    def _save_report(self, report: ComplianceReport):
        """Save report to disk."""
        # Save as JSON
        filename = f"{report.report_id}_{report.generated_at.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.report_dir / report.report_type.value / filename
        
        with open(filepath, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        logger.info(f"Report saved: {filepath}")
    
    def _export_to_csv(self, report: ComplianceReport, path: Path):
        """Export report data to CSV format."""
        # Create a flattened view of the report
        rows = []
        
        for section in report.sections:
            for key, value in section.content.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        rows.append({
                            'section': section.title,
                            'metric': f"{key}.{sub_key}",
                            'value': sub_value
                        })
                else:
                    rows.append({
                        'section': section.title,
                        'metric': key,
                        'value': value
                    })
        
        # Write to CSV
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
    
    def _export_to_text(self, report: ComplianceReport, path: Path):
        """Export report as formatted text."""
        with open(path, 'w') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write(f"COMPLIANCE REPORT: {report.report_type.value.upper()}\n")
            f.write("=" * 80 + "\n\n")
            
            # Metadata
            f.write(f"Report ID: {report.report_id}\n")
            f.write(f"Period: {report.period_start.strftime('%Y-%m-%d')} to {report.period_end.strftime('%Y-%m-%d')}\n")
            f.write(f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n" + "-" * 80 + "\n\n")
            
            # Sections
            for section in report.sections:
                f.write(f"{section.title.upper()}\n")
                f.write("-" * len(section.title) + "\n\n")
                
                # Content
                for key, value in section.content.items():
                    if isinstance(value, dict):
                        f.write(f"{key}:\n")
                        for sub_key, sub_value in value.items():
                            f.write(f"  {sub_key}: {sub_value}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                
                # Notes
                if section.notes:
                    f.write("\nNotes:\n")
                    for note in section.notes:
                        f.write(f"- {note}\n")
                
                f.write("\n")
    
    # Additional helper methods for other report types would go here...
    def _create_weekly_summary(self, events: List, start: datetime, end: datetime) -> ReportSection:
        """Create weekly summary section."""
        # Implementation details...
        pass
    
    def _analyze_violations(self, events: List) -> ReportSection:
        """Analyze compliance violations."""
        # Implementation details...
        pass
    
    def _create_market_comparison(self, events: List) -> ReportSection:
        """Create market comparison section."""
        # Implementation details...
        pass
    
    def _analyze_weekly_trends(self, events: List, start: datetime) -> ReportSection:
        """Analyze weekly trends."""
        # Implementation details...
        pass
    
    def _create_executive_summary(self, events: List, start: datetime, end: datetime) -> ReportSection:
        """Create executive summary for monthly report."""
        # Implementation details...
        pass
    
    def _detailed_compliance_analysis(self, events: List) -> ReportSection:
        """Detailed compliance analysis."""
        # Implementation details...
        pass
    
    def _create_tax_compliance_section(self, events: List) -> ReportSection:
        """Create tax compliance section."""
        # Implementation details...
        pass
    
    def _analyze_manual_interventions(self, events: List) -> ReportSection:
        """Analyze manual interventions."""
        # Implementation details...
        pass
    
    def _analyze_system_performance(self, events: List) -> ReportSection:
        """Analyze system performance."""
        # Implementation details...
        pass
    
    def _create_attestation_section(self, events: List, start: datetime, end: datetime) -> ReportSection:
        """Create regulatory attestation section."""
        # Implementation details...
        pass
    
    def _calculate_tax_summary(self, events: List, market: Market) -> ReportSection:
        """Calculate tax summary."""
        # Implementation details...
        pass
    
    def _tax_by_category(self, events: List, market: Market) -> ReportSection:
        """Tax breakdown by category."""
        # Implementation details...
        pass
    
    def _municipal_tax_breakdown(self, events: List) -> ReportSection:
        """Municipal tax breakdown."""
        # Implementation details...
        pass