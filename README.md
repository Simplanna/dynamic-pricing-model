# Cannabis Dynamic Pricing Engine - Phase 1

## Overview
Dynamic pricing system for cannabis dispensaries in Rhode Island and Massachusetts, optimizing prices based on market conditions, inventory levels, and competitor analysis.

## Phase 1 Completed Components

### 1. Core Data Models (`pricing_engine/core/models.py`)
- **Product**: Cannabis product with THC/CBD percentages, categories, and validation
- **CompetitorPrice**: Competitor pricing with distance-based weighting
- **MarketSegment**: Market analysis with dynamic radius calculation
- **ProductMatch**: Confidence-based product matching across dispensaries

### 2. ScraperBee Integration (`pricing_engine/scrapers/`)
- **ScraperBeeClient**: API integration for web scraping with rate limiting
- **DispensaryMenuScraper**: Platform detection and menu extraction
- **Platform Parsers**: Dutchie, iHeartJane, Meadow, and Tymber support

### 3. Data Validation (`pricing_engine/utils/validation.py`)
- Price validation ($1-$1000 range)
- THC/CBD percentage validation (0-40% THC, 0-30% CBD)
- Category normalization and fuzzy matching
- Product matching with 80% confidence threshold

### 4. Market Analysis (`pricing_engine/analysis/market_analysis.py`)
- Cross-border shopping pattern detection
- Seasonal pricing analysis (4/20, summer tourism, holidays)
- Competitor density calculation with dynamic radius
- Inventory age impact analysis

### 5. Streamlit Dashboard (`app.py`)
- Real-time market overview
- Competitor scraping interface
- Data validation dashboard
- Product matching visualization
- Market analysis charts

## Installation

```bash
# Clone repository
git clone <repository-url>
cd "Dynamic Pricing Model"

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

```bash
# Run Streamlit dashboard
streamlit run app.py

# Run Phase 1 tests
python test_phase1.py
```

## Key Features

### Market Intelligence
- **MA Market**: 360 dispensaries, $4.44/gram average (62% decline from 2018)
- **RI Market**: 7 dispensaries, $12.40/gram average, premium pricing maintained
- **Cross-Border Analysis**: 20-mile border zone price compression effects

### Data Quality Assurance
- Automated validation of pricing, cannabinoid levels, and categories
- Confidence-based product matching across platforms
- Real-time data quality reporting

### Competitive Analysis
- Dynamic radius adjustment (10-15 miles based on density)
- Distance-weighted competitor influence
- Platform-agnostic menu parsing

## Architecture

```
pricing_engine/
├── core/           # Data models and business logic
├── scrapers/       # Web scraping and parsing
├── data/           # Data schemas and storage
├── analysis/       # Market analysis algorithms
└── utils/          # Validation and utilities
```

## Phase 1 Achievements

✅ Established data models with cannabis-specific attributes
✅ Integrated ScraperBee API for competitor monitoring
✅ Built parsers for 4 major e-commerce platforms
✅ Implemented 80%+ accuracy product matching
✅ Created market analysis framework
✅ Developed interactive Streamlit dashboard
✅ Set up comprehensive data validation pipeline

## Next Steps (Phase 2)

1. Implement core pricing algorithms
2. Add inventory age optimization
3. Build demand forecasting models
4. Create A/B testing framework
5. Develop API endpoints for POS integration

---

**PHASE 1 COMPLETE**