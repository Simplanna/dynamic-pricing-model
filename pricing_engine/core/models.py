from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict
from enum import Enum


class ProductCategory(Enum):
    FLOWER = "flower"
    EDIBLES = "edibles"
    CONCENTRATES = "concentrates"
    VAPE = "vape"
    TOPICALS = "topicals"
    PREROLLS = "prerolls"
    TINCTURES = "tinctures"
    ACCESSORIES = "accessories"


class MarketType(Enum):
    MASSACHUSETTS = "MA"
    RHODE_ISLAND = "RI"


class CompetitorDensity(Enum):
    SPARSE = "sparse"  # < 5 competitors in radius
    MODERATE = "moderate"  # 5-15 competitors
    DENSE = "dense"  # > 15 competitors


@dataclass
class Product:
    """Core product model with cannabis-specific attributes"""
    id: str
    name: str
    category: ProductCategory
    brand: str
    strain: Optional[str] = None
    thc_percentage: float = 0.0
    cbd_percentage: float = 0.0
    weight_grams: Optional[float] = None
    unit_count: Optional[int] = None
    description: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if self.thc_percentage < 0 or self.thc_percentage > 40:
            raise ValueError(f"THC percentage must be between 0 and 40, got {self.thc_percentage}")
        if self.cbd_percentage < 0 or self.cbd_percentage > 30:
            raise ValueError(f"CBD percentage must be between 0 and 30, got {self.cbd_percentage}")


@dataclass
class CompetitorPrice:
    """Competitor pricing data with distance weighting"""
    dispensary_id: str
    dispensary_name: str
    product_match: Dict[str, any]  # Matched product details
    price: float
    distance_miles: float
    platform: str  # Dutchie, iHeartJane, Meadow, Tymber
    scraped_at: datetime
    match_confidence: float  # 0-1 confidence score
    url: Optional[str] = None
    
    def __post_init__(self):
        if self.price < 1 or self.price > 1000:
            raise ValueError(f"Price must be between $1 and $1000, got ${self.price}")
        if self.match_confidence < 0 or self.match_confidence > 1:
            raise ValueError(f"Match confidence must be between 0 and 1, got {self.match_confidence}")
    
    @property
    def distance_weight(self) -> float:
        """Calculate distance-based influence weight (inverse square)"""
        if self.distance_miles <= 0:
            return 1.0
        return 1.0 / (1 + (self.distance_miles / 10) ** 2)


@dataclass
class Dispensary:
    """Dispensary location and market info"""
    id: str
    name: str
    address: str
    city: str
    state: MarketType
    latitude: float
    longitude: float
    platform: str
    menu_url: str
    is_medical: bool = False
    is_recreational: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class MarketSegment:
    """Market segmentation data"""
    state: MarketType
    density: CompetitorDensity
    avg_price_per_gram: float
    competitor_count: int
    radius_miles: float
    date_analyzed: datetime = field(default_factory=datetime.now)
    
    @property
    def search_radius(self) -> float:
        """Dynamic radius based on competitor density"""
        if self.density == CompetitorDensity.SPARSE:
            return 15.0
        elif self.density == CompetitorDensity.MODERATE:
            return 12.0
        else:  # DENSE
            return 10.0


@dataclass
class PricingHistory:
    """Historical pricing data for trend analysis"""
    product_id: str
    dispensary_id: str
    price: float
    recorded_at: datetime
    inventory_level: Optional[int] = None
    days_since_harvest: Optional[int] = None
    
    
@dataclass
class ProductMatch:
    """Result of matching products across dispensaries"""
    source_product: Product
    matched_product: Dict[str, any]
    confidence_score: float
    match_criteria: Dict[str, bool]  # Which criteria matched
    price_difference: Optional[float] = None
    
    def is_valid_match(self, threshold: float = 0.8) -> bool:
        """Check if match meets confidence threshold"""
        return self.confidence_score >= threshold