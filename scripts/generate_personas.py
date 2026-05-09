"""
Synthetic HCP Persona Generator

Generates realistic Healthcare Provider personas with:
- Demographics (specialty, region, years of practice, setting)
- Prescribing behavior (volume, brand preference, formulary access)
- Promotional engagement history (channel preferences, response rates)
- Segmentation attributes (tier, digital affinity, promotional sensitivity)

Uses published AMA physician workforce statistics as reference distributions.

Author: Fab Admasu
License: MIT
"""

import json
import random
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────
# Reference distributions (based on published AMA/AAMC data)
# ─────────────────────────────────────────────────────────────

SPECIALTIES = {
    "Internal Medicine": 0.18,
    "Family Medicine": 0.14,
    "Cardiology": 0.08,
    "Oncology": 0.06,
    "Pulmonology": 0.05,
    "Rheumatology": 0.04,
    "Dermatology": 0.05,
    "Endocrinology": 0.04,
    "Neurology": 0.06,
    "Infectious Disease": 0.05,
    "Gastroenterology": 0.05,
    "Nephrology": 0.04,
    "Pediatrics": 0.10,
    "OB/GYN": 0.06,
}

REGIONS = {
    "Northeast": 0.22,
    "Southeast": 0.24,
    "Midwest": 0.21,
    "Southwest": 0.15,
    "West": 0.18,
}

PRACTICE_SETTINGS = {
    "Academic Medical Center": 0.15,
    "Hospital-Employed": 0.30,
    "Large Group Practice": 0.20,
    "Small Group Practice": 0.20,
    "Solo Practice": 0.10,
    "Community Health Center": 0.05,
}

CHANNELS = ["Email", "Rep_Visit", "Webinar", "Conference", "Digital_Ad", "Sample"]

TIERS = {
    "Tier 1 (High Value)": 0.15,
    "Tier 2 (Medium Value)": 0.35,
    "Tier 3 (Low Value)": 0.50,
}


@dataclass
class HCPPersona:
    """A synthetic HCP persona with attributes for targeting."""
    persona_id: str
    specialty: str
    region: str
    practice_setting: str
    years_in_practice: int
    patients_per_week: int
    tier: str
    digital_affinity: float          # 0-1
    promotional_sensitivity: float   # 0-1
    formulary_access: Dict[str, bool] = field(default_factory=dict)
    channel_preferences: Dict[str, float] = field(default_factory=dict)
    engagement_history: List[Dict] = field(default_factory=list)
    annual_rx_volume: int = 0
    brand_loyalty: float = 0.0       # 0-1


def sample_from_distribution(dist: Dict[str, float]) -> str:
    """Sample a category from a probability distribution."""
    categories = list(dist.keys())
    probs = list(dist.values())
    probs = np.array(probs) / sum(probs)  # Normalize
    return np.random.choice(categories, p=probs)


def generate_persona(persona_id: int) -> HCPPersona:
    """Generate a single synthetic HCP persona."""
    specialty = sample_from_distribution(SPECIALTIES)
    region = sample_from_distribution(REGIONS)
    setting = sample_from_distribution(PRACTICE_SETTINGS)
    tier = sample_from_distribution(TIERS)
    
    # Years in practice: skewed distribution (more senior HCPs)
    years = max(1, int(np.random.lognormal(mean=2.5, sigma=0.6)))
    years = min(years, 45)
    
    # Patients per week: depends on specialty and setting
    base_patients = {"Academic Medical Center": 15, "Hospital-Employed": 20,
                     "Large Group Practice": 25, "Small Group Practice": 22,
                     "Solo Practice": 18, "Community Health Center": 28}
    patients = max(5, int(np.random.normal(
        base_patients.get(setting, 20), 5
    )))
    
    # Digital affinity: younger HCPs tend higher
    digital_base = max(0.1, 1.0 - (years / 50))
    digital_affinity = round(np.clip(
        np.random.normal(digital_base, 0.15), 0.05, 0.99
    ), 3)
    
    # Promotional sensitivity: varies
    promo_sensitivity = round(np.clip(
        np.random.beta(2, 3), 0.05, 0.95
    ), 3)
    
    # Channel preferences (sum to ~1)
    raw_prefs = {}
    for ch in CHANNELS:
        if ch == "Email":
            raw_prefs[ch] = digital_affinity * np.random.uniform(0.8, 1.2)
        elif ch == "Rep_Visit":
            raw_prefs[ch] = (1 - digital_affinity) * np.random.uniform(0.5, 1.5)
        elif ch == "Digital_Ad":
            raw_prefs[ch] = digital_affinity * np.random.uniform(0.6, 1.0)
        else:
            raw_prefs[ch] = np.random.uniform(0.2, 0.8)
    total = sum(raw_prefs.values())
    channel_preferences = {
        ch: round(v / total, 3) for ch, v in raw_prefs.items()
    }
    
    # Formulary access (brand availability)
    brands = ["Cardivex", "Immunolex", "OncoPrime", "NeuraStar",
              "RespiClear", "DermaShield", "VaxGuard", "EndoBalance"]
    formulary = {b: random.random() > 0.3 for b in brands}
    
    # Annual Rx volume
    annual_rx = int(patients * 52 * np.random.uniform(0.3, 0.7))
    
    # Brand loyalty
    loyalty = round(np.clip(np.random.beta(3, 5), 0.1, 0.9), 3)
    
    # Generate engagement history (last 12 months)
    history = []
    for month in range(1, 13):
        n_touches = np.random.poisson(lam=1.5)
        for _ in range(n_touches):
            ch = sample_from_distribution(channel_preferences)
            responded = random.random() < promo_sensitivity
            history.append({
                "month": month,
                "channel": ch,
                "responded": responded,
                "brand": random.choice(brands),
            })
    
    return HCPPersona(
        persona_id=f"HCP-{persona_id:05d}",
        specialty=specialty,
        region=region,
        practice_setting=setting,
        years_in_practice=years,
        patients_per_week=patients,
        tier=tier,
        digital_affinity=digital_affinity,
        promotional_sensitivity=promo_sensitivity,
        formulary_access=formulary,
        channel_preferences=channel_preferences,
        engagement_history=history,
        annual_rx_volume=annual_rx,
        brand_loyalty=loyalty,
    )


def generate_dataset(n_personas: int, output_dir: Path):
    """Generate full persona dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    personas = []
    for i in range(n_personas):
        p = generate_persona(i)
        personas.append(asdict(p))
    
    output_path = output_dir / "personas.jsonl"
    with open(output_path, "w") as f:
        for p in personas:
            f.write(json.dumps(p) + "\n")
    
    # Print summary statistics
    specialties = [p["specialty"] for p in personas]
    regions = [p["region"] for p in personas]
    tiers = [p["tier"] for p in personas]
    
    print(f"✅ Generated {n_personas} synthetic HCP personas")
    print(f"   📁 {output_path}")
    print(f"\n   Specialty distribution:")
    for sp in sorted(set(specialties)):
        count = specialties.count(sp)
        pct = count / len(specialties) * 100
        print(f"     {sp}: {count} ({pct:.1f}%)")
    print(f"\n   Region distribution:")
    for reg in sorted(set(regions)):
        count = regions.count(reg)
        pct = count / len(regions) * 100
        print(f"     {reg}: {count} ({pct:.1f}%)")
    print(f"\n   Tier distribution:")
    for t in sorted(set(tiers)):
        count = tiers.count(t)
        pct = count / len(tiers) * 100
        print(f"     {t}: {count} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic HCP personas for targeting"
    )
    parser.add_argument("--n_personas", type=int, default=5000)
    parser.add_argument("--output_dir", type=str, default="data")
    args = parser.parse_args()
    
    generate_dataset(args.n_personas, Path(args.output_dir))


if __name__ == "__main__":
    main()
