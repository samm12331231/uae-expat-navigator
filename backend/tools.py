import re
from typing import List, Dict, Any
from langchain_core.tools import tool

from backend.config import SOUTH_ASIAN_COUNTRIES


@tool
def fee_calculator(fees: List[str]) -> Dict[str, Any]:
    """Calculate the total of a list of AED fee strings, deduplicating by amount.
    
    Args:
        fees: List of fee strings like ["AED 200", "AED 180", "AED 20"]
    
    Returns:
        Dict with total, breakdown, and currency.
    """
    total = 0.0
    breakdown = []
    seen_amounts = set()
    pattern = re.compile(r"AED[\s]*([\d,]+(?:\.\d+)?)", re.IGNORECASE)

    for fee_str in fees:
        match = pattern.search(fee_str)
        if not match:
            continue
        amount_str = match.group(1).replace(",", "")
        amount = float(amount_str)

        # Skip amounts under 10 — likely page numbers or references
        if amount < 10:
            continue

        if amount not in seen_amounts:
            seen_amounts.add(amount)
            total += amount
            breakdown.append({"item": fee_str, "amount": amount})

    return {
        "total": round(total, 2),
        "breakdown": breakdown,
        "currency": "AED",
    }


@tool
def eligibility_checker(country: str, process_type: str) -> Dict[str, Any]:
    """Check if a country is eligible for direct UAE driving-license exchange.
    
    Args:
        country: Country name like "India" or "United Kingdom"
        process_type: The UAE process, e.g. "driving_license_exchange"
    
    Returns:
        Dict with eligibility status, reason, and alternative path.
    """
    country_title = country.title()

    eligible_for_direct = {
        "driving_license_exchange": [
            "Australia", "Austria", "Bahrain", "Belgium", "Canada", "Denmark",
            "Finland", "France", "Germany", "Greece", "Ireland", "Italy", "Japan",
            "South Korea", "Kuwait", "Netherlands", "New Zealand", "Norway", "Poland",
            "Portugal", "Qatar", "Romania", "Saudi Arabia", "South Africa", "Spain",
            "Sweden", "Switzerland", "Turkey", "United Kingdom", "United States",
        ]
    }

    south_asian = SOUTH_ASIAN_COUNTRIES
    is_eligible = country_title in eligible_for_direct.get(process_type, [])
    is_south_asian = country_title in south_asian

    if is_eligible:
        reason = (
            f"{country_title} is eligible for direct {process_type}. "
            "Valid original license + eye test required."
        )
    elif is_south_asian:
        reason = (
            f"{country_title} is not eligible for direct exchange. "
            "However, UAE residents with a valid home-country license can apply for the "
            "Golden Chance (direct road test) or reduced training hours "
            "(typically 10 instead of 20)."
        )
    else:
        reason = (
            f"{country_title} is not on the exception list for {process_type}. "
            "Alternative paths may be available through RTA-approved driving institutes."
        )

    return {
        "eligible": is_eligible,
        "country": country_title,
        "process_type": process_type,
        "reason": reason,
        "alternative_path": (
            "Golden Chance (direct road test) or reduced classes"
            if is_south_asian
            else "Check with RTA driving institutes"
        ),
    }