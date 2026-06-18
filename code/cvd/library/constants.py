"""
Shared constants and configuration for the CVD library.

This module centralizes constants so that split modules can import them
without creating circular dependencies.
"""
from typing import Tuple, List


CRITICAL_CHECKUP_COLS: List[str] = [
    "user_id", "checkup_id", "checkup_date",
    "site_id", "account_id", "prescription_id"
]

OUTLIER_STRATEGY: str = "cap"
OUTLIER_METHOD: str = "quantile"
CLIP_QUANTILES: Tuple[float, float] = (0.005, 0.995)
IQR_K: float = 1.5
MAD_Z: float = 5.0


BOUNDS = {
    "height": (80, 230),
    "weight": (20, 250),
    "bmi": (8, 70),
    "bp_sys": (60, 260),
    "bp_dia": (30, 200),
    "pulse_rate": (30, 220),
    "temperature": (33, 43),
    "oxygen_of_blood": (50, 100),
    "waist": (30, 200),
    "hip": (30, 200),
    "whr_calc": (0.6, 2.0),
    "blood_glucose": (30, 800),
    "cholesterol": (50, 500),
    "uric_acid": (1, 20),
}

LAB_NUMERIC_COLS: List[str] = ["fbs", "chol_total", "hdl", "ldl", "tg"]


BD_DISTRICT_CANON = [
    "Dhaka", "Gazipur", "Kishoreganj", "Manikganj", "Munshiganj", "Narayanganj", "Narsingdi",
    "Tangail", "Faridpur", "Gopalganj", "Madaripur", "Rajbari", "Shariatpur",
    "Chattogram", "Coxs Bazar", "Feni", "Noakhali", "Lakshmipur", "Brahmanbaria", "Cumilla", "Chandpur",
    "Rangamati", "Khagrachari", "Bandarban",
    "Rajshahi", "Natore", "Chapainawabganj", "Naogaon", "Joypurhat", "Bogura", "Sirajganj", "Pabna",
    "Khulna", "Bagerhat", "Satkhira", "Jashore", "Jhenaidah", "Narail", "Magura", "Kushtia", "Chuadanga", "Meherpur",
    "Barishal", "Bhola", "Patuakhali", "Pirojpur", "Jhalokati", "Barguna",
    "Sylhet", "Moulvibazar", "Habiganj", "Sunamganj",
    "Rangpur", "Dinajpur", "Thakurgaon", "Nilphamari", "Lalmonirhat", "Panchagarh", "Kurigram", "Gaibandha",
    "Mymensingh", "Jamalpur", "Sherpur", "Netrakona"
]

ALIAS = {
    "Chittagong": "Chattogram",
    "Comilla": "Cumilla",
    "Cox": "Coxs Bazar",
    "Cox'": "Coxs Bazar",
    "Coxs": "Coxs Bazar",
    "Bramhanbaria": "Brahmanbaria",
    "Bogra": "Bogura",
    "Jessor": "Jashore",
    "Jessore": "Jashore",
    "Barisal": "Barishal"
}

__all__ = [
    "CRITICAL_CHECKUP_COLS",
    "OUTLIER_STRATEGY",
    "OUTLIER_METHOD",
    "CLIP_QUANTILES",
    "IQR_K",
    "MAD_Z",
    "BOUNDS",
    "LAB_NUMERIC_COLS",
    "BD_DISTRICT_CANON",
    "ALIAS",
]
