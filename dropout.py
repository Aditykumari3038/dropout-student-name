import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any


# ──────────────────────────────────────────────
# FEATURE DEFINITIONS
# ──────────────────────────────────────────────

FEATURE_WEIGHTS = {
    "attendance_rate":       0.22,
    "avg_marks":             0.18,
    "family_income_level":   0.14,
    "parental_education":    0.10,
    "lms_engagement_score":  0.09,
    "distance_to_school_km": 0.08,
    "failed_subjects":       0.07,
    "extra_activities":      0.04,
    "health_issues":         0.04,
    "has_part_time_job":     0.04,
}

FEATURE_LABELS = {
    "attendance_rate":       "Attendance Rate (%)",
    "avg_marks":             "Average Marks (%)",
    "family_income_level":   "Family Income Level",
    "parental_education":    "Parental Education",
    "lms_engagement_score":  "LMS / Digital Engagement (%)",
    "distance_to_school_km": "Distance to School (km)",
    "failed_subjects":       "No. of Failed Subjects",
    "extra_activities":      "Extracurricular Activities",
    "health_issues":         "Reported Health Issues",
    "has_part_time_job":     "Has Part-Time Job",
}

RISK_THRESHOLDS = {
    "critical": 75,
    "high":     55,
    "medium":   35,
    "low":       0,
}

ROOT_CAUSE_MAP = {
    "attendance_rate": {
        "trigger_below": 70,
        "cause": "Chronic Absenteeism",
        "detail": "Attendance below 70% is the strongest single predictor of dropout. Missing class creates compounding knowledge gaps.",
        "icon": "🏫",
    },
    "avg_marks": {
        "trigger_below": 50,
        "cause": "Academic Under-Performance",
        "detail": "Marks below 50% indicate foundational gaps that, if unaddressed, lead to academic alienation and eventual withdrawal.",
        "icon": "📉",
    },
    "family_income_level": {
        "trigger_below": 2,
        "cause": "Financial Hardship",
        "detail": "Low family income forces students into labor roles, reducing study time and increasing dropout likelihood by ~3×.",
        "icon": "💸",
    },
    "lms_engagement_score": {
        "trigger_below": 40,
        "cause": "Digital Disengagement",
        "detail": "Low LMS usage signals early mental dropout — the student has mentally checked out before physically leaving.",
        "icon": "💻",
    },
    "failed_subjects": {
        "trigger_above": 1,
        "cause": "Subject Failure Accumulation",
        "detail": "Each failed subject reduces confidence and increases the probability of repeating a year, a major dropout trigger.",
        "icon": "❌",
    },
    "distance_to_school_km": {
        "trigger_above": 10,
        "cause": "Geographic Barrier",
        "detail": "Students traveling >10 km face extreme fatigue and transport costs, significantly impacting attendance.",
        "icon": "🚌",
    },
    "has_part_time_job": {
        "trigger_above": 0,
        "cause": "Employment Conflict",
        "detail": "Part-time employment competes directly with study time, increasing dropout risk by ~40% in secondary students.",
        "icon": "⚙️",
    },
    "health_issues": {
        "trigger_above": 0,
        "cause": "Health & Wellbeing",
        "detail": "Physical or mental health issues reduce class participation and increase absenteeism — a silent dropout driver.",
        "icon": "🏥",
    },
}


# ──────────────────────────────────────────────
# CORE PREDICTION ENGINE
# ──────────────────────────────────────────────

def calculate_risk_score(student: Dict[str, Any]) -> float:
    """
    Rule-based weighted risk scoring engine.
    Returns a float 0–100 where 100 = highest dropout risk.
    """
    score = 0.0

    # Attendance (inverse — low attendance = high risk)
    att = student.get("attendance_rate", 80)
    score += FEATURE_WEIGHTS["attendance_rate"] * max(0, (100 - att)) / 100 * 100

    # Marks (inverse)
    marks = student.get("avg_marks", 60)
    score += FEATURE_WEIGHTS["avg_marks"] * max(0, (100 - marks)) / 100 * 100

    # Family income (1=very low, 5=high) — inverse
    income = student.get("family_income_level", 3)
    score += FEATURE_WEIGHTS["family_income_level"] * max(0, (5 - income)) / 4 * 100

    # Parental education (1=none, 5=postgrad) — inverse
    par_edu = student.get("parental_education", 3)
    score += FEATURE_WEIGHTS["parental_education"] * max(0, (5 - par_edu)) / 4 * 100

    # LMS engagement (inverse)
    lms = student.get("lms_engagement_score", 60)
    score += FEATURE_WEIGHTS["lms_engagement_score"] * max(0, (100 - lms)) / 100 * 100

    # Distance (0–50 km normalized)
    dist = min(student.get("distance_to_school_km", 5), 50)
    score += FEATURE_WEIGHTS["distance_to_school_km"] * dist / 50 * 100

    # Failed subjects (0–6 normalized)
    fails = min(student.get("failed_subjects", 0), 6)
    score += FEATURE_WEIGHTS["failed_subjects"] * fails / 6 * 100

    # Extra activities (boolean inverse)
    extra = student.get("extra_activities", 1)
    score += FEATURE_WEIGHTS["extra_activities"] * (1 - extra) * 100

    # Health issues (boolean)
    health = student.get("health_issues", 0)
    score += FEATURE_WEIGHTS["health_issues"] * health * 100

    # Part-time job (boolean)
    job = student.get("has_part_time_job", 0)
    score += FEATURE_WEIGHTS["has_part_time_job"] * job * 100

    return round(min(score, 100), 1)


def get_risk_level(score: float) -> Tuple[str, str, str]:
    """Returns (level_name, color_hex, emoji)."""
    if score >= RISK_THRESHOLDS["critical"]:
        return "CRITICAL", "#ff4d4d", "🔴"
    elif score >= RISK_THRESHOLDS["high"]:
        return "HIGH", "#ff8c00", "🟠"
    elif score >= RISK_THRESHOLDS["medium"]:
        return "MEDIUM", "#f5c518", "🟡"
    else:
        return "LOW", "#2ecc71", "🟢"


def get_feature_contributions(student: Dict[str, Any]) -> List[Dict]:
    """
    SHAP-style feature importance breakdown.
    Returns list of {feature, label, contribution, weight, direction}.
    """
    contributions = []

    raw = {
        "attendance_rate":       (100 - student.get("attendance_rate", 80)) / 100,
        "avg_marks":             (100 - student.get("avg_marks", 60)) / 100,
        "family_income_level":   (5 - student.get("family_income_level", 3)) / 4,
        "parental_education":    (5 - student.get("parental_education", 3)) / 4,
        "lms_engagement_score":  (100 - student.get("lms_engagement_score", 60)) / 100,
        "distance_to_school_km": min(student.get("distance_to_school_km", 5), 50) / 50,
        "failed_subjects":       min(student.get("failed_subjects", 0), 6) / 6,
        "extra_activities":      1 - student.get("extra_activities", 1),
        "health_issues":         student.get("health_issues", 0),
        "has_part_time_job":     student.get("has_part_time_job", 0),
    }

    for feat, weight in FEATURE_WEIGHTS.items():
        contrib = weight * raw[feat] * 100
        contributions.append({
            "feature":      feat,
            "label":        FEATURE_LABELS[feat],
            "contribution": round(contrib, 2),
            "weight":       weight,
            "raw_value":    raw[feat],
            "direction":    "risk" if raw[feat] > 0.5 else "safe",
        })

    contributions.sort(key=lambda x: x["contribution"], reverse=True)
    return contributions


def identify_root_causes(student: Dict[str, Any]) -> List[Dict]:
    """Identify which root causes are active for this student."""
    active = []
    for feat, cfg in ROOT_CAUSE_MAP.items():
        val = student.get(feat, None)
        if val is None:
            continue
        triggered = False
        if "trigger_below" in cfg and val < cfg["trigger_below"]:
            triggered = True
        if "trigger_above" in cfg and val > cfg["trigger_above"]:
            triggered = True
        if triggered:
            active.append({
                "feature": feat,
                "cause":   cfg["cause"],
                "detail":  cfg["detail"],
                "icon":    cfg["icon"],
                "value":   val,
            })
    return active


# ──────────────────────────────────────────────
# INTERVENTION ENGINE
# ──────────────────────────────────────────────

INTERVENTIONS = {
    "attendance_rate": {
        "title": "Attendance Recovery Program",
        "priority": "Immediate",
        "color": "#ff4d4d",
        "icon": "📅",
        "actions": [
            "Assign a dedicated student mentor for weekly check-ins",
            "Set up automated SMS alerts to parents after 2 consecutive absences",
            "Offer flexible scheduling or blended learning options",
            "Connect with transport support schemes if distance is a factor",
        ],
    },
    "avg_marks": {
        "title": "Academic Support Plan",
        "priority": "High",
        "color": "#ff8c00",
        "icon": "📚",
        "actions": [
            "Schedule bi-weekly remedial classes in weak subjects",
            "Assign peer tutoring partner from top-performing students",
            "Provide structured study guides and past paper practice",
            "Implement micro-assessment checkpoints every 2 weeks",
        ],
    },
    "family_income_level": {
        "title": "Financial Aid & Scholarship Connect",
        "priority": "High",
        "color": "#ff8c00",
        "icon": "💰",
        "actions": [
            "Apply for government scholarship schemes (NSP, State Merit Awards)",
            "Connect with NGO education funds in the district",
            "Explore fee waiver or installment plan with institution",
            "Provide subsidized meals and stationery through school welfare",
        ],
    },
    "lms_engagement_score": {
        "title": "Digital Re-Engagement Initiative",
        "priority": "Medium",
        "color": "#f5c518",
        "icon": "💻",
        "actions": [
            "Assign weekly digital learning targets with gamified rewards",
            "Provide device access through school computer lab after hours",
            "Pair with a digitally confident peer for LMS navigation help",
            "Faculty to send personalized video messages to re-engage",
        ],
    },
    "failed_subjects": {
        "title": "Remedial Academic Intervention",
        "priority": "Immediate",
        "color": "#ff4d4d",
        "icon": "🔁",
        "actions": [
            "Enroll in supplementary exam preparation course",
            "Create a subject-specific remediation roadmap",
            "Weekly 1:1 session with subject teacher",
            "Adjust exam strategy and attempt prioritization",
        ],
    },
    "distance_to_school_km": {
        "title": "Transportation & Logistics Support",
        "priority": "Medium",
        "color": "#f5c518",
        "icon": "🚌",
        "actions": [
            "Apply for government school bus or transport allowance",
            "Explore hostel/residential facility if distance > 20 km",
            "Connect with local carpooling initiatives among students",
            "Offer hybrid attendance model for remote days",
        ],
    },
    "has_part_time_job": {
        "title": "Work-Study Balance Counseling",
        "priority": "Medium",
        "color": "#f5c518",
        "icon": "⚖️",
        "actions": [
            "Counsel student and family on long-term ROI of completing education",
            "Explore part-time scholarships that replace the need for employment",
            "Offer flexible class timing (morning/evening batch if available)",
            "Connect with school-affiliated skill stipend programs",
        ],
    },
    "health_issues": {
        "title": "Health & Wellbeing Support",
        "priority": "High",
        "color": "#ff8c00",
        "icon": "🏥",
        "actions": [
            "Refer to school counselor for mental health screening",
            "Connect with district health center for medical support",
            "Offer assignment extensions and missed-class catch-up plan",
            "Set up a student wellness check-in routine",
        ],
    },
}


def generate_interventions(root_causes: List[Dict], risk_score: float) -> List[Dict]:
    """Generate prioritized interventions based on root causes."""
    interventions = []
    seen = set()
    for cause in root_causes:
        feat = cause["feature"]
        if feat in INTERVENTIONS and feat not in seen:
            seen.add(feat)
            interventions.append(INTERVENTIONS[feat])

    # Sort: Immediate first, then High, then Medium
    priority_order = {"Immediate": 0, "High": 1, "Medium": 2, "Low": 3}
    interventions.sort(key=lambda x: priority_order.get(x["priority"], 99))

    # If high risk but no specific cause matched, add generic
    if risk_score >= 60 and not interventions:
        interventions.append({
            "title": "Comprehensive Student Review",
            "priority": "High",
            "color": "#ff8c00",
            "icon": "🔍",
            "actions": [
                "Schedule full academic counseling session",
                "Conduct home visit or parent interview",
                "Review all subject performance and attendance together",
                "Create a personalized 30-day recovery action plan",
            ],
        })

    return interventions


# ──────────────────────────────────────────────
# ANALYTICS DATA GENERATOR
# ──────────────────────────────────────────────

def generate_analytics_data() -> Dict[str, Any]:
    """
    Returns synthetic analytics datasets for the Analytics dashboard.
    In production, replace with real DB queries.
    """
    np.random.seed(42)

    # Monthly dropout trend (last 12 months)
    months = ["May'24","Jun","Jul","Aug","Sep","Oct","Nov","Dec","Jan'25","Feb","Mar","Apr"]
    dropout_trend = [18, 16, 14, 17, 15, 13, 12, 14, 11, 10, 9, 8]
    intervention_rate = [20, 22, 28, 30, 35, 40, 45, 48, 55, 58, 62, 67]

    # Risk distribution
    risk_dist = {
        "Critical (75-100)": 8,
        "High (55-74)":      17,
        "Medium (35-54)":    31,
        "Low (0-34)":        44,
    }

    # Feature importance (global model)
    global_importance = {k: v * 100 for k, v in FEATURE_WEIGHTS.items()}

    # Dropout causes breakdown
    causes_breakdown = {
        "Financial Hardship":     40,
        "Academic Alienation":    25,
        "Family Issues":          15,
        "Geographic Barrier":     10,
        "Health Issues":           8,
        "Digital Disengagement":   2,
    }

    # Grade-wise dropout rate
    grade_dropout = {
        "Grade 6":  4.2,
        "Grade 7":  5.1,
        "Grade 8":  7.8,
        "Grade 9": 12.6,
        "Grade 10":16.3,
        "Grade 11":11.4,
        "Grade 12": 8.9,
    }

    # Regional comparison
    regional = {
        "Bihar":        23.4,
        "Assam":        21.8,
        "Rajasthan":    18.2,
        "Uttar Pradesh":17.1,
        "Jharkhand":    16.8,
        "Gujarat":       9.4,
        "Maharashtra":   8.1,
        "Karnataka":     7.3,
        "Tamil Nadu":    5.6,
        "Kerala":        2.4,
    }

    # Gender breakdown
    gender_data = {
        "Male":  {"dropout_rate": 13.2, "enrolled": 52},
        "Female":{"dropout_rate": 11.9, "enrolled": 48},
    }

    # Intervention success rates
    intervention_success = {
        "Academic Mentoring":       72,
        "Financial Aid Connection": 68,
        "Family Counseling":        61,
        "Transport Support":        55,
        "Digital Re-engagement":    48,
        "Health Support":           63,
    }

    return {
        "months":              months,
        "dropout_trend":       dropout_trend,
        "intervention_rate":   intervention_rate,
        "risk_dist":           risk_dist,
        "global_importance":   global_importance,
        "causes_breakdown":    causes_breakdown,
        "grade_dropout":       grade_dropout,
        "regional":            regional,
        "gender_data":         gender_data,
        "intervention_success":intervention_success,
        "total_students":      1247,
        "at_risk":             312,
        "interventions_active":189,
        "success_rate":        67.4,
        "model_accuracy":      87.4,
        "model_precision":     84.1,
        "model_recall":        89.2,
        "model_f1":            86.6,
    }


# ──────────────────────────────────────────────
# FULL PREDICTION PIPELINE
# ──────────────────────────────────────────────

def run_full_prediction(student: Dict[str, Any]) -> Dict[str, Any]:
    """
    Master function: run complete prediction pipeline on a student dict.
    Returns all results needed by the frontend.
    """
    risk_score    = calculate_risk_score(student)
    level, color, emoji = get_risk_level(risk_score)
    contributions = get_feature_contributions(student)
    root_causes   = identify_root_causes(student)
    interventions = generate_interventions(root_causes, risk_score)

    # Dropout probability (sigmoid-like mapping)
    dropout_prob = round(1 / (1 + np.exp(-0.08 * (risk_score - 50))) * 100, 1)

    # Months-to-dropout estimate
    if risk_score >= 75:
        months_estimate = "1–2 months"
    elif risk_score >= 55:
        months_estimate = "3–5 months"
    elif risk_score >= 35:
        months_estimate = "6–12 months"
    else:
        months_estimate = "> 12 months / Unlikely"

    return {
        "risk_score":       risk_score,
        "risk_level":       level,
        "risk_color":       color,
        "risk_emoji":       emoji,
        "dropout_prob":     dropout_prob,
        "months_estimate":  months_estimate,
        "contributions":    contributions,
        "root_causes":      root_causes,
        "interventions":    interventions,
        "top_3_factors":    contributions[:3],
    }


# ──────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "attendance_rate":       58,
        "avg_marks":             42,
        "family_income_level":   1,
        "parental_education":    2,
        "lms_engagement_score":  30,
        "distance_to_school_km": 18,
        "failed_subjects":       3,
        "extra_activities":      0,
        "health_issues":         1,
        "has_part_time_job":     1,
    }
    result = run_full_prediction(sample)
    print(f"Risk Score : {result['risk_score']}")
    print(f"Risk Level : {result['risk_level']}")
    print(f"Dropout Prob: {result['dropout_prob']}%")
    print(f"Timeline   : {result['months_estimate']}")
    print(f"Top Cause  : {result['root_causes'][0]['cause'] if result['root_causes'] else 'N/A'}")