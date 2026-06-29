from __future__ import annotations

ROLE_FAMILIES = ["GK", "CB", "FB", "MID", "WINGER", "CF", "UNKNOWN"]

_POSITION_RULES = {
    "GK": ("Goalkeeper",),
    "CB": ("Center Back", "Centre Back", "Left Center Back", "Right Center Back"),
    "FB": ("Left Back", "Right Back", "Left Wing Back", "Right Wing Back", "Wing Back"),
    "MID": ("Defensive Midfield", "Center Midfield", "Centre Midfield", "Left Center Midfield", "Right Center Midfield", "Attacking Midfield"),
    "WINGER": ("Left Wing", "Right Wing", "Left Midfield", "Right Midfield"),
    "CF": ("Center Forward", "Centre Forward", "Striker", "Secondary Striker"),
}

def infer_role_family(position_text: str | None) -> str:
    if not position_text:
        return "UNKNOWN"
    text = str(position_text).lower()
    for family, names in _POSITION_RULES.items():
        if any(name.lower() in text for name in names):
            return family
    if "goalkeeper" in text:
        return "GK"
    if "back" in text and "wing" not in text:
        return "CB" if "center" in text or "centre" in text else "FB"
    if "midfield" in text:
        return "WINGER" if "left" in text or "right" in text else "MID"
    if "wing" in text:
        return "WINGER"
    if "forward" in text or "striker" in text:
        return "CF"
    return "UNKNOWN"
