"""Tablas de referencia de estados: FIPS, abreviatura y huso horario.

Las fuentes del plan identifican los estados de tres formas distintas (EPHT y
gridMET por FIPS, el coverage_history de EAGLE-I por abreviatura postal, los
CSV de EAGLE-I por nombre). Esta tabla es el unico lugar donde se traducen.

Huso horario: el *predominante* de cada estado. Sirve para llevar marcas UTC
(EAGLE-I) al dia civil local en que la gente vive el calor. En estados
partidos (TX, FL, KS, NE, ND, SD, ID, OR, KY, TN, IN, MI) los condados del
huso minoritario quedan desplazados una hora; se declara como limitacion en
vez de fingir una precision que requeriria la geometria de los husos.
"""

from __future__ import annotations

# fips -> (abreviatura, nombre, huso IANA predominante)
STATES: dict[str, tuple[str, str, str]] = {
    "01": ("AL", "Alabama", "America/Chicago"),
    "02": ("AK", "Alaska", "America/Anchorage"),
    "04": ("AZ", "Arizona", "America/Phoenix"),
    "05": ("AR", "Arkansas", "America/Chicago"),
    "06": ("CA", "California", "America/Los_Angeles"),
    "08": ("CO", "Colorado", "America/Denver"),
    "09": ("CT", "Connecticut", "America/New_York"),
    "10": ("DE", "Delaware", "America/New_York"),
    "11": ("DC", "District of Columbia", "America/New_York"),
    "12": ("FL", "Florida", "America/New_York"),
    "13": ("GA", "Georgia", "America/New_York"),
    "15": ("HI", "Hawaii", "Pacific/Honolulu"),
    "16": ("ID", "Idaho", "America/Boise"),
    "17": ("IL", "Illinois", "America/Chicago"),
    "18": ("IN", "Indiana", "America/Indiana/Indianapolis"),
    "19": ("IA", "Iowa", "America/Chicago"),
    "20": ("KS", "Kansas", "America/Chicago"),
    "21": ("KY", "Kentucky", "America/New_York"),
    "22": ("LA", "Louisiana", "America/Chicago"),
    "23": ("ME", "Maine", "America/New_York"),
    "24": ("MD", "Maryland", "America/New_York"),
    "25": ("MA", "Massachusetts", "America/New_York"),
    "26": ("MI", "Michigan", "America/Detroit"),
    "27": ("MN", "Minnesota", "America/Chicago"),
    "28": ("MS", "Mississippi", "America/Chicago"),
    "29": ("MO", "Missouri", "America/Chicago"),
    "30": ("MT", "Montana", "America/Denver"),
    "31": ("NE", "Nebraska", "America/Chicago"),
    "32": ("NV", "Nevada", "America/Los_Angeles"),
    "33": ("NH", "New Hampshire", "America/New_York"),
    "34": ("NJ", "New Jersey", "America/New_York"),
    "35": ("NM", "New Mexico", "America/Denver"),
    "36": ("NY", "New York", "America/New_York"),
    "37": ("NC", "North Carolina", "America/New_York"),
    "38": ("ND", "North Dakota", "America/Chicago"),
    "39": ("OH", "Ohio", "America/New_York"),
    "40": ("OK", "Oklahoma", "America/Chicago"),
    "41": ("OR", "Oregon", "America/Los_Angeles"),
    "42": ("PA", "Pennsylvania", "America/New_York"),
    "44": ("RI", "Rhode Island", "America/New_York"),
    "45": ("SC", "South Carolina", "America/New_York"),
    "46": ("SD", "South Dakota", "America/Chicago"),
    "47": ("TN", "Tennessee", "America/Chicago"),
    "48": ("TX", "Texas", "America/Chicago"),
    "49": ("UT", "Utah", "America/Denver"),
    "50": ("VT", "Vermont", "America/New_York"),
    "51": ("VA", "Virginia", "America/New_York"),
    "53": ("WA", "Washington", "America/Los_Angeles"),
    "54": ("WV", "West Virginia", "America/New_York"),
    "55": ("WI", "Wisconsin", "America/Chicago"),
    "56": ("WY", "Wyoming", "America/Denver"),
    "72": ("PR", "Puerto Rico", "America/Puerto_Rico"),
}

ABBR_TO_FIPS = {abbr: fips for fips, (abbr, _, _) in STATES.items()}


def state_tz(fips: str) -> str:
    return STATES[fips.zfill(2)][2]
