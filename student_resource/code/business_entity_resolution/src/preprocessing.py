"""
preprocessing.py — Enhanced text preprocessing for Business Entity Resolution.

Key improvements over v1:
  - More thorough legal suffix / abbreviation normalisation
  - Better handling of non-Latin scripts (Hindi/Devanagari question-marks)
  - Dedicated combined-text generation for TF-IDF blocking
  - Phonetic key generation (Soundex)
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# Legal suffix normalisation (order: longest match first)
# ---------------------------------------------------------------------------
# Two-word suffixes (checked first)
LEGAL_BIGRAMS = {
    "private limited": "pvt ltd", "pvt limited": "pvt ltd",
    "pvt ltd": "pvt ltd", "pvt. ltd.": "pvt ltd",
    "pvt.ltd.": "pvt ltd", "pvt. ltd": "pvt ltd",
    "pvt.ltd": "pvt ltd",
}

# Single-word suffixes
LEGAL_UNIGRAMS = {
    "incorporated": "inc", "incorporation": "inc", "inc.": "inc",
    "corporation": "corp", "corp.": "corp",
    "company": "co", "co.": "co",
    "limited": "ltd", "ltd.": "ltd",
    "private": "pvt", "pvt.": "pvt",
    "llc": "llc", "l.l.c.": "llc", "l.l.c": "llc",
    "llp": "llp", "l.l.p.": "llp", "l.l.p": "llp",
    "plc": "plc", "p.l.c.": "plc",
    "enterprises": "ent", "enterprise": "ent",
    "associates": "assoc", "association": "assoc",
    "technologies": "tech", "technology": "tech",
    "solutions": "sol", "solution": "sol",
    "services": "svc", "service": "svc",
    "international": "intl",
    "industries": "ind", "industry": "ind",
    "consultants": "consult", "consulting": "consult", "consultancy": "consult",
    "foundation": "fdn", "institute": "inst",
    "group": "grp", "holdings": "hldg", "holding": "hldg",
    "manufacturing": "mfg", "management": "mgmt",
    "construction": "const", "communications": "comm", "communication": "comm",
    "corporation": "corp", "corporate": "corp",
    "laboratories": "lab", "laboratory": "lab", "labs": "lab",
    "properties": "prop", "property": "prop",
    "investments": "inv", "investment": "inv",
    "development": "dev", "developments": "dev",
    "engineering": "eng", "engineers": "eng",
    "financial": "fin", "finance": "fin",
    "education": "edu", "educational": "edu",
    "marketing": "mkt", "logistics": "log",
    # French
    "societe": "soc", "société": "soc",
    "sarl": "sarl", "s.a.r.l.": "sarl", "s.a.r.l": "sarl",
    "sas": "sas", "s.a.s.": "sas", "s.a.s": "sas",
    "eurl": "eurl", "e.u.r.l.": "eurl",
    "sa": "sa", "s.a.": "sa",
}

# Address abbreviations
ADDRESS_ABBREVS = {
    "street": "st", "st.": "st",
    "road": "rd", "rd.": "rd",
    "avenue": "ave", "ave.": "ave",
    "boulevard": "blvd", "blvd.": "blvd",
    "drive": "dr", "dr.": "dr",
    "lane": "ln", "ln.": "ln",
    "court": "ct", "ct.": "ct",
    "place": "pl", "pl.": "pl",
    "circle": "cir", "cir.": "cir",
    "highway": "hwy", "hwy.": "hwy",
    "parkway": "pkwy", "pkwy.": "pkwy",
    "terrace": "ter", "ter.": "ter",
    "trail": "trl", "trl.": "trl",
    "way": "way",
    "apartment": "apt", "apt.": "apt",
    "suite": "ste", "ste.": "ste",
    "building": "bldg", "bldg.": "bldg",
    "floor": "fl", "fl.": "fl",
    "unit": "unit", "room": "rm", "rm.": "rm",
    "number": "no", "no.": "no", "num": "no", "#": "no",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "mount": "mt", "mt.": "mt", "fort": "ft", "ft.": "ft",
    "saint": "st", "point": "pt", "pt.": "pt",
    # Indian
    "nagar": "nagar", "colony": "colony", "sector": "sector",
    "block": "blk", "phase": "phase", "plot": "plot",
    "khasra": "kh", "kh.": "kh", "kh": "kh",
    "district": "dist", "dist.": "dist",
    "tehsil": "teh", "taluk": "tlk", "mandal": "mdl",
    "village": "vlg", "mohalla": "moh",
    # Noise words to drop
    "near": "", "opp": "", "opposite": "", "behind": "", "beside": "",
    "adjacent": "", "next": "", "front": "",
    # French
    "rue": "rue", "allée": "allee", "allee": "allee",
    "impasse": "imp", "chemin": "ch", "passage": "pass",
    "quartier": "qtr", "arrondissement": "arr",
    "cedex": "cedex",
}

# US states (full → abbreviation)
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
}

# Indian states
INDIAN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as",
    "bihar": "br", "chhattisgarh": "cg", "goa": "ga",
    "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj",
    "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk",
    "west bengal": "wb", "delhi": "dl", "new delhi": "dl",
    "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
    "jammu and kashmir": "jk", "ladakh": "la",
}


def normalize_unicode(text):
    """Strip accents, normalise to ASCII-safe lowercase, remove non-printable."""
    if not isinstance(text, str):
        return ""
    # Replace question marks that indicate encoding issues
    text = re.sub(r"\?{2,}", " ", text)
    # NFKD decomposition + strip combining marks
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Remove non-printable / control characters
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
    return text.lower().strip()


def normalize_business_name(name):
    """Normalise a business name for comparison."""
    if not isinstance(name, str) or not name.strip():
        return ""
    name = normalize_unicode(name)
    # Remove URLs
    name = re.sub(r"https?://\S+", "", name)
    name = re.sub(r"\S+\.(com|org|net|co\.in|co|io|in)\b", "", name)
    # & → and
    name = name.replace("&", " and ")
    # Remove punctuation except hyphens
    name = re.sub(r"[^\w\s-]", " ", name)
    # Normalise legal suffixes (bigrams first, then unigrams)
    tokens = name.split()
    normalised = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens):
            bigram = tokens[i] + " " + tokens[i + 1]
            if bigram in LEGAL_BIGRAMS:
                normalised.append(LEGAL_BIGRAMS[bigram])
                i += 2
                continue
        if tokens[i] in LEGAL_UNIGRAMS:
            normalised.append(LEGAL_UNIGRAMS[tokens[i]])
        else:
            normalised.append(tokens[i])
        i += 1
    name = " ".join(normalised)
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()
    return name


def normalize_address(address):
    """Normalise a business address for comparison."""
    if not isinstance(address, str) or not address.strip():
        return ""
    address = normalize_unicode(address)
    address = address.replace("&", " and ")
    # Remove punctuation except hyphens and slashes
    address = re.sub(r"[^\w\s/\-]", " ", address)
    # Normalise tokens
    tokens = address.split()
    normalised = []
    i = 0
    while i < len(tokens):
        lower = tokens[i].lower()
        # Check two-word state names
        if i + 1 < len(tokens):
            bigram = lower + " " + tokens[i + 1].lower()
            if bigram in US_STATES:
                normalised.append(US_STATES[bigram])
                i += 2
                continue
            if bigram in INDIAN_STATES:
                normalised.append(INDIAN_STATES[bigram])
                i += 2
                continue
        if lower in ADDRESS_ABBREVS:
            replacement = ADDRESS_ABBREVS[lower]
            if replacement:
                normalised.append(replacement)
        elif lower in US_STATES:
            normalised.append(US_STATES[lower])
        elif lower in INDIAN_STATES:
            normalised.append(INDIAN_STATES[lower])
        else:
            normalised.append(tokens[i])
        i += 1
    address = " ".join(normalised)
    address = re.sub(r"\s+", " ", address).strip()
    return address


def make_blocking_text(name_norm, addr_norm):
    """Create a combined text for TF-IDF blocking — name weighted 2x."""
    parts = []
    if name_norm:
        parts.append(name_norm)
        parts.append(name_norm)  # double-weight name
    if addr_norm:
        parts.append(addr_norm)
    return " ".join(parts)


def extract_name_tokens(name_norm):
    """Extract meaningful tokens from normalised name (remove stop/suffix words)."""
    if not name_norm:
        return set()
    stop = {"inc", "corp", "co", "ltd", "pvt", "llc", "llp", "plc",
            "the", "of", "and", "a", "an", "in", "for", "on", "at", "to",
            "ent", "assoc", "tech", "sol", "svc", "intl", "ind", "grp",
            "hldg", "fdn", "inst", "consult", "sa", "sas", "sarl", "eurl",
            "soc", "pvt ltd", "mfg", "mgmt", "const", "comm", "lab",
            "prop", "inv", "dev", "eng", "fin", "edu", "mkt", "log"}
    tokens = set(name_norm.split()) - stop
    return {t for t in tokens if len(t) > 1}


def extract_address_tokens(addr_norm):
    """Extract meaningful tokens from normalised address."""
    if not addr_norm:
        return set()
    stop = {"st", "rd", "ave", "blvd", "dr", "ln", "ct", "pl", "cir",
            "hwy", "pkwy", "ter", "trl", "way", "apt", "ste", "bldg",
            "fl", "unit", "rm", "no", "n", "s", "e", "w",
            "ne", "nw", "se", "sw", "po", "box", "blk", "phase",
            "plot", "sector", "dist", "teh", "tlk", "mdl", "vlg", "moh",
            "rue", "allee", "imp", "ch", "pass", "qtr", "arr", "cedex",
            "mt", "ft", "pt"}
    tokens = set(addr_norm.split()) - stop
    return {t for t in tokens if len(t) > 1}


def extract_numbers(text):
    """Extract all numeric substrings from text."""
    if not text:
        return set()
    return set(re.findall(r"\d+", text))


def get_name_ngrams(name_norm, n=3):
    """Generate character n-grams from normalised name (spaces removed)."""
    if not name_norm or len(name_norm) < n:
        return set()
    name_clean = name_norm.replace(" ", "")
    return {name_clean[i:i + n] for i in range(len(name_clean) - n + 1)}


def soundex(name):
    """Soundex phonetic encoding."""
    if not name or not isinstance(name, str):
        return ""
    name = re.sub(r"[^a-z]", "", name.lower())
    if not name:
        return ""
    coding = {
        "b": "1", "f": "1", "p": "1", "v": "1",
        "c": "2", "g": "2", "j": "2", "k": "2", "q": "2", "s": "2",
        "x": "2", "z": "2",
        "d": "3", "t": "3", "l": "4", "m": "5", "n": "5", "r": "6",
    }
    result = name[0].upper()
    prev = coding.get(name[0], "0")
    for c in name[1:]:
        code = coding.get(c, "0")
        if code != "0" and code != prev:
            result += code
        if code != "0":
            prev = code
        if len(result) == 4:
            break
    return result.ljust(4, "0")
