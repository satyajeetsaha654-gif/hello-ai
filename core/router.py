# core/router.py
"""
Hello AI - Intent Router

কাজ:
- User question-এর মূল intent নির্ধারণ করা
- Creator / time / weather / route / location / web / math / chat আলাদা করা
- Bengali, Hindi, English এবং Roman Bengali/Hindi support করা

গুরুত্বপূর্ণ:
এই ফাইল নিজে answer তৈরি করে না।
শুধু প্রশ্নটি কোন service-এর কাছে যাবে তা নির্ধারণ করে।
"""

from __future__ import annotations

import ast
import re
import unicodedata
from typing import Optional


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    User input-কে routing-এর জন্য normalize করে।

    Bengali/Hindi Unicode নষ্ট করে না।
    শুধু:
    - Unicode normalization
    - lowercase/casefold
    - punctuation cleanup
    - extra whitespace cleanup
    """
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.casefold().strip()

    # URL punctuation বা useful mathematical symbols বাদ না দিয়ে
    # সাধারণ punctuation-কে space করা হচ্ছে।
    text = re.sub(r"[,\u3001;:!?।,]+", " ", text)

    # Quotes/brackets etc.
    text = re.sub(r"""["'“”‘’`(){}\[\]]""", " ", text)

    # Multiple whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# BASIC HELPERS
# ============================================================

def _has_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _contains_word(text: str, words: tuple[str, ...]) -> bool:
    """
    English/Roman words-এর জন্য word boundary check।
    Bengali/Hindi words-এর জন্যও সাধারণ substring fallback কাজ করবে।
    """
    for word in words:
        word = word.strip().casefold()
        if not word:
            continue

        if re.search(r"^[a-z0-9_]+$", word):
            if re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE):
                return True
        elif word in text:
            return True

    return False


def _looks_like_arithmetic_expression(text: str) -> bool:
    """
    Pure mathematical expression কি না দেখে।

    যেমন:
        25 + 30
        100 / 4
        (20 + 5) * 3

    কিন্তু সাধারণ sentence-কে math হিসেবে ধরবে না।
    """
    value = text.strip()

    if not value:
        return False

    # খুব বড় input math expression হিসেবে নেব না।
    if len(value) > 200:
        return False

    # শুধু number/operator/bracket/space থাকলে candidate।
    if not re.fullmatch(r"[0-9+\-*/%().\s]+", value):
        return False

    # অন্তত একটি digit থাকতে হবে।
    if not re.search(r"\d", value):
        return False

    # শুধু punctuation/operator হলে reject।
    if not re.search(r"\d\s*[\+\-\*/%]", value) and not re.search(
        r"[\+\-\*/%]\s*\d", value
    ):
        return False

    try:
        tree = ast.parse(value, mode="eval")
    except Exception:
        return False

    allowed = (
        ast.Expression,
        ast.Constant,
        ast.UnaryOp,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
    )

    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            return False

        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                return False

    return True


# ============================================================
# CREATOR
# ============================================================

CREATOR_PATTERNS = (
    r"\bwho\s+(?:created|made|built|developed)\s+you\b",
    r"\bwho\s+is\s+your\s+creator\b",
    r"\bwho\s+created\s+hello\s*ai\b",
    r"\bwho\s+made\s+hello\s*ai\b",
    r"\bcreator\s+of\s+hello\s*ai\b",

    r"তোমাকে\s+কে\s+(?:তৈরি|বানিয়েছে|বানিয়েছে|তৈরি\s+করেছে)",
    r"তুমি\s+কে\s+বানিয়েছে",
    r"তোমার\s+নির্মাতা\s+কে",
    r"হ্যালো\s*এআই.*কে\s+তৈরি",
    r"হ্যালো\s*এআই.*কে\s+বানিয়েছে",
    r"হ্যালো\s*এআই.*কে\s+বানিয়েছে",

    r"\bke\s+(?:tomake|tomay)\s+(?:toiri|tairi|banieche|baniyeche|banalo)\b",
    r"\btomake\s+ke\s+toiri\s+koreche\b",
    r"\bhello\s*ai\s+ke\s+toiri\s+koreche\b",
    r"\bhello\s*ai\s+ke\s+baniyeche\b",

    r"तुम्हें\s+किसने\s+बनाया",
    r"आपको\s+किसने\s+बनाया",
    r"तुम्हारा\s+निर्माता\s+कौन\s+है",
)


def is_creator_question(text: str) -> bool:
    normalized = normalize_text(text)

    if not normalized:
        return False

    return _has_any(normalized, CREATOR_PATTERNS)


# ============================================================
# TIME / DATE
# ============================================================

TIME_PATTERNS = (
    r"\bwhat\s+time\s+is\s+it\b",
    r"\bwhat\s+time\b",
    r"\bcurrent\s+time\b",
    r"\btime\s+now\b",
    r"\btime\s+right\s+now\b",
    r"\bdate\s+today\b",
    r"\btoday'?s\s+date\b",
    r"\bwhat\s+day\s+is\s+today\b",
    r"\bwhich\s+day\s+is\s+today\b",

    r"কয়টা\s+বাজে",
    r"কয়টা\s+বাজে",
    r"কটা\s+বাজে",
    r"কত\s+বাজে",
    r"এখন\s+কয়টা",
    r"এখন\s+কয়টা",
    r"এখন\s+কটা",
    r"এখন\s+কত\s+বাজে",
    r"আজকের\s+তারিখ",
    r"আজ\s+কত\s+তারিখ",
    r"আজ\s+কি\s+বার",
    r"আজ\s+কোন\s+বার",

    r"\bkoyta\s+baje\b",
    r"\bkota\s+baje\b",
    r"\bkoita\s+baje\b",
    r"\bkotay\s+baje\b",
    r"\bekhon\s+koyta\b",
    r"\bekhon\s+kota\b",
    r"\bajker\s+tarikh\b",
    r"\baj\s+koto\s+tarikh\b",
    r"\baj\s+ki\s+bar\b",

    r"अभी\s+कितने\s+बजे",
    r"कितने\s+बजे\s+हैं",
    r"आज\s+की\s+तारीख",
    r"आज\s+कौन\s+सा\s+दिन",
)


def is_time_question(text: str) -> bool:
    normalized = normalize_text(text)

    if not normalized:
        return False

    return _has_any(normalized, TIME_PATTERNS)


# ============================================================
# WEATHER
# ============================================================

WEATHER_PATTERNS = (
    r"\bweather\b",
    r"\btemperature\b",
    r"\bforecast\b",
    r"\bclimate\s+today\b",
    r"\bhow\s+hot\b",
    r"\bhow\s+cold\b",
    r"\bwill\s+it\s+rain\b",
    r"\bis\s+it\s+raining\b",

    r"আবহাওয়া",
    r"আবহাওয়া",
    r"তাপমাত্রা",
    r"বৃষ্টি\s+হবে",
    r"বৃষ্টি\s+হচ্ছে",
    r"বৃষ্টি\s+পড়বে",
    r"বৃষ্টি\s+পড়বে",
    r"আজ\s+আবহাওয়া",
    r"আজ\s+আবহাওয়া",

    r"\babohawa\b",
    r"\babohawa\s+kemon\b",
    r"\btapmatra\b",
    r"\btemperature\s+koto\b",
    r"\bbristi\s+hobe\b",
    r"\bbristi\s+porbe\b",
    r"\baj\s+er\s+abohawa\b",

    r"मौसम",
    r"तापमान",
    r"बारिश",
    r"आज\s+का\s+मौसम",
)


def is_weather_question(text: str) -> bool:
    normalized = normalize_text(text)

    if not normalized:
        return False

    return _has_any(normalized, WEATHER_PATTERNS)


# ============================================================
# ROUTE / DIRECTIONS
# ============================================================

ROUTE_PATTERNS = (
    # English
    r"\bhow\s+(?:do|can)\s+i\s+get\s+from\b",
    r"\bhow\s+to\s+go\s+from\b",
    r"\bdirections?\s+from\b",
    r"\broute\s+from\b",
    r"\bway\s+from\b",
    r"\bget\s+from\b.+\bto\b",
    r"\btravel\s+from\b.+\bto\b",

    # Bengali
    r"থেকে.+কীভাবে\s+যাব",
    r"থেকে.+কিভাবে\s+যাব",
    r"থেকে.+কী\s+করে\s+যাব",
    r"থেকে.+কী\s+করে\s+যেতে",
    r"থেকে.+কিভাবে\s+যেতে",
    r"থেকে.+কীভাবে\s+যেতে",
    r"থেকে.+যাব",
    r"থেকে.+যেতে\s+হবে",
    r"কীভাবে\s+যাব",
    r"কিভাবে\s+যাব",
    r"কী\s+করে\s+যাব",

    # Roman Bengali
    r"\btheke\b.+\bki\s+kore\s+jabo\b",
    r"\btheke\b.+\bkivabe\s+jabo\b",
    r"\btheke\b.+\bkibhabe\s+jabo\b",
    r"\btheke\b.+\bki\s+kore\s+jete\b",
    r"\btheke\b.+\bkivabe\s+jete\b",
    r"\btheke\b.+\bkibhabe\s+jete\b",
    r"\btheke\b.+\bjabo\b",
    r"\btheke\b.+\bjete\s+hobe\b",
    r"\bhow\s+to\s+jabo\b",

    # Hindi
    r"से.+कैसे\s+जाएं",
    r"से.+कैसे\s+जायें",
    r"से.+कैसे\s+जाना",
    r"से.+कैसे\s+जाना\s+है",
    r"से.+कैसे\s+जाऊं",
    r"से.+कैसे\s+जाएँ",

    r"\bse\b.+\bkaise\s+jaye\b",
    r"\bse\b.+\bkaise\s+jayen\b",
    r"\bse\b.+\bkaise\s+jana\b",
    r"\bse\b.+\bkaise\s+jana\s+hai\b",
)


def is_route_question(text: str) -> bool:
    normalized = normalize_text(text)

    if not normalized:
        return False

    # Strong route indicators.
    if _has_any(normalized, ROUTE_PATTERNS):
        return True

    # Explicit "from X to Y" is almost always a route request
    # when accompanied by a movement word.
    if re.search(
        r"\bfrom\b.+\bto\b",
        normalized,
        re.IGNORECASE,
    ):
        if _contains_word(
            normalized,
            (
                "go",
                "get",
                "reach",
                "travel",
                "route",
                "way",
                "jabo",
                "jabo",
                "jete",
                "যাব",
                "যেতে",
                "পৌঁছ",
            ),
        ):
            return True

    # Bengali explicit X থেকে Y
    if " থেকে " in f" {normalized} ":
        if _contains_word(
            normalized,
            (
                "যাব",
                "যেতে",
                "পৌঁছ",
                "রুট",
                "রাস্তা",
                "কীভাবে",
                "কিভাবে",
                "jabo",
                "jete",
                "kivabe",
                "kibhabe",
                "route",
                "way",
            ),
        ):
            return True

    return False


# ============================================================
# LOCATION
# ============================================================

LOCATION_PATTERNS = (
    # English
    r"\bwhere\s+is\b",
    r"\bwhere\s+are\b",
    r"\bwhere\s+can\s+i\s+find\b",
    r"\blocation\s+of\b",
    r"\baddress\s+of\b",
    r"\baddress\b",
    r"\blocated\b",
    r"\bpin\s*code\b",
    r"\bpincode\b",
    r"\bpostal\s+code\b",
    r"\bpostcode\b",
    r"\bnear\s+me\b",
    r"\bnearby\b",
    r"\bnearest\b",

    # Bengali
    r"কোথায়",
    r"কোথায়",
    r"কোথায়\s+আছে",
    r"কোথায়\s+আছে",
    r"ঠিকানা",
    r"অবস্থান",
    r"পিন\s*কোড",
    r"পোস্টাল\s+কোড",
    r"কাছাকাছি",
    r"কোথায়\s+পাওয়া\s+যায়",
    r"কোথায়\s+পাওয়া\s+যায়",

    # Roman Bengali
    r"\bkothay\b",
    r"\bkothay\s+ache\b",
    r"\bthikana\b",
    r"\bobosthan\b",
    r"\bpin\s*code\b",
    r"\bpincode\b",
    r"\bpost\s*code\b",
    r"\bkache\b",
    r"\bkachakachi\b",

    # Hindi
    r"कहाँ",
    r"कहां",
    r"कहाँ\s+है",
    r"कहां\s+है",
    r"पता",
    r"स्थान",
    r"पिन\s*कोड",
    r"पोस्टल\s+कोड",
    r"पास\s+में",
)


def is_location_question(text: str) -> bool:
    normalized = normalize_text(text)

    if not normalized:
        return False

    # Route must win over ordinary location.
    if is_route_question(normalized):
        return True

    return _has_any(normalized, LOCATION_PATTERNS)


# ============================================================
# WEB / CURRENT INFORMATION
# ============================================================

WEB_PATTERNS = (
    # Current / latest
    r"\blatest\b",
    r"\bcurrent\b",
    r"\bright\s+now\b",
    r"\btoday\b",
    r"\btonight\b",
    r"\btomorrow\b",
    r"\byesterday\b",
    r"\brecent\b",
    r"\brecently\b",
    r"\blive\b",
    r"\bbreaking\b",
    r"\bupdated\b",
    r"\bupdate\b",
    r"\bnews\b",
    r"\bstatus\b",
    r"\bschedule\b",
    r"\bprice\b",
    r"\bcost\b",
    r"\brate\b",
    r"\bexchange\s+rate\b",
    r"\bresult\b",
    r"\bresults\b",

    # Bengali
    r"সর্বশেষ",
    r"সাম্প্রতিক",
    r"বর্তমান",
    r"এখনকার",
    r"আজকের",
    r"আজ",
    r"কালকের",
    r"আগামীকাল",
    r"গতকাল",
    r"সরাসরি",
    r"লাইভ",
    r"খবর",
    r"নিউজ",
    r"আপডেট",
    r"দাম",
    r"মূল্য",
    r"রেট",
    r"সময়সূচি",
    r"সময়সূচি",
    r"ফলাফল",

    # Roman Bengali
    r"\bsorbosesh\b",
    r"\bsamprotik\b",
    r"\bbortoman\b",
    r"\bajker\b",
    r"\bakhonkar\b",
    r"\bkalke\b",
    r"\bagamikal\b",
    r"\bnews\b",
    r"\bkhabar\b",
    r"\bupdate\b",
    r"\bdam\b",
    r"\bmullo\b",
    r"\brate\b",
    r"\bsomoysuchi\b",
    r"\bfolafol\b",

    # Hindi
    r"नवीनतम",
    r"ताज़ा",
    r"ताजा",
    r"वर्तमान",
    r"आज",
    r"अभी",
    r"कल",
    r"समाचार",
    r"खबर",
    r"अपडेट",
    r"कीमत",
    r"दाम",
    r"रेट",
    r"समय\s*सारणी",
    r"नतीजा",
    r"परिणाम",
)


def has_specific_year(text: str) -> bool:
    """
    1900–2099-এর explicit year আছে কি না।
    """
    return bool(re.search(r"\b(?:19|20)\d{2}\b", text))


def is_web_search_question(text: str) -> bool:
    normalized = normalize_text(text)

    if not normalized:
        return False

    # Explicit year থাকলে web verification দরকার হতে পারে।
    if has_specific_year(normalized):
        return True

    return _has_any(normalized, WEB_PATTERNS)


# ============================================================
# NATURAL LANGUAGE MATH
# ============================================================

NATURAL_MATH_PATTERNS = (
    r"\bcalculate\b",
    r"\bsolve\b",
    r"\bwhat\s+is\b.+\d",
    r"\bhow\s+much\s+is\b",
    r"\bplus\b",
    r"\bminus\b",
    r"\btimes\b",
    r"\bmultiplied\s+by\b",
    r"\bdivided\s+by\b",
    r"\bpercent\b",
    r"\bpercentage\b",

    r"হিসাব\s+কর",
    r"হিসাব\s+করে\s+দাও",
    r"গণনা\s+কর",
    r"যোগ\s+কর",
    r"বিয়োগ\s+কর",
    r"বিয়োগ\s+কর",
    r"গুণ\s+কর",
    r"ভাগ\s+কর",
    r"শতাংশ",
    r"কত\s+হবে",

    r"\bhisab\s+koro\b",
    r"\bhisab\s+kore\s+dao\b",
    r"\byog\s+koro\b",
    r"\bbiog\s+koro\b",
    r"\bgun\s+koro\b",
    r"\bhag\s+koro\b",
    r"\bshotangsho\b",

    r"गणना",
    r"हल\s+करो",
    r"जोड़",
    r"घटाना",
    r"गुणा",
    r"भाग",
    r"प्रतिशत",
    r"कितना\s+होगा",
)


def is_math_question(text: str) -> bool:
    normalized = normalize_text(text)

    if not normalized:
        return False

    if _looks_like_arithmetic_expression(normalized):
        return True

    return _has_any(normalized, NATURAL_MATH_PATTERNS)


# ============================================================
# CHAT
# ============================================================

def is_chat_question(text: str) -> bool:
    """
    যদি অন্য কোনো বিশেষ intent না মেলে,
    সাধারণ conversation হিসেবে ধরা হবে।
    """
    return bool(normalize_text(text))


# ============================================================
# MAIN INTENT DETECTOR
# ============================================================

def detect_intent(text: str) -> str:
    """
    Main router.

    Priority খুব গুরুত্বপূর্ণ:

        creator
        time
        weather
        route
        location
        web_search
        math
        chat

    Route আগে location-এর তুলনায় বেশি priority পাবে,
    যাতে "Kanchrapara থেকে Halisahar কীভাবে যাব?"
    শুধু location question হিসেবে ধরা না হয়।
    """

    normalized = normalize_text(text)

    if not normalized:
        return "chat"

    # --------------------------------------------------------
    # 1. CREATOR
    # --------------------------------------------------------
    if is_creator_question(normalized):
        return "creator"

    # --------------------------------------------------------
    # 2. TIME / DATE
    # --------------------------------------------------------
    if is_time_question(normalized):
        return "time"

    # --------------------------------------------------------
    # 3. WEATHER
    # --------------------------------------------------------
    if is_weather_question(normalized):
        return "weather"

    # --------------------------------------------------------
    # 4. ROUTE
    # --------------------------------------------------------
    if is_route_question(normalized):
        return "route"

    # --------------------------------------------------------
    # 5. LOCATION
    # --------------------------------------------------------
    if is_location_question(normalized):
        return "location"

    # --------------------------------------------------------
    # 6. WEB / CURRENT / YEAR-SPECIFIC
    # --------------------------------------------------------
    if is_web_search_question(normalized):
        return "web_search"

    # --------------------------------------------------------
    # 7. MATH
    # --------------------------------------------------------
    if is_math_question(normalized):
        return "math"

    # --------------------------------------------------------
    # 8. DEFAULT CHAT
    # --------------------------------------------------------
    return "chat"


# ============================================================
# OPTIONAL PUBLIC HELPERS
# ============================================================

def get_intent(text: str) -> str:
    """
    Backward-compatible alias.
    """
    return detect_intent(text)


__all__ = [
    "normalize_text",
    "detect_intent",
    "get_intent",
    "is_creator_question",
    "is_time_question",
    "is_weather_question",
    "is_route_question",
    "is_location_question",
    "is_web_search_question",
    "is_math_question",
    "has_specific_year",
]