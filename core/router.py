"""Intent routing for Hello AI."""

import re
import unicodedata


def normalize_text(question: str) -> str:
    """Normalize text while preserving Bengali, Hindi and other Unicode text."""
    if not isinstance(question, str):
        return ""

    text = unicodedata.normalize("NFKC", question).casefold().strip()
    text = re.sub(r"[?!.,;:।,]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _matches_any(text: str, patterns: list[str]) -> bool:
    """Check whether any intent pattern matches."""
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def detect_intent(question: str) -> str:
    """Classify a question before selecting a service."""
    text = normalize_text(question)

    if not text:
        return "empty"

    # CREATOR
    creator_patterns = [
        r"\bwho created you\b",
        r"\bwho made you\b",
        r"\bwho is your creator\b",
        r"\bwho developed you\b",
        r"\bwho built you\b",
        r"\bwho is your developer\b",
        r"তোমাকে কে তৈরি করেছে",
        r"তোমাকে কে বানিয়েছে",
        r"কে তোমাকে বানিয়েছে",
        r"তোমার নির্মাতা কে",
        r"তোমার স্রষ্টা কে",
        r"आपको किसने बनाया",
        r"तुम्हें किसने बनाया",
        r"आपका निर्माता कौन है",
        r"तुम्हारा निर्माता कौन है",
    ]

    if _matches_any(text, creator_patterns):
        return "creator"

    # TIME AND DATE
    time_patterns = [
        r"\btoday date\b",
        r"\bwhat(?:'s| is)? the time\b",
        r"\bwhat time is it\b",
        r"\b(?:current|local) time\b",
        r"\btime now\b",
        r"\btell me the time\b",
        r"\btime in [a-z][a-z .'-]*",
        r"\bwhat(?:'s| is)? (?:today'?s? )?date\b",
        r"\bdate today\b",
        r"\btoday'?s date\b",
        r"\bcurrent date\b",
        r"\bwhat day is (?:it|today)\b",
        r"\bwhat date is it\b",
        r"\bkota baje\b",
        r"\bkoyta baje\b",
        r"\bkoita baje\b",
        r"\bakhon kota baje\b",
        r"\bakhon koyta baje\b",
        r"\bakhon koto baje\b",
        r"\btime koto\b",
        r"\bakhon time koto\b",
        r"\bsomoy koto\b",
        r"\bekhon somoy koto\b",
        r"\baj koto tarik\b",
        r"\bajker tarikh\b",
        r"\baj ki tarikh\b",
        r"\baj ki bar\b",
        r"\babhi kitne baje\b",
        r"\bsamay kya hai\b",
        r"\bkitne baje hain\b",
        r"\baaj ki tareekh\b",
        r"\baaj kya tareekh hai\b",
        r"কটা বাজে|কয়টা বাজে|কয়টা বাজে|কোটা বাজে",
        r"এখন সময় কত|এখন সময় কত|এখন কটা|এখন কয়টা বাজে",
        r"আজ কত তারিখ|আজকের তারিখ|আজ কী বার|আজ কি বার",
        r"বর্তমান সময়|বর্তমান সময়|এখন কয়টা বাজে",
        r"अभी कितने बजे|अभी समय क्या है|समय क्या है",
        r"कितने बजे हैं|आज की तारीख|आज कितना तारीख",
        r"अभी टाइम क्या है",
    ]

    if _matches_any(text, time_patterns):
        return "time"

    # WEATHER
    weather_patterns = [
        r"\brain today\b",
        r"\bweather\b",
        r"\btemperature\b",
        r"\bforecast\b",
        r"\bwill it rain\b",
        r"\bweather today\b",
        r"\bmausam\b",
        r"\bbarish\b",
        r"\btemperature koto\b",
        r"\bbristi\b",
        r"\bbrishti\b",
        r"\baj bristi hobe\b",
        r"বৃষ্টি|আবহাওয়া|আবহাওয়া|তাপমাত্রা|মেঘ",
        r"আজ বৃষ্টি হবে|বৃষ্টি হবে কি",
        r"मौसम|बारिश|तापमान|मौसम कैसा",
    ]

    if _matches_any(text, weather_patterns):
        return "weather"

    # ROUTES AND DIRECTIONS
    route_patterns = [
        r"\bhow do i get to\b",
r"স্টেশনে যাওয়ার রাস্তা|স্টেশনে যাওয়ার রাস্তা",
r"एयरपोर्ट कैसे पहुंचें|एयरपोर्ट कैसे पहुँचें",
        r"\bdirections?\b",
        r"\broute\b",
        r"\bnavigate\b",
        r"\bhow to reach\b",
        r"\bdistance between\b",
        r"\bhow far\b",
        r"\brasta batao\b",
        r"\bkaise jaun\b",
        r"\bkitna dur\b",
        r"\bkivabe jabo\b",
        r"\bkibhabe jabo\b",
        r"পথ দেখাও|রাস্তা দেখাও|কীভাবে যাব|কিভাবে যাব",
        r"কত দূর|রাস্তা কোথায়|রুট দেখাও",
        r"रास्ता बताओ|कैसे पहुँचें|कितनी दूर",
    ]

    if _matches_any(text, route_patterns):
        return "route"

    # EXPLICIT WEB SEARCH / NEWS REQUESTS
    web_patterns = [
        r"\bwho won the latest election\b",
        r"वेब पर खोजो|वेब पर खोजें",
        r"\bsearch the web\b",
        r"\bsearch online\b",
        r"\blook up\b",
        r"\blatest news\b",
        r"\brecent news\b",
        r"\bcurrent news\b",
        r"\bnews today\b",
        r"\bsource links\b",
        r"\bsearch for\b",
        r"\bfind online\b",
        r"\baaj ki khabar\b",
        r"ওয়েবে খোঁজ|ওয়েবে খোঁজ|সর্বশেষ খবর",
        r"আজকের খবর|সাম্প্রতিক খবর|তাজা খবর",
        r"आज की खबर|ताज़ा खबर|ताजा खबर|हाल की खबर",
    ]

    if _matches_any(text, web_patterns):
        return "web_search"

    # ARITHMETIC EXPRESSIONS
    arithmetic = text.replace(",", "")

    if re.fullmatch(r"[\d\s()+\-*/%.]+", arithmetic):
        if re.search(r"\d", arithmetic) and re.search(r"[+\-*/%]", arithmetic):
            return "math"

    # NATURAL-LANGUAGE MATH
    math_patterns = [
        r"\bcalculate\b",
        r"\bcompute\b",
        r"\bsolve\b",
        r"\bwhat is \d",
        r"\bhow much is \d",
        r"\bcalculate \d",
        r"\bsolve \d",
        r"হিসাব কর|গণনা কর|সমাধান কর|যোগ কর|বিয়োগ কর|বিয়োগ কর",
        r"গুণ কর|ভাগ কর",
        r"गणना करो|हल करो|जोड़ो|घटाओ|गुणा करो|भाग करो",
    ]

    if _matches_any(text, math_patterns):
        return "math"

    return "chat"