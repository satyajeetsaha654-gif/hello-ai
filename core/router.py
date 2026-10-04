"""Intent routing for Hello AI.

Classifies user questions so the application can choose an appropriate
service. Classification does not replace web verification or AI reasoning.
"""

import re
import unicodedata


def normalize_text(question: str) -> str:
    """Normalize text while preserving Unicode scripts."""
    if not isinstance(question, str):
        return ""

    text = unicodedata.normalize("NFKC", question).casefold().strip()
    text = re.sub(r"[?!.,;:।,]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def detect_intent(question: str) -> str:
    """Classify a question before the application selects a service."""
    text = normalize_text(question)

    if not text:
        return "empty"

    # 1. CREATOR — keep the app's existing deterministic creator response.
    creator_patterns = [
        r"\bwho (?:created|made|built|developed) you\b",
        r"\bwho is your (?:creator|developer)\b",
        r"তোমাকে কে (?:তৈরি করেছে|বানিয়েছে|বানিয়েছে)",
        r"কে তোমাকে বানিয়েছে|কে তোমাকে বানিয়েছে",
        r"তোমার (?:নির্মাতা|স্রষ্টা) কে",
        r"आपको किसने बनाया|तुम्हें किसने बनाया",
        r"आपका निर्माता कौन है|तुम्हारा निर्माता कौन है",
    ]
    if _matches_any(text, creator_patterns):
        return "creator"

    # 2. TIME AND DATE
    time_patterns = [
        r"\bwhat(?:'s| is)? the time\b",
        r"\bwhat time is it\b",
        r"\b(?:current|local) time\b",
        r"\btime now\b|\btell me the time\b",
        r"\btime in [a-z][a-z .'-]*",
        r"\bwhat(?:'s| is)? (?:today'?s? )?date\b",
        r"\bdate today\b|\btoday'?s date\b|\bcurrent date\b",
        r"\bwhat day is (?:it|today)\b|\bwhat date is it\b",
        r"\b(?:akhon|ekhon) (?:kota|koyta|koto) baje\b",
        r"\b(?:kota|koyta|koita) baje\b|\bsomoy koto\b|\btime koto\b",
        r"\baj(?:ker)? (?:ki )?tarikh\b|\baj ki bar\b",
        r"\babhi kitne baje\b|\bsamay kya hai\b|\bkitne baje hain\b",
        r"\baaj ki tareekh\b|\baaj kya tareekh hai\b",
        r"কটা বাজে|কয়টা বাজে|কয়টা বাজে|কোটা বাজে",
        r"এখন সময় কত|এখন সময় কত|এখন কটা|এখন কয়টা বাজে",
        r"আজ কত তারিখ|আজকের তারিখ|আজ কী বার|আজ কি বার",
        r"বর্তমান সময়|বর্তমান সময়",
        r"अभी कितने बजे|अभी समय क्या है|समय क्या है|आज की तारीख",
    ]
    if _matches_any(text, time_patterns):
        return "time"

    # 3. WEATHER — prioritize this before general web/location questions.
    weather_patterns = [
        r"\bweather\b|\btemperature\b|\bforecast\b",
        r"\brain today\b|\bwill it rain\b",
        r"\bmausam\b|\bbarish\b|\bbaarish\b|\bbrishti\b|\bbristi\b",
        r"\b(?:aaj|aj|kal) (?:ka )?mausam\b",
        r"\btemperature koto\b|\bbristi hobe\b|\bbristi porbe\b",
        r"আবহাওয়া|আবহাওয়া|তাপমাত্রা|বৃষ্টি|বৃষ্টিপাত|মেঘ",
        r"আজ বৃষ্টি হবে|বৃষ্টি হবে কি|আবহাওয়া কেমন|আবহাওয়া কেমন",
        r"मौसम|बारिश|तापमान|मौसम कैसा|बारिश होगी",
    ]
    if _matches_any(text, weather_patterns):
        return "weather"

    # 4. ROUTES, DIRECTIONS AND TRAVEL
    route_patterns = [
        r"\bhow do i (?:get|travel|go) from\b",
        r"\bhow can i (?:get|travel|go) from\b",
        r"\bhow to (?:get|travel|go) from\b",
        r"\bhow do i get to\b|\bhow to reach\b",
        r"\bbest way to (?:get|travel|go) from\b",
        r"\broute from\b|\bdirections? to\b|\bdirections? from\b",
        r"\bnavigate to\b|\bdistance between\b|\bdistance from\b",
        r"\bhow far is\b|\btravel from .+ to\b|\bgo from .+ to\b",
        r"\b(?:rasta|raasta) batao\b|\bkaise jaun\b|\bkaise jaaun\b",
        r"\bkitna dur\b|\bkitni door\b|\bkivabe jabo\b|\bkibhabe jabo\b",
        r"\b(?:kolkata|city|station|airport) theke .+ (?:jabo|jawar|jaowar)\b",
        r"\b.+ se .+ (?:kaise jaye|kaise jaaye|kaise pahunche|kaise pahunchu)\b",
        r"\b.+ se .+ jane ka rasta\b|\b.+ tak kaise pahunche\b",
        r"পথ দেখাও|রাস্তা দেখাও|কীভাবে যাব|কিভাবে যাব|কীভাবে যাবো|কিভাবে যাবো",
        r"কত দূর|রাস্তা কোথায়|রাস্তা কোথায়|রুট দেখাও",
        r"কোথা দিয়ে যাব|কোন রাস্তা দিয়ে|যাওয়ার উপায়|যাওয়ার উপায়",
        r"স্টেশনে যাওয়ার রাস্তা|স্টেশনে যাওয়ার রাস্তা",
        r"কলকাতা থেকে .+ কীভাবে যাব|.+ থেকে .+ কীভাবে যাব",
        r"रास्ता बताओ|कैसे पहुँचें|कैसे पहुंचें|कितनी दूर",
        r"कैसे जाएं|कैसे जायें|जाने का रास्ता|रास्ता कौन सा है",
        r"कहाँ से जाएं|कहां से जाएं|.+ से .+ कैसे जाएं",
    ]
    if _matches_any(text, route_patterns):
        return "route"

    # 5. PLACE, ADDRESS AND POSTAL LOOKUPS
    location_patterns = [
        r"\bwhere is\b|\bwhere (?:is it )?located\b",
        r"\blocation of\b|\baddress of\b|\bfull address\b",
        r"\bpincode\b|\bpin code\b|\bpostal code\b|\bzip code\b",
        r"\bpost office\b|\bpostcode\b|\bpostal address\b",
        r"\bwhich district\b|\bwhich state\b|\bwhich country\b",
        r"কোথায় অবস্থিত|কোথায় অবস্থিত|কোথায় আছে|কোথায় আছে",
        r"পিনকোড|পিন কোড|ডাক কোড|ডাকঘর|পোস্ট অফিস|ঠিকানা",
        r"কোথায় পাব|কোথায় পাব|কোন এলাকায়|কোন এলাকায়",
        r"পোস্টাল কোড|কোন জেলায়|কোন জেলায়",
        r"पिनकोड|पिन कोड|डाक कोड|डाकघर|पोस्ट ऑफिस|पता",
        r"कहाँ है|कहां है|कहाँ स्थित है|कहां स्थित है",
        r"किस जिले में|किस राज्य में|किस देश में",
    ]
    if _matches_any(text, location_patterns):
        return "web_search"

    # 6. CURRENT / CHANGING INFORMATION
    web_patterns = [
        r"\blatest\b|\brecent\b|\bcurrent news\b|\bnews today\b",
        r"\bwhat happened today\b|\bwhat is happening\b",
        r"\btoday's news\b|\btodays news\b",
        r"\bwho won\b|\bmatch result\b|\blive score\b",
        r"\blatest (?:cricket|football|soccer|tennis|basketball|sports)\b",
        r"\b(?:cricket|football|soccer|tennis|basketball) (?:match|score|result) (?:today|yesterday|latest)\b",
        r"\bwho is the current\b|\bcurrent president\b|\bcurrent prime minister\b",
        r"\bprice today\b|\bexchange rate\b|\bstock price\b",
        r"\bsearch the web\b|\bsearch online\b|\blook up\b|\bfind online\b",
        r"\bsearch for\b|\bsource links\b",
        r"\bwhen is .+ (?:festival|puja|election|exam)\b",
        r"\bwhen will .+ (?:happen|start|end)\b",
        r"\b(?:aaj|aj) ki khabar\b|\btaza khabar\b|\btaaza khabar\b",
        r"\bcricket ka result\b|\bmatch kisne jeeta\b|\baaj ka match\b",
        r"ওয়েবে খোঁজ|ওয়েবে খোঁজ|সর্বশেষ খবর|আজকের খবর|সাম্প্রতিক খবর|তাজা খবর",
        r"আজকের ম্যাচ|কে জিতেছে|সর্বশেষ ফলাফল|বর্তমান দাম|আজকের দাম",
        r"দুর্গাপুজো কবে|দুর্গাপূজা কবে|পুজো কবে|পূজা কবে",
        r"আজকের ক্রিকেট|ক্রিকেট ম্যাচের ফল|আজকের খেলার ফল",
        r"आज की खबर|ताज़ा खबर|ताजा खबर|हाल की खबर|आज का मैच",
        r"कौन जीता|मैच का नतीजा|आज का क्रिकेट|दुर्गा पूजा कब है",
        r"इस साल .+ कब है|वर्तमान कीमत|आज का भाव",
    ]
    if _matches_any(text, web_patterns):
        return "web_search"

    # 7. ARITHMETIC EXPRESSIONS
    arithmetic = text.replace(",", "")
    if re.fullmatch(r"[\d\s()+\-*/%.]+", arithmetic):
        if re.search(r"\d", arithmetic) and re.search(r"[+\-*/%]", arithmetic):
            return "math"

    # 8. NATURAL-LANGUAGE MATH
    math_patterns = [
        r"\bcalculate\b|\bcompute\b|\bsolve\b",
        r"\bwhat is \d|\bhow much is \d",
        r"\bpercentage of\b|\bsquare root of\b",
        r"হিসাব কর|গণনা কর|সমাধান কর|যোগ কর|বিয়োগ কর|বিয়োগ কর",
        r"গুণ কর|ভাগ কর|শতকরা কত|বর্গমূল",
        r"गणना करो|हल करो|जोड़ो|घटाओ|गुणा करो|भाग करो",
        r"प्रतिशत कितना|वर्गमूल",
    ]
    if _matches_any(text, math_patterns):
        return "math"

    # 9. GENERAL KNOWLEDGE, REASONING, CODING AND FOLLOW-UPS
    # These intentionally go to the AI/chat service unless a specialized
    # intent above is confidently detected.
    return "chat"