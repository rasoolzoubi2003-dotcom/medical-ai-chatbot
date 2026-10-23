"""Fast text checks that run before retrieval: emergencies, prompt injection and greetings,
in English and Arabic, plus the fixed replies in both languages.

They are plain string and regex rules: cheap, predictable, and easy to test.
"""

import re

ARABIC_WORD = re.compile(r"[ء-ي]+")
LATIN_WORD = re.compile(r"[A-Za-z]+")
ARABIC_DIACRITICS = re.compile(r"[ً-ْٰـ]")  # short vowels, tanween, shadda, tatweel


def detect_language(text: str) -> str:
    """'ar' if the text has more Arabic words than Latin ones, else 'en'.

    Words, not letters, so an Arabic question with an English medical term
    ("ما هو diabetes؟") still counts as Arabic.
    """
    return "ar" if len(ARABIC_WORD.findall(text)) > len(LATIN_WORD.findall(text)) else "en"


def normalize(text: str) -> str:
    """Lowercase, and smooth over Arabic spelling variants so rules match how people type.

    Diacritics are removed and letter variants unified (أ/إ/آ -> ا, ى -> ي, ة -> ه),
    so "أريد أن أموت" and "اريد ان اموت" are treated the same.
    """
    return _fold_arabic(text.lower())


def _fold_arabic(text: str) -> str:
    text = ARABIC_DIACRITICS.sub("", text)
    return text.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"}))


def _patterns(*patterns: str) -> list[re.Pattern]:
    # Only fold Arabic letters: lowercasing would turn regex escapes like \S into \s.
    return [re.compile(_fold_arabic(p)) for p in patterns]


# Describe an emergency happening right now.
URGENT_PATTERNS = _patterns(
    r"can'?t breathe", r"cannot breathe", r"can not breathe", r"not breathing", r"stopped breathing",
    r"unconscious", r"won'?t wake up", r"suicidal", r"kill myself", r"end my life", r"want to die",
    r"took too many pills", r"overdosed", r"bleeding heavily", r"heavy bleeding", r"severe bleeding",
    r"won'?t stop bleeding",
    r"(لا|ما|مش)\s*(استطيع|يستطيع|تستطيع|بقدر|قادر|قادره)\s*(ان\s*)?(اتنفس|يتنفس|تتنفس|التنفس)",
    r"(لا|ما)\s*(يتنفس|تتنفس)", r"توقف\S* عن التنفس", r"فاقد\S* الوعي", r"مغمى علي", r"اغمي علي",
    r"(لا|ما)\s*(يستيقظ|تستيقظ|يصحى|تصحى)", r"انتحار", r"انتحر", r"اقتل نفسي", r"انهي حياتي",
    r"اريد ان اموت", r"بدي موت", r"جرعه زائده", r"حبوب\S*\s+كثير", r"نزيف شديد", r"(انزف|ينزف|تنزف) بشده",
    r"النزيف (لا|ما)\s*(يتوقف|بيوقف)",
)

# Conditions that are emergencies when someone has them now, but are also
# normal topics to ask about ("What are the warning signs of a stroke?").
EMERGENCY_CONDITIONS = _patterns(
    r"chest pain", r"stroke", r"seizure", r"heart attack", r"anaphylaxis", r"severe allergic reaction",
    r"poisoning", r"overdose", r"shortness of breath", r"difficulty breathing", r"face drooping",
    r"slurred speech", r"numb on one side", r"severe burn", r"collapsed",
    r"(الم|وجع)\s*(في|ب)\s*(ال)?صدر", r"جلطه", r"سكته", r"نوبه قلبيه", r"ازمه قلبيه", r"نوب\S* صرع",
    r"تشنج", r"حساسيه شديده", r"صدمه تحسسيه", r"تسمم", r"ضيق\s*(في\s*)?(ال)?تنفس",
    r"صعوبه\s*(في|ب)\s*(ال)?تنفس", r"سقط مغمى", r"انهار",
)

# Wording that suggests the condition is happening to someone right now.
HAPPENING_NOW = _patterns(
    r"\b(i'?m|i am|i think i|i feel|i have|i've|my \w+ (is|has|was)|(he|she|they)('s| is| are| has| have)"
    r"|someone (is|has)|is having|are having|having an?)\b",
    r"(^|\s)(انا|عندي|عنده|عندها|معي|معه|معها|اشعر|احس|حاسس|حاسه|اعاني|يعاني|تعاني|اصاب|اصبت|الان|هلق|هلا"
    r"|حاليا|ابي|امي|ابني|بنتي|ابنتي|زوجي|زوجتي|اخي|اختي|صديقي|صديقتي|طفلي|جدي|جدتي)(\s|$|[،,.؟?!])",
)

# Instruction-like wording used to override the assistant. Patterns need
# instruction context, so normal questions such as "How does insulin act as a
# hormone?" are not blocked.
INJECTION_PATTERNS = _patterns(
    r"\b(ignore|disregard|forget|skip|override|bypass)\b[^.?!]{0,40}\b(instructions?|rules|prompt|guidelines|above)\b",
    r"\b(reveal|show|print|repeat)\b[^.?!]{0,40}\b(instructions?|prompt|text above)\b",
    r"\bsystem prompt\b", r"\bnew instructions?\b", r"\byou are now\b", r"\bfrom now on,? you\b",
    r"\bpretend (to be|you are|you're)\b", r"\bjailbreak\b", r"\bdo anything now\b",
    r"(تجاهل|انس|اهمل|تخط|تجاوز)\S*[^.؟?!]{0,40}(تعليمات|قواعد|اوامر|توجيهات)",
    r"(اكشف|اظهر|اعرض|اطبع|كرر)\S*[^.؟?!]{0,40}(تعليمات|البرومبت|موجه)",
    r"تعليمات النظام", r"موجه النظام", r"تعليمات جديده", r"(^|\s)انت الان(\s|$)",
    r"من الان (فصاعدا|وصاعدا|ورايح)", r"(^|\s)تظاهر",
)

GREETINGS = tuple(normalize(g) for g in (
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "مرحبا", "اهلا", "أهلين", "هلا", "السلام عليكم", "سلام", "صباح الخير", "مساء الخير", "هاي",
))
THANKS = tuple(normalize(t) for t in (
    "thanks", "thank you", "شكرا", "مشكور", "مشكورة", "يعطيك العافية", "تسلم",
))
ABOUT_ME = tuple(normalize(a) for a in (
    "who are you", "what are you", "what can you do", "what can you help me with", "what do you do",
    "من أنت", "مين أنت", "ما أنت", "ماذا تستطيع أن تفعل", "ماذا يمكنك أن تفعل", "بماذا يمكنك مساعدتي",
    "كيف يمكنك مساعدتي", "شو بتقدر تعمل", "شو بتعمل",
))

# Fixed replies, in the language of the question.
REPLIES = {
    "en": {
        "emergency": (
            "This may be a medical emergency. Please call your local emergency number "
            "(e.g. 911 or 199) or go to the nearest emergency room right away. "
            "I'm not able to provide emergency care through this chat."
        ),
        "emergency_note": (
            "\n\nIf you or someone near you has these symptoms right now, "
            "call your local emergency number immediately."
        ),
        "disclaimer": (
            "\n\nThis is general information, not a medical diagnosis. "
            "Please confirm with a licensed doctor before acting on it."
        ),
        "off_topic": (
            "I apologize, but I am specialized only in medical and health-related questions. "
            "Please ask a health-related question."
        ),
        "greeting": (
            "Hello! I'm a medical information assistant. I can answer questions about diseases, "
            "symptoms, treatments and prevention, using information from NIH sources. "
            "What would you like to know?"
        ),
        "thanks": "You're welcome! Let me know if you have another health question.",
        "knowledge_base_not_ready": "The knowledge base is still being built. Please try again soon.",
        "llm_unavailable": "The AI model is not available right now. Please try again in a moment.",
    },
    "ar": {
        "emergency": (
            "قد تكون هذه حالة طبية طارئة. يرجى الاتصال فوراً برقم الطوارئ في بلدك "
            "(مثل 911 أو 199) أو التوجه إلى أقرب قسم طوارئ. "
            "لا أستطيع تقديم رعاية طارئة عبر هذه المحادثة."
        ),
        "emergency_note": (
            "\n\nإذا كانت هذه الأعراض تحدث لك أو لشخص قريب منك الآن، "
            "فاتصل برقم الطوارئ فوراً."
        ),
        "disclaimer": (
            "\n\nهذه معلومات عامة وليست تشخيصاً طبياً. "
            "يرجى مراجعة طبيب مختص قبل اتخاذ أي قرار بناءً عليها."
        ),
        "off_topic": (
            "عذراً، أنا متخصص فقط في الأسئلة الطبية والصحية. "
            "يرجى طرح سؤال يتعلق بالصحة."
        ),
        "greeting": (
            "مرحباً! أنا مساعد للمعلومات الطبية. يمكنني الإجابة عن أسئلة حول الأمراض والأعراض "
            "والعلاج والوقاية، اعتماداً على معلومات من مصادر المعاهد الوطنية للصحة الأمريكية (NIH). "
            "ماذا تريد أن تعرف؟"
        ),
        "thanks": "على الرحب والسعة! إذا كان لديك سؤال صحي آخر فأنا هنا.",
        "knowledge_base_not_ready": "قاعدة المعرفة ما زالت قيد البناء. يرجى المحاولة بعد قليل.",
        "llm_unavailable": "نموذج الذكاء الاصطناعي غير متاح حالياً. يرجى المحاولة بعد قليل.",
    },
}


def reply(name: str, language: str) -> str:
    """A fixed reply ('emergency', 'off_topic', ...) in the given language ('en' or 'ar')."""
    return REPLIES.get(language, REPLIES["en"])[name]


def _matches_any(text: str, patterns: list[re.Pattern]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def mentions_emergency_condition(text: str) -> bool:
    """True if the text names a condition that can be an emergency (e.g. stroke)."""
    return _matches_any(normalize(text), EMERGENCY_CONDITIONS)


def is_emergency(text: str) -> bool:
    """True if the message describes an emergency happening now."""
    text = normalize(text)
    if _matches_any(text, URGENT_PATTERNS):
        return True
    return _matches_any(text, EMERGENCY_CONDITIONS) and _matches_any(text, HAPPENING_NOW)


def contains_injection_attempt(text: str) -> bool:
    """True if the message tries to override or reveal the assistant's instructions."""
    return _matches_any(normalize(text), INJECTION_PATTERNS)


def _plain_words(text: str) -> str:
    """Normalized words without punctuation: 'Hello there!' -> 'hello there'."""
    return " ".join(re.sub(r"[^\w' ]", " ", normalize(text)).split())


def greeting_reply(text: str) -> str | None:
    """Fixed reply for a short greeting, thanks, or 'who are you' message, else None."""
    words = _plain_words(text)
    language = detect_language(text)
    if words in ABOUT_ME:
        return reply("greeting", language)
    if len(words.split()) > 4:
        return None
    if any(words == t or words.startswith(t + " ") for t in THANKS):
        return reply("thanks", language)
    if any(words == g or words.startswith(g + " ") for g in GREETINGS):
        return reply("greeting", language)
    return None


def precheck(question: str) -> str | None:
    """Run the text checks in priority order: 'emergency', 'injection', 'greeting', or None."""
    if is_emergency(question):
        return "emergency"
    if contains_injection_attempt(question):
        return "injection"
    if greeting_reply(question):
        return "greeting"
    return None
