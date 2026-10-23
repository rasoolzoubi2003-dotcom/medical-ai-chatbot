"""Fast text checks that run before retrieval: emergencies, prompt injection and greetings.

They are plain string and regex rules: cheap, predictable, and easy to test.
"""

import re

# Phrases that describe an emergency happening right now.
URGENT_PHRASES = [
    "can't breathe", "cannot breathe", "can not breathe", "not breathing", "stopped breathing",
    "unconscious", "won't wake up", "suicidal", "kill myself", "end my life",
    "want to die", "took too many pills", "overdosed", "bleeding heavily", "heavy bleeding",
    "severe bleeding", "won't stop bleeding",
]

# Conditions that are emergencies when someone has them now, but are also
# normal topics to ask about ("What are the warning signs of a stroke?").
EMERGENCY_CONDITIONS = [
    "chest pain", "stroke", "seizure", "heart attack", "anaphylaxis", "severe allergic reaction",
    "poisoning", "overdose", "shortness of breath", "difficulty breathing", "face drooping",
    "slurred speech", "numb on one side", "severe burn", "collapsed",
]

# Wording that suggests the condition is happening to someone right now.
HAPPENING_NOW = re.compile(
    r"\b(i'?m|i am|i think i|i feel|i have|i've|my \w+ (is|has|was)|(he|she|they)('s| is| are| has| have)"
    r"|someone (is|has)|is having|are having|having an?)\b"
)

# Instruction-like wording used to override the assistant. Patterns need
# instruction context, so normal questions such as "How does insulin act as a
# hormone?" are not blocked.
INJECTION_PATTERNS = [
    re.compile(pattern) for pattern in [
        r"\b(ignore|disregard|forget|skip|override|bypass)\b[^.?!]{0,40}\b(instructions?|rules|prompt|guidelines|above)\b",
        r"\b(reveal|show|print|repeat)\b[^.?!]{0,40}\b(instructions?|prompt|text above)\b",
        r"\bsystem prompt\b",
        r"\bnew instructions?\b",
        r"\byou are now\b",
        r"\bfrom now on,? you\b",
        r"\bpretend (to be|you are|you're)\b",
        r"\bjailbreak\b",
        r"\bdo anything now\b",
    ]
]

GREETINGS = ("hi", "hello", "hey", "good morning", "good afternoon", "good evening")
THANKS = ("thanks", "thank you")
ABOUT_ME = ("who are you", "what are you", "what can you do", "what can you help me with", "what do you do")

EMERGENCY_RESPONSE = (
    "This may be a medical emergency. Please call your local emergency number "
    "(e.g. 911 or 199) or go to the nearest emergency room right away. "
    "I'm not able to provide emergency care through this chat."
)

EMERGENCY_NOTE = (
    "\n\nIf you or someone near you has these symptoms right now, "
    "call your local emergency number immediately."
)

MEDICAL_DISCLAIMER = (
    "\n\nThis is general information, not a medical diagnosis. "
    "Please confirm with a licensed doctor before acting on it."
)

OFF_TOPIC_RESPONSE = (
    "I apologize, but I am specialized only in medical and health-related questions. "
    "Please ask a health-related question."
)

GREETING_RESPONSE = (
    "Hello! I'm a medical information assistant. I can answer questions about diseases, "
    "symptoms, treatments and prevention, using information from NIH sources. "
    "What would you like to know?"
)

THANKS_RESPONSE = "You're welcome! Let me know if you have another health question."


def mentions_emergency_condition(text: str) -> bool:
    """True if the text names a condition that can be an emergency (e.g. stroke)."""
    lowered = text.lower()
    return any(condition in lowered for condition in EMERGENCY_CONDITIONS)


def is_emergency(text: str) -> bool:
    """True if the message describes an emergency happening now."""
    lowered = text.lower()
    if any(phrase in lowered for phrase in URGENT_PHRASES):
        return True
    return mentions_emergency_condition(lowered) and bool(HAPPENING_NOW.search(lowered))


def contains_injection_attempt(text: str) -> bool:
    """True if the message tries to override or reveal the assistant's instructions."""
    lowered = text.lower()
    return any(pattern.search(lowered) for pattern in INJECTION_PATTERNS)


def _plain_words(text: str) -> str:
    """Lowercase and drop punctuation: 'Hello there!' -> 'hello there'."""
    return " ".join(re.sub(r"[^a-z' ]", " ", text.lower()).split())


def greeting_reply(text: str) -> str | None:
    """Canned reply for a short greeting, thanks, or 'who are you' message, else None."""
    words = _plain_words(text)
    if words in ABOUT_ME:
        return GREETING_RESPONSE
    if len(words.split()) > 4:
        return None
    if any(words == g or words.startswith(g + " ") for g in THANKS):
        return THANKS_RESPONSE
    if any(words == g or words.startswith(g + " ") for g in GREETINGS):
        return GREETING_RESPONSE
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
