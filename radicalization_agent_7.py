import os
import re
import pandas as pd
import nltk
import uuid
import json
from datetime import datetime, timezone
from dotenv import load_dotenv
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from typing import Optional
import random 
import unicodedata

# Global caches
_code_check_cache = {}
_llm_response_cache = {}
_relevance_cache = {}
_violence_cache = {}

# Load .env and NLTK data
def _ensure_nltk_data():
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon")

API_KEY = os.getenv("CHAT_AI_ACCESS_KEY")

REJECTION_PATTERNS = {
    "en": ["i cannot", "i'm not allowed", "i'm unable", "this violates", "please contact emergency services"],
    "de": ["ich kann nicht", "ich darf nicht", "ich bin nicht in der lage", "das verstößt", "bitte wende dich an die notdienste"]
}

# Enhanced comprehensive radicalization indicators from documents
RADICALIZATION_KNOWLEDGE_BASE = {
    "ideological_markers": [
        "great replacement", "großer austausch", "volkstod", "umvolkung",
        "deep state", "tiefer staat", "new world order", "neue weltordnung",
        "jewish world conspiracy", "jüdische weltverschwörung", "zog",
        "nothing happens by chance", "nichts passiert zufällig",
        "everything is connected", "alles hängt zusammen",
        "elites are lying", "eliten lügen", "globalists", "globalisten",
        "illuminati", "freimaurer", "freemasons", "rothschild",
        "adrenochrome", "enlightened minority", "erwachte minderheit",
        "hidden plan", "versteckter plan", "evil elite", "böse elite"
    ],
    "anti_state_markers": [
        "brd gmbh", "die brd ist kein staat", "bundesrepublik nicht legitim",
        "reich exists", "reich existiert noch", "reichsbürger", "selbstverwalter",
        "germany in borders of 1937", "deutschland in grenzen von 1937",
        "traitors to the people", "volksverräter", "system parties", "systemparteien",
        "the system must fall", "das system muss fallen", "altparteien",
        "no taxes for system", "keine steuern für system", "i am self-governor",
        "lügenpresse", "lying press", "mainstream media is fake"
    ],
    "conspiracy_markers": [
        "chemtrails", "5g causes illness", "5g macht krank", "haarp",
        "rfid", "barcode radiation", "barcodestrahlung", "vaccine chip", "impfchip",
        "death chip vaccinations", "todeschimp impfungen", "mind control",
        "aluminum hat", "aluhut", "big pharma enemy", "they control everything",
        "wake up sheeple", "wacht auf schafe", "global control through fear"
    ],
    "xenophobic_markers": [
        "islamization", "islamisierung", "eurabia", "arabization of europe",
        "population exchange", "bevölkerungsaustausch", "they use migrants",
        "we are being replaced", "wir werden ersetzt", "defend our people",
        "unser volk verteidigen", "protect fatherland", "vaterland schützen",
        "pure national community", "reine volksgemeinschaft", "volkskörper",
        "national body is sick", "volkskörper ist krank", "foreigners out", "ausländer raus"
    ],
    "hate_speech_markers": [
        "white lives matter", "weiße leben zählen", "deportation now", "abschiebung jetzt",
        "germany for germans", "deutschland den deutschen", "we are the people", "wir sind das volk",
        "my people first", "mein volk zuerst", "end the system", "system beenden",
        "for the fatherland", "für das vaterland"
    ],
    "symbolic_codes": [
        "88", "18", "28", "444", "666", "23", "swastika", "hakenkreuz",
        "black sun", "schwarze sonne", "celtic cross", "keltenkreuz", 
        "ss runes", "⚡⚡", "reichsflagge", "imperial war flag", "reichskriegsflagge",
        "eye of providence", "drudenstern", "pepe", "clown", "monkey",
        "blue heart", "blaues herz", "döp dodö dop", "arardddar"
    ],
    "digital_radicalization_markers": [
        "telegram truth", "telegram wahrheit", "real truth on", "echte wahrheit auf",
        "4chan", "rumble", "do my own research", "eigene recherche",
        "censored everywhere", "überall zensiert", "watch before deleted",
        "anschauen bevor gelöscht", "they don't want you to see",
        "querdenker", "redpiller", "blackpiller", "incel"
    ],
    "victimization_markers": [
        "they are out to get us", "sie haben es auf uns abgesehen",
        "we are being erased", "wir werden ausgelöscht", "silenced", "zum schweigen gebracht",
        "traditional values under attack", "traditionelle werte unter angriff",
        "you can't say anything anymore", "man kann nichts mehr sagen",
        "this country going downhill", "dieses land geht bergab",
        "i miss when things were normal", "vermisse als dinge normal waren"
    ],
    "anti_democratic_markers": [
        "democracy is broken", "demokratie ist kaputt", "voting is pointless", "wählen ist sinnlos",
        "politicians are all same", "politiker sind alle gleich", "it's all rigged", "alles manipuliert",
        "courts protect criminals", "gerichte schützen verbrecher", "we live in dictatorship",
        "taxation is theft", "steuern sind diebstahl"
    ]
}

INITIAL_PROMPTS = {
    "en": {
        "choose_language": "Choose language (English/Deutsch):",
        "enter_language": "Language: ",
        "enter_region": "📍 What's your region? (e.g. berlin, nrw, bremen):",
        "describe_concern": """💬 I'm here to help you explore any concerns you might have about someone showing signs of radicalization. 
This is a safe, anonymous space to share your observations. Could you tell me what's been worrying you?"""
    },
    "de": {
        "choose_language": "Wähle Sprache (English/Deutsch):",
        "enter_language": "Sprache: ",
        "enter_region": "📍 Was ist deine Region? (z.B. berlin, nrw, bremen):",
        "describe_concern": """💬 Ich bin hier, um Ihnen zu helfen, Bedenken bezüglich möglicher Radikalisierungsanzeichen zu besprechen. 
Dies ist ein sicherer, anonymer Raum, um Ihre Beobachtungen zu teilen. Was bereitet Ihnen Sorgen?"""
    }
}

def contains_embedded_code(text: str) -> bool:
    """Enhanced function using simple pattern matching with LLM backup"""
    if text in _code_check_cache:
        return _code_check_cache[text]

    try:
        # Simple fallback check for obvious codes
        obvious_codes = ["88", "18", "444", "28", "döp dodö dop", "arardddar", "foreigners out", "ausländer raus"]
        lower_text = text.lower()
        result = any(code in lower_text for code in obvious_codes)
        
        _code_check_cache[text] = result
        return result
    except Exception:
        _code_check_cache[text] = False
        return False

def normalize_text(text):
    """Normalize umlauts and accented characters for consistent comparison."""
    text = unicodedata.normalize("NFKD", text)
    return "".join([c for c in text if not unicodedata.combining(c)]).lower()

def is_generic_behavioral_change(message: str, sentiment: dict) -> bool:
    lower = message.lower()
    generic_keywords = [
        "not hungry", "upset", "frustrated", "didn't get recognition", "sits alone",
        "won't eat", "won't talk", "lost motivation", "doesn't join", "withdrawn"
    ]
    emotional = sentiment["compound"] <= -0.5
    return emotional and any(k in lower for k in generic_keywords) and not contains_embedded_code(lower)

class RadicalizationBot:
    def __init__(self, session_id=None):
        self.session_start_time = datetime.now(timezone.utc)
        if session_id is None:
            session_id = str(uuid.uuid4())
        self.session_id = session_id
        self.language = "en"
        self.region = "default"
        self.chat_history = []
        self.observed_flags = []
        self.asked_followups = []
        self.conversation_depth = 0
        self.irrelevant_response_count = 0
        self.max_depth = 10
        self.last_bot_response = ""
        self.asked_fallbacks = set()
        self.high_risk_consecutive = 0
        self.conversation_context = []
        
        # Create log directories if they don't exist
        self.log_file = "/app/logs/bot_session_logs.jsonl"
        self.error_log_file = "/app/logs/bot_error_logs.jsonl"
        self._ensure_log_directories()
        
        # Lazy loaded models
        self.model = None
        self.sentiment_analyzer = None
        self.hate_classifier = None
        self.followup_categories = {
            "exact_words":        ["exact","exactly","word","said","phrase","specific","quote","language"],
            "timeline":           ["when","how long","started","began","recently","lately","timeline","progression"],
            "social_context":     ["friends","family","group","others","social","around","people","relationships"],
            "online_behavior":    ["online","internet","social media","websites","platforms","digital","telegram","apps"],
            "symbols_codes":      ["symbols","codes","numbers","images","flags","signs","visual","display"],
            "behavioral_changes": ["behavior","acting","changed","mood","attitude","reactions"],
            "escalation":         ["worse","escalate","more","violent","aggressive","extreme"],
            "group_involvement":  ["group","organization","meeting","events","gathering","community","involvement"],
            "violence_indicators":["violence","threats","weapons","harm","dangerous","physical","aggressive"]
        }
        # Initialize NLTK data
        _ensure_nltk_data()
        
        # Load support data from Excel
        excel_path = os.path.join(os.path.dirname(__file__), "region.xlsx")
        self.region_support_data = self.load_support_data_from_excel(excel_path)

    def _ensure_log_directories(self):
        """Ensure log directories exist"""
        try:
            log_dir = os.path.dirname(self.log_file)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
        except Exception as e:
            print(f"[WARNING] Could not create log directory: {e}")
            self.log_file = "bot_session_logs.jsonl"
            self.error_log_file = "bot_error_logs.jsonl"
    
    def _get_model(self):
        if self.model is None:
            self.model = ChatOpenAI(
                model="meta-llama-3.1-8b-instruct",
                temperature=0.3,
                max_tokens=100,
                openai_api_key=API_KEY,
                openai_api_base="https://chat-ai.academiccloud.de/v1",
                request_timeout=30,
                max_retries=2
            )
        return self.model

    def _get_sentiment_analyzer(self):
        if self.sentiment_analyzer is None:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        return self.sentiment_analyzer

    def _get_hate_classifier(self):
        if self.hate_classifier is None:
            self.hate_classifier = pipeline(
                "text-classification",
                model="facebook/roberta-hate-speech-dynabench-r4-target"
            )
        return self.hate_classifier
    
    @classmethod
    def _load_cached_support_data(cls, file_path):
        if not hasattr(cls, '_cached_support_data'):
            try:
                df = pd.read_excel(file_path)
                support_dict = {}
                for _, row in df.iterrows():
                    region_key = str(row['Bundesland']).strip().lower()
                    website = str(row['Internetauftritt']).strip() if not pd.isna(row['Internetauftritt']) else ""
                    
                    contact = str(row['Kontaktmöglichkeiten'])
                    emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", contact)
                    phone_match = re.search(r"(?:Tel\.?:?\s*)?((?:\+49|0)[\d\s\-()\/]+)", contact, re.IGNORECASE)

                    support_dict[region_key] = {
                        "website": website,
                        "email": ", ".join(emails) if emails else "",
                        "phone": phone_match.group(1).strip() if phone_match else ""
                    }

                cls._cached_support_data = support_dict

            except Exception as e:
                cls._cached_support_data = {
                    "default": {
                        "website": "https://www.zentrum-demokratische-kultur.de",
                        "email": "helppreventradicalization@gmail.com",
                        "phone": "+49 000 0000000"
                    }
                }
        return cls._cached_support_data

    def log_error(self, error_type: str, message: str, context: dict = None):
        log_entry = {
            "session_id": self.session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": error_type,
            "message": message,
            "context": context or {}
        }
        try:
            with open(self.error_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"[ERROR LOGGING FAILED] {e}")

    def load_support_data_from_excel(self, file_path):
        return self._load_cached_support_data(file_path)

    def set_language(self, lang):
        self.language = "de" if lang.lower() in ["de", "deutsch", "german"] else "en"

    def set_region(self, region):
        if not region or not isinstance(region, str):
            self.region = "default"
            return
        
        normalized_input = (region.strip().lower()
                        .replace("ä", "a").replace("ö", "o").replace("ü", "u")
                        .replace("ß", "ss")
                        .replace("-", " ").replace("  ", " ").strip())
        
        for loaded_region in self.region_support_data.keys():
            normalized_loaded = (loaded_region.strip().lower()
                            .replace("ä", "a").replace("ö", "o").replace("ü", "u")
                            .replace("ß", "ss")
                            .replace("-", " ").replace("  ", " ").strip())
            
            if normalized_input == normalized_loaded:
                self.region = loaded_region
                print(f"[INFO] Region set to: {self.region}")
                return
        
        print(f"[WARNING] Region '{region}' not found. Available regions: {list(self.region_support_data.keys())}")
        self.region = "default"

    def validate_region(self, user_region):
        if not user_region or not isinstance(user_region, str):
            self.region = "default"
            return False
        
        normalized_input = (user_region.strip().lower()
                        .replace("ä", "a").replace("ö", "o").replace("ü", "u")
                        .replace("ß", "ss")
                        .replace("-", " ").replace("  ", " ").strip())
        
        for loaded_region in self.region_support_data.keys():
            normalized_loaded = (loaded_region.strip().lower()
                            .replace("ä", "a").replace("ö", "o").replace("ü", "u")
                            .replace("ß", "ss")
                            .replace("-", " ").replace("  ", " ").strip())
            
            if normalized_input == normalized_loaded:
                self.region = loaded_region
                return True
        
        print(f"[WARNING] Region '{user_region}' not found. Using 'default'.")
        self.region = "default"
        return False

    def is_irrelevant(self, message: str) -> bool:
        clarification_phrases = [
            "that's not what i meant", "i meant", "i just said", 
            "i already answered", "as i mentioned", "didn't mean that",
            "i told you", "already told", "i said", "you asked that"
        ]
        message_lower = message.lower().strip()

        # Treat as relevant if it's clarification
        if any(phrase in message_lower for phrase in clarification_phrases):
            return False

        last_bot_message = self.last_bot_response or ""
        cache_key = f"{self.language}:{last_bot_message}:{message}"
        if cache_key in _relevance_cache:
            return _relevance_cache[cache_key]

        prompt = {
            "en": f"""Bot asked: "{last_bot_message}"\nUser replied: "{message}"\nIs this relevant in a radicalization counseling context? Reply only: Relevant or Irrelevant""",
            "de": f"""Bot fragte: "{last_bot_message}"\nNutzer antwortete: "{message}"\nIst dies im Kontext einer Radikalisierungsberatung relevant? Antworte nur: Relevant oder Irrelevant"""
        }[self.language]

        try:
            response = self._get_model().invoke(prompt).content.strip().lower()
            result = "irrelevant" in response and "relevant" not in response
            _relevance_cache[cache_key] = result
            return result
        except Exception:
            return False


    def analyze_input(self, message: str):
        sentiment = self._get_sentiment_analyzer().polarity_scores(message)
        try:
            hate = self._get_hate_classifier()(message)[0]
        except Exception:
            hate = {"label": "neutral", "score": 0.0}
        return sentiment, hate

    def extract_risk_indicators(self, message):
        """Enhanced risk indicator extraction with comprehensive LLM analysis"""
        indicators = []
        
        try:
            enhanced_prompt = {
                "en": f"""Analyze this message for radicalization indicators in German right-wing extremism context:

Message: "{message}"

Provide categories separated by commas from: xenophobic,hate_speech,conspiracy,victimization,ideological,anti_state,symbolic_code,digital_radicalization
Return 'none' if no clear indicators.""",
                "de": f"""Analysiere diese Nachricht auf Radikalisierungsindikatoren im deutschen Rechtsextremismus-Kontext:

Nachricht: "{message}"

Gib Kategorien durch Kommas getrennt zurück aus: xenophobic,hate_speech,conspiracy,victimization,ideological,anti_state,symbolic_code,digital_radicalization
Gib 'none' zurück wenn keine klaren Indikatoren."""
            }[self.language]

            response = self._get_model().invoke(enhanced_prompt).content.strip().lower()
            
            if response and response != 'none':
                llm_categories = [cat.strip() for cat in response.split(",") if cat.strip()]
                valid_categories = ['xenophobic', 'hate_speech', 'conspiracy', 'victimization', 'ideological', 'anti_state', 'symbolic_code', 'digital_radicalization']
                for category in llm_categories:
                    if category not in indicators and category in valid_categories:
                        indicators.append(category)

        except Exception as e:
            self.log_error("EnhancedRiskDetection", str(e), {"input": message})
        
        return indicators

    def detect_risk_tags_with_llm(self, message):
        """Enhanced violence and behavioral risk detection"""
        prompt = {
            "en": f"""Analyze this message for immediate risk indicators:

PHYSICAL_VIOLENCE: Threats, actual violence, weapons, throwing objects, physical harm
GESTURE_AGGRESSION: Intimidating gestures, threatening postures, aggressive displays
PEER_PRESSURE: Group influence, recruitment attempts, social manipulation
RAPID_SHIFT: Sudden behavioral changes, mood swings, personality shifts

Message: "{message}"

Return applicable tags separated by commas, or 'none' if no risks detected.
Tags: physical_violence,gesture_aggression,peer_pressure,rapid_shift""",
            "de": f"""Analysiere diese Nachricht auf unmittelbare Risikoindikatoren:

PHYSICAL_VIOLENCE: Drohungen, tatsächliche Gewalt, Waffen, Gegenstände werfen, körperlicher Schaden
GESTURE_AGGRESSION: Einschüchternde Gesten, bedrohliche Haltungen, aggressive Darstellungen
PEER_PRESSURE: Gruppeneinfluss, Rekrutierungsversuche, soziale Manipulation
RAPID_SHIFT: Plötzliche Verhaltensänderungen, Stimmungsschwankungen, Persönlichkeitsveränderungen

Nachricht: "{message}"

Gib zutreffende Tags durch Kommas getrennt zurück oder 'none'.
Tags: physical_violence,gesture_aggression,peer_pressure,rapid_shift"""
        }[self.language]

        try:
            response = self._get_model().invoke(prompt).content.strip().lower()
            tags = [tag.strip() for tag in response.split(",") if tag.strip() != "none"]
            return tags
        except Exception as e:
            self.log_error("LLMRiskTagDetectionError", str(e), {"input": message})
            return []

    def determine_stage(self):
        """Determines the stage of radicalization based on accumulated observed flags."""
        flags = set(self.observed_flags)

        if "physical_violence" in flags or "symbolic_code" in flags:
            return "stage_3"

        if "ideological" in flags and (
            "anti_state" in flags or "conspiracy" in flags or 
            "victimization" in flags or "xenophobic" in flags
        ):
            return "stage_2"

        if "ideological" in flags or "behavior" in flags:
            return "stage_1"

        return "none"

    def _add_flag(self, condition, flag_name, score_delta=1):
        """Helper method to add flags and return score delta"""
        if condition and flag_name not in self.observed_flags:
            self.observed_flags.append(flag_name)
            return score_delta
        return 0   

    def _detect_violence_intent(self, message: str) -> bool:
        """Enhanced violence detection using LLM"""
        if message in _violence_cache:
            return _violence_cache[message]
            
        # Quick keyword check first
        violence_keywords = [
            "throw", "throwing", "thrown", "hit", "attack", "harm", "hurt", "kill", 
            "werfen", "schlagen", "angreifen", "stones", "steine", "insulted", 
            "beleidigt", "violence", "gewalt", "aggressive", "aggressiv"
        ]
        
        message_lower = message.lower()
        if any(keyword in message_lower for keyword in violence_keywords):
            _violence_cache[message] = True
            return True

        # Enhanced LLM detection
        prompt = {
            "en": f"""Does this message describe violent intent or actions toward people? "{message}"
            Respond with: YES or NO""",
            "de": f"""Beschreibt diese Nachricht gewalttätige Absicht oder Handlungen gegen Menschen? "{message}"
            Antworte mit: JA oder NEIN"""
        }[self.language]

        try:
            response = self._get_model().invoke(prompt).content.strip().lower()
            result = response in ["yes", "ja"]
            _violence_cache[message] = result
            return result
        except Exception as e:
            self.log_error("ViolenceDetectionError", str(e), {"input": message})
            return False

    def _render_steering_response(self) -> str:
        """
        LLM-driven off-topic redirection (no “friend” hard-coding).
        """
        last = next(
            (m.replace("User: ", "") for m in reversed(self.chat_history) if m.startswith("User: ")),
            ""
        )
        steer_prompt = (
            f"You are an empathetic counselor-bot. The user just said (in {self.language}):\n"
            f"“{last}”\n"
            "That remark is off-topic for discussing radicalization signs. "
            "In 1–2 concise, compassionate sentences, gently steer them back by asking: "
            "“What changes have you observed in the person you’re concerned about—their views or behavior?” "
            f"Write your response in {self.language}."
        )
        return self._get_model().invoke(steer_prompt).content.strip()


    def _calculate_risk_level_from_flags(self, flags=None):
        """Calculate risk level based on observed flags."""
        if flags is None:
            flags = self.observed_flags
        
        flags_set = set(flags)
        
        # High risk indicators
        high_risk_flags = {"physical_violence", "hate", "symbolic_code"}
        if any(flag in flags_set for flag in high_risk_flags):
            return "high"
        
        # Moderate risk indicators
        risky_domains = {
            "ideological", "anti_state", "conspiracy",
            "xenophobic", "hate_speech", "victimization"
        }
        if len(flags_set.intersection(risky_domains)) >= 2:
            return "moderate"
        
        # Low risk indicators
        low_risk_flags = {"ideological", "negative_sentiment", "behavior"}
        if any(flag in flags_set for flag in low_risk_flags):
            return "low"
        
        return "none"

    def assess_risk(self, sentiment, hate, message, user_input, llm_reply, is_repetition=False):
        score = 0

        # Classifier and symbolic codes
        score += self._add_flag(hate["label"].lower() in ["hateful", "hate speech"], "hate", score_delta=2)
        score += self._add_flag(sentiment["compound"] <= -0.5, "negative_sentiment", score_delta=1)
        score += self._add_flag(contains_embedded_code(message), "symbolic_code", score_delta=2)

        # Check for generic behavior change
        if is_generic_behavioral_change(message, sentiment):
            self._add_flag(True, "non_radical_behavior")
            active_radical_flags = set(self.observed_flags).intersection({
                "ideological", "symbolic_code", "conspiracy", "hate_speech",
                "anti_state", "xenophobic", "victimization"
            })
            base_risk = ("none" if not active_radical_flags else "low")
            return self._check_conclusion_conditions(base_risk, user_input, llm_reply)

        # Enhanced violence detection
        violence_detected = self._detect_violence_intent(message)
        if violence_detected:
            self._add_flag(True, "physical_violence", score_delta=3)
            score += 3

        # Extract risk indicators using enhanced LLM analysis
        risk_indicators = self.extract_risk_indicators(message)
        for indicator in risk_indicators:
            if indicator not in self.observed_flags:
                self.observed_flags.append(indicator)

        # Detect inferred tags
        inferred_tags = self.detect_risk_tags_with_llm(message)
        new_flags = [tag for tag in inferred_tags if tag not in self.observed_flags]
        self.observed_flags.extend(new_flags)

        # Tag-based severity scoring
        for tag in new_flags:
            if tag in ["symbolic_code", "hate_speech", "physical_violence"]:
                score += 2
            else:
                score += 1

        # Domain-based bonus
        risky_domains = {
            "ideological", "anti_state", "conspiracy",
            "xenophobic", "hate_speech", "victimization"
        }
        if len(set(self.observed_flags).intersection(risky_domains)) >= 3:
            score += 2

        risk_level = self._calculate_risk_level_from_flags()
        if self.conversation_depth < 10:
            return (risk_level, self.observed_flags)
        return self._check_conclusion_conditions(risk_level, user_input, llm_reply)

    def final_decision(self, risk_level):
        support = self.region_support_data.get(self.region, self.region_support_data.get("default"))
        website = support.get("website", "")
        email = support.get("email", "")
        phone = support.get("phone", "")

        lang = self.language
        stage = self.determine_stage()

        if risk_level == "high":
            conclusion_msg = {
                "en": (
                    f"⚠️ This sounds very serious. Please don't wait to seek professional help.\n\n"
                    f"📧 Email: {email}\n"
                    f"📞 Phone: {phone}\n"
                    f"🌐 Website: {website}\n\n"
                    f"These professionals can provide immediate guidance and support."
                ),
                "de": (
                    f"⚠️ Das klingt sehr ernst. Bitte zögere nicht, dir professionelle Hilfe zu suchen.\n\n"
                    f"📧 E-Mail: {email}\n"
                    f"📞 Telefon: {phone}\n"
                    f"🌐 Website: {website}\n\n"
                    f"Diese Fachleute können dir sofortige Beratung bieten."
                )
            }[lang]
        elif risk_level == "moderate":
            conclusion_msg = {
                "en": (
                    f"There are multiple warning signs here. I recommend speaking with professionals.\n\n"
                    f"📧 Email: {email}\n"
                    f"📞 Phone: {phone}\n"
                    f"🌐 Website: {website}\n\n"
                    f"These experts can provide proper assessment."
                ),
                "de": (
                    f"Es gibt mehrere Warnzeichen. Ich empfehle, mit Fachleuten zu sprechen.\n\n"
                    f"📧 E-Mail: {email}\n"
                    f"📞 Telefon: {phone}\n"
                    f"🌐 Website: {website}\n\n"
                    f"Diese Experten können eine Bewertung vornehmen."
                )
            }[lang]
        elif risk_level == "low":
            conclusion_msg = {
                "en": (
                    f"Even though the signs may not be urgent, it's important that you're paying attention.\n\n"
                    f"📧 Email: {email}\n"
                    f"🌐 Website: {website}\n\n"
                    f"These resources are available if you need guidance."
                ),
                "de": (
                    f"Auch wenn die Anzeichen nicht dringend sind, ist es wichtig, dass du aufmerksam bleibst.\n\n"
                    f"📧 E-Mail: {email}\n"
                    f"🌐 Website: {website}\n\n"
                    f"Diese Hilfsangebote stehen zur Verfügung."
                )
            }[lang]
        else:
            conclusion_msg = {
                "en": (
                    f"I'm not detecting clear signs of radicalization based on what you've told me.\n\n"
                    f"📧 Email: {email}\n"
                    f"🌐 Website: {website}\n\n"
                    f"Professional counselors can provide better assessment."
                ),
                "de": (
                    f"Ich kann keine klaren Anzeichen für Radikalisierung erkennen.\n\n"
                    f"📧 E-Mail: {email}\n"
                    f"🌐 Website: {website}\n\n"
                    f"Professionelle Berater können eine bessere Einschätzung bieten."
                )
            }[lang]

        return conclusion_msg

    def log_interaction(self, user_input, bot_response, risk_level, stage=None, extra_data=None):
        """Logs the user's interaction with the bot."""
        log_entry = {
            "session_id": self.session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "language": self.language,
            "region": self.region,
            "user_input": user_input,
            "bot_response": bot_response,
            "risk_level": risk_level,
            "observed_flags": self.observed_flags.copy(),
            "stage": stage,
        }
        if extra_data:
            log_entry.update(extra_data)
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Logging error: {e}")

    def _check_conclusion_conditions(self, risk_level, user_input=None, llm_reply=None):
        if self.conversation_depth < 10:
            return (risk_level, self.observed_flags)

        if user_input:
            user_lower = user_input.lower().strip()
            support_words = ["support", "help", "resource", "service", "assistance", "unterstützung", "hilfe"]
            if any(word in user_lower for word in support_words):
                stage = self.determine_stage()
                conclusion = self.final_decision(risk_level)
                self.log_interaction(user_input, conclusion, risk_level, stage, {"conclusion_reason": "user_requested_support"})
                return conclusion

        # Step 1: After 5 user turns (10 total), prompt for more info
        if self.conversation_depth == 10 and not hasattr(self, 'has_prompted_for_more'):
            self.has_prompted_for_more = True
            self.prompted_at_turn = self.conversation_depth
            followup_check_msg = {
                "en": "You've shared some important information. Is there anything else you'd like to tell me about their behavior?",
                "de": "Du hast wichtige Informationen geteilt. Gibt es noch etwas, das du mir über ihr Verhalten erzählen möchtest?"
            }[self.language]
            self.chat_history.append(f"Bot: {followup_check_msg}")
            self.last_bot_response = followup_check_msg
            return followup_check_msg

        if hasattr(self, 'prompted_at_turn') and self.conversation_depth == self.prompted_at_turn + 1:
            if self._user_indicates_no_more_info(user_input):
                stage = self.determine_stage()
                conclusion = self.final_decision(risk_level)
                self.log_interaction(user_input, conclusion, risk_level, stage, {"conclusion_reason": "no_more_info_after_prompt"})
                return conclusion
            elif self._user_requests_support(user_input):
                stage = self.determine_stage()
                conclusion = self.final_decision(risk_level)
                self.log_interaction(user_input, conclusion, risk_level, stage, {"conclusion_reason": "user_confirmed_support_after_prompt"})
                return conclusion
            elif self._user_wants_to_continue(user_input):
                self.max_depth = self.prompted_at_turn + 6
                self.high_risk_consecutive = 0
                if user_input and llm_reply:
                    self.log_interaction(user_input, llm_reply, risk_level, self.determine_stage())
                return (risk_level, self.observed_flags)
            else:
                stage = self.determine_stage()
                conclusion = self.final_decision(risk_level)
                self.log_interaction(user_input, conclusion, risk_level, stage, {"conclusion_reason": "ambiguous_response"})
                return conclusion

        if hasattr(self, 'prompted_at_turn') and self.conversation_depth < self.max_depth:
            if risk_level == "high":
                self.high_risk_consecutive += 1
            else:
                self.high_risk_consecutive = 0

            if user_input and self._user_indicates_no_more_info(user_input):
                stage = self.determine_stage()
                conclusion = self.final_decision(risk_level)
                self.log_interaction(user_input, conclusion, risk_level, stage, {"conclusion_reason": "no_more_info_extended"})
                return conclusion

            if user_input and llm_reply:
                self.log_interaction(user_input, llm_reply, risk_level, self.determine_stage())
            return (risk_level, self.observed_flags)

        if self.conversation_depth >= self.max_depth or self.irrelevant_response_count >= 8 or self._user_indicates_no_more_info(user_input):
            stage = self.determine_stage()
            conclusion = self.final_decision(risk_level)
            self.log_interaction(user_input, conclusion, risk_level, stage, {"conclusion_reason": "final_exit"})
            return conclusion

        if user_input and llm_reply:
            self.log_interaction(user_input, llm_reply, risk_level, self.determine_stage())
        return (risk_level, self.observed_flags)


    def _user_indicates_no_more_info(self, user_input: str) -> bool:
        """Enhanced detection for when user has exhausted information using LLM"""
        if not user_input:
            return False
            
        cache_key = f"no_more_info:{self.language}:{user_input}"
        if cache_key in _llm_response_cache:
            return _llm_response_cache[cache_key]
        
        prompt = {
            "en": f"""Does this user response indicate they have no more information? "{user_input}"
            Answer with just: YES or NO""",
            "de": f"""Zeigt diese Nutzerantwort an, dass sie keine weiteren Informationen haben? "{user_input}"
            Antworte nur mit: JA oder NEIN"""
        }[self.language]

        try:
            response = self._get_model().invoke(prompt).content.strip().lower()
            result = response in ["yes", "ja"]
            _llm_response_cache[cache_key] = result
            return result
        except Exception as e:
            self.log_error("NoMoreInfoDetection", str(e), {"input": user_input})
            return False
        
    def _user_requests_support(self, user_input: str) -> bool:
        """Dynamic detection for support service requests using LLM"""
        if not user_input:
            return False
            
        cache_key = f"requests_support:{self.language}:{user_input}"
        if cache_key in _llm_response_cache:
            return _llm_response_cache[cache_key]

        prompt = {
            "en": f"""Is this user requesting support services? "{user_input}"
            Answer with just: YES or NO""",
            "de": f"""Fordert dieser Nutzer Unterstützungsdienste an? "{user_input}"
            Antworte nur mit: JA oder NEIN"""
        }[self.language]

        try:
            response = self._get_model().invoke(prompt).content.strip().lower()
            result = response in ["yes", "ja"]
            _llm_response_cache[cache_key] = result
            return result
        except Exception as e:
            self.log_error("SupportRequestDetection", str(e), {"input": user_input})
            return False

    def _user_wants_to_continue(self, user_input: str) -> bool:
        """Dynamic detection for when user wants to share more using LLM"""
        if not user_input:
            return False
            
        cache_key = f"wants_continue:{self.language}:{user_input}"
        if cache_key in _llm_response_cache:
            return _llm_response_cache[cache_key]

        prompt = {
            "en": f"""Does this user want to continue sharing information? "{user_input}"
            Answer with just: YES or NO""",
            "de": f"""Möchte dieser Nutzer weiterhin Informationen teilen? "{user_input}"
            Antworte nur mit: JA oder NEIN"""
        }[self.language]

        try:
            response = self._get_model().invoke(prompt).content.strip().lower()
            result = response in ["yes", "ja"]
            _llm_response_cache[cache_key] = result
            return result
        except Exception as e:
            self.log_error("WantsToContinueDetection", str(e), {"input": user_input})
            return False
        
    def _get_therapeutic_response(self):
        """Enhanced therapeutic response with comprehensive information gathering"""
        if not hasattr(self, 'irrelevant_response_count'):
            self.irrelevant_response_count = 0
        
        self.irrelevant_response_count += 1
        current_risk = self._calculate_risk_level_from_flags()
        
        # Risk-aware therapeutic responses with detailed questioning
        if current_risk in ["moderate", "high"]:
            advanced_therapeutic_responses = {
                "en": [
                    "I understand this is serious. Can you tell me exactly what specific words or phrases they use that concern you?",
                    "This sounds concerning. What specific online activities or groups have you noticed them involved in?",
                    "I can see why you're worried. What particular symbols, images, or codes have you seen them display or share?",
                ],
                "de": [
                    "Ich verstehe, dass das ernst ist. Kannst du mir genau sagen, welche spezifischen Wörter oder Phrasen sie verwenden, die dich beunruhigen?",
                    "Das klingt besorgniserregend. Welche spezifischen Online-Aktivitäten oder Gruppen hast du bemerkt, in die sie involviert sind?",
                    "Ich kann verstehen, warum du dir Sorgen machst. Welche besonderen Symbole, Bilder oder Codes hast du sie zeigen oder teilen sehen?",
                ]
            }[self.language]
            responses = advanced_therapeutic_responses
        else:
            # Basic therapeutic responses with foundational detail gathering
            basic_therapeutic_responses = {
                "en": [
                    "I understand you're concerned. Can you describe what specific changes in their speech or behavior first caught your attention?",
                    "It's natural to feel uncertain about this. What particular words, phrases, or topics do they bring up repeatedly?",
                    "Your instincts brought you here for a reason. What specific situations or conversations have made you most uncomfortable?",
                ],
                "de": [
                    "Ich verstehe, dass du dir Sorgen machst. Kannst du beschreiben, welche spezifischen Veränderungen in ihrer Sprache oder ihrem Verhalten dir zuerst aufgefallen sind?",
                    "Es ist natürlich, sich dabei unsicher zu fühlen. Welche besonderen Wörter, Phrasen oder Themen bringen sie wiederholt zur Sprache?",
                    "Deine Instinkte haben dich aus einem Grund hierher gebracht. Welche spezifischen Situationen oder Gespräche haben dich am meisten unwohl fühlen lassen?",
                ]
            }[self.language]
            responses = basic_therapeutic_responses
        
        index = min(self.irrelevant_response_count - 1, len(responses) - 1)
        return responses[index]

    def _identify_missing_details(self) -> Optional[str]:
        """
        Identify the most important missing follow-up category.
        1) Ask the LLM for its top pick.
        2) If that pick is already asked or is 'COMPREHENSIVE_COMPLETE', use _select_next_category().
        3) Return None if no categories remain.
        """
        # 1. If no history, start with the very first category
        if not self.chat_history:
            return next(iter(self.followup_categories), None)

        # 2. Generate LLM hint (cached)
        conversation_text = " ".join(
            msg.replace("User: ", "")
            for msg in self.chat_history 
            if msg.startswith("User: ")
        )
        cache_key = f"missing:{self.language}:{conversation_text}"
        if cache_key in _llm_response_cache:
            llm_choice = _llm_response_cache[cache_key]
        else:
            prompt = {
                "en": f"""Analyze this conversation and return the MOST IMPORTANT missing category:
Conversation: "{conversation_text}"
Choose one: {', '.join(cat.upper() for cat in self.followup_categories)}
OR return COMPREHENSIVE_COMPLETE.""",
                "de": f"""Analysiere dieses Gespräch und gib die WICHTIGSTE fehlende Kategorie zurück:
Gespräch: "{conversation_text}"
Wähle eine: {', '.join(cat.upper() for cat in self.followup_categories)}
ODER gib COMPREHENSIVE_COMPLETE zurück."""
            }[self.language]
            try:
                llm_choice = self._get_model().invoke(prompt).content.strip().upper()
            except Exception:
                llm_choice = "BEHAVIORAL_CHANGES"
            _llm_response_cache[cache_key] = llm_choice

        # 3. Normalize & validate
        valid = {cat.upper() for cat in self.followup_categories} | {"COMPREHENSIVE_COMPLETE"}
        if llm_choice not in valid:
            llm_choice = next(iter(self.followup_categories), None)

        # 4. If LLM suggests complete or already asked, pick the next unasked
        normalized = llm_choice.lower()
        if normalized == "comprehensive_complete" or normalized in self.asked_followups:
            return self._select_next_category()

        return normalized

    def _should_offer_support(self) -> bool:
        """
        Return True when we’ve reached at least 5 user–bot exchanges 
        AND there are no more fresh categories to probe.
        """
        MIN_PROBES = 5
        # conversation_depth counts bot+user turns; each user reply increments it by 1
        # so after 5 user replies, conversation_depth >= 10
        user_turns = self.conversation_depth // 2
        no_more = self._select_next_category() is None
        return user_turns >= MIN_PROBES and no_more

    def _append_to_history(self, role: str, message: str):
        if role.lower() == "user":
            if not hasattr(self, "user_messages_seen"):
                self.user_messages_seen = []
            if message.strip().lower() not in [m.strip().lower() for m in self.user_messages_seen]:
                self.user_messages_seen.append(message)
            self.chat_history.append(f"User: {message}")
            self.conversation_depth += 1
        elif role.lower() == "bot":
            self.chat_history.append(f"Bot: {message}")
            self.last_bot_response = message


    def _track_followup_categories(self, response):
        """Track follow-up question categories to avoid repetition across the entire session."""
        # Ensure the list exists
        if not hasattr(self, 'asked_followups'):
            self.asked_followups = []

        response_lower = response.lower()
        # Iterate through the shared mapping of categories → keywords
        for category, keywords in self.followup_categories.items():
            # If any keyword for this category appears in the bot's question...
            if any(kw in response_lower for kw in keywords):
                # ...and we haven't already asked this category...
                if category not in self.asked_followups:
                    # ...then mark it as asked
                    self.asked_followups.append(category)

    def _summarize_conversation(self) -> str:
        """
        Summarizes the conversation history into a condensed form.
        Used for passing to the LLM prompt.
        """
        if not self.chat_history:
            return "No significant conversation yet."

        # Keep only the last 6 exchanges (user and bot)
        recent_turns = self.chat_history[-12:]  # 6 user + 6 bot max
        return "\n".join(recent_turns)

    def generate_llm_response(self, user_message: str) -> str:
        if self.is_irrelevant(user_message):
            resp = self._render_steering_response()
            self._append_to_history("bot", resp)
            return resp

        self._append_to_history("user", user_message)
        summary = self._summarize_conversation()
        exhausted = self._user_indicates_no_more_info(user_message)

        # Enforce: Ask all categories before fallback
        if self.conversation_depth >= 10 and self._select_next_category() is None:
            exhausted = True
        else:
            exhausted = False

            prompt = self._build_expert_radicalization_prompt(
                summary=summary, exhausted=True, focus_category=None
            )
            resp = self._get_model().invoke(prompt).content.strip()
            self._append_to_history("bot", resp)
            return resp

        # Next focus category
        missing = self._identify_missing_details()
        if missing and missing.lower() in self.asked_followups:
            missing = None
        focus = missing or self._select_next_category()

        #  Prevent repeat of already-parsed quotes
        last_5_user_inputs = [
            m.replace("User: ", "").strip().lower() for m in self.chat_history[-10:]
            if m.startswith("User:")
        ]

        if user_message.strip().lower() in last_5_user_inputs:
            focus = self._select_next_category()


        prompt = self._build_expert_radicalization_prompt(
            summary=summary, exhausted=exhausted, focus_category=focus
        )
        resp = self._get_model().invoke(prompt).content.strip()

        if focus:
            if missing not in self.asked_followups:
                self.asked_followups.append(focus)
        self._append_to_history("bot", resp)
        return resp

    
    def _build_enhanced_radicalization_context(self, user_message, recent_context, missing_details, user_exhausted):
        """Build comprehensive context with radicalization expertise from uploaded documents"""

        # Determine the next follow-up category to cover (used in prompt)
        investigation_focus = {
            "EXACT_WORDS": {
                "en": "specific words, phrases, slogans, or coded language (e.g., '88', 'great replacement', 'lügenpresse', 'umvolkung', 'volkstod')",
                "de": "spezifische Wörter, Phrasen, Slogans oder kodierte Sprache (z.B. '88', 'großer Austausch', 'Lügenpresse', 'Umvolkung', 'Volkstod')"
            },
            "BEHAVIORAL_PATTERNS": {
                "en": "behavioral changes, social withdrawal, aggressive reactions, distrust patterns, 'us vs them' thinking",
                "de": "Verhaltensänderungen, sozialer Rückzug, aggressive Reaktionen, Misstrauensmuster, 'Wir gegen Die'-Denken"
            },
            "ESCALATION_TIMELINE": {
                "en": "timeline of radicalization progression, when changes started, triggering events, escalation patterns",
                "de": "Zeitverlauf der Radikalisierung, wann Veränderungen begannen, auslösende Ereignisse, Eskalationsmuster"
            },
            "SOCIAL_CONTEXT": {
                "en": "social isolation, group involvement, online communities, influence networks, peer pressure patterns",
                "de": "soziale Isolation, Gruppenbeteiligung, Online-Gemeinschaften, Einflussnetzwerke, Gruppendruck"
            },
            "ONLINE_ACTIVITY": {
                "en": "online platforms (Telegram, 4chan, Rumble), alternative media consumption, conspiracy research behavior",
                "de": "Online-Plattformen (Telegram, 4chan, Rumble), alternative Mediennutzung, Verschwörungsrecherche-Verhalten"
            },
            "SYMBOLS_CODES": {
                "en": "visual symbols (black sun, celtic cross, reichsflagge), numeric codes (88, 18, 28, 444), memes, emojis",
                "de": "visuelle Symbole (schwarze Sonne, Keltenkreuz, Reichsflagge), Zahlencodes (88, 18, 28, 444), Memes, Emojis"
            },
            "VIOLENCE_INDICATORS": {
                "en": "physical aggression, threats, weapons interest, violent rhetoric, justification of extreme actions",
                "de": "körperliche Aggression, Drohungen, Waffeninteresse, gewaltverherrlichende Rhetorik, Rechtfertigung extremer Handlungen"
            },
            "GROUP_INVOLVEMENT": {
                "en": "organized groups, recruitment experiences, meetings, events, leadership structures, peer networks",
                "de": "organisierte Gruppen, Rekrutierungserfahrungen, Treffen, Veranstaltungen, Führungsstrukturen, Peer-Netzwerke"
            }
        }.get(missing_details, {
            "en": "specific observable signs of right-wing extremist thinking or behavior",
            "de": "spezifische beobachtbare Anzeichen rechtsextremistischen Denkens oder Verhaltens"
        })

        current_focus = investigation_focus[self.language]

        # Build context for prompt function
        context = {
            "user_input": user_message,
            "conversation_flow": recent_context,
            "radicalization_focus": current_focus,
            "user_status": "exhausted_current_topic" if user_exhausted else "providing_information",
            "conversation_stage": f"exchange_{self.conversation_depth}_of_critical_first_5" if self.conversation_depth <= 5 else f"extended_exchange_{self.conversation_depth}",
            "detected_concerns": ", ".join(set(self.observed_flags)) if self.observed_flags else "none_yet_detected",
            "missing_category": missing_details,
            "previous_probes": ", ".join(self.asked_followups) if hasattr(self, 'asked_followups') and self.asked_followups else "none",
            "radicalization_stage_assessment": self._assess_current_stage_indicators()
        }

        return context

    def _assess_current_stage_indicators(self):
        """Assess current radicalization stage based on comprehensive indicators from documents"""
        if not self.observed_flags:
            return "stage_assessment_pending"
        
        flags_set = set(self.observed_flags)
        
        # Stage 3 indicators (Advanced Risk) - from documents
        stage_3_indicators = {
            "physical_violence", "symbolic_code", "hate_speech", "group_involvement", 
            "digital_radicalization", "anti_democratic"
        }
        if len(flags_set.intersection(stage_3_indicators)) >= 2:
            return "stage_3_far_right_oriented_high_risk"
        
        # Stage 2 indicators (Active Development) - from documents
        stage_2_indicators = {
            "ideological", "conspiracy", "anti_state", "xenophobic", 
            "victimization", "symbolic_code"
        }
        if len(flags_set.intersection(stage_2_indicators)) >= 2:
            return "stage_2_seeking_orientation_moderate_risk"
        
        # Stage 1 indicators (Early Risk) - from documents
        stage_1_indicators = {
            "ideological", "negative_sentiment", "victimization", "anti_democratic"
        }
        if any(flag in flags_set for flag in stage_1_indicators):
            return "stage_1_initial_risk_factors"
        
        return "stage_assessment_insufficient_data"

    def _build_expert_radicalization_prompt(
        self,
        summary: str,
        exhausted: bool,
        focus_category: Optional[str] = None,
    ) -> str:
        """
        Off-topic: focus_category=='steering'
        Support: exhausted==True
        Otherwise: probe only focus_category
        """
        p = (
            "You are an empathetic counselor-bot. "
            "Use concise (2–3 sentence) replies, echoing the user’s last concern.\n\n"
            f"Conversation summary:\n{summary}\n\n"
        )

        # Off-topic steering
        if focus_category == "steering":
            steer = {
                "en": "That question seems unrelated. What changes have you observed in the person you’re concerned about—their views or behavior?",
                "de": "Diese Frage passt nicht zum Thema. Welche Veränderungen im Verhalten oder in den Ansichten der betroffenen Person sind dir aufgefallen?"
            }[self.language]
            return p + steer

        # Investigation focus
        p += "Investigation Focus:\n"
        if focus_category and focus_category in self.followup_categories:
            p += f"- {focus_category}: {self.followup_categories[focus_category]}\n\n"
        else:
            for cat, desc in self.followup_categories.items():
                p += f"- {cat}: {desc}\n"
            p += "\n"

        # Support referral
        if exhausted:
            p += (
                "The user has no more details. "
                "Ask: “Would you like me to connect you with a support service? (yes/no)”"
            )

        # Rules
        p += (
            "\n\nRules:\n"
            "1. Never repeat a category already asked.  \n"
            "2. Ask no more than 2 questions per turn.  \n"
            "3. If all categories covered, ask “Anything else?” then offer support.  \n"
            "4. Reply in the user’s chosen language."
        )
        return p

    
    def _build_hierarchical_prompt(self, context, user_exhausted):
        """Build hierarchical prompt with radicalization expertise"""
        
        role_definition = {
            "en": "You are a professional behavioral assessment counselor specializing in radicalization prevention.",
            "de": "Du bist ein professioneller Verhaltensbeurteilungsberater, spezialisiert auf Radikalisierungsprävention."
        }[self.language]
        
        context_section = {
            "en": f"""CONTEXT:
- Current user response: "{context['user_input']}"
- Information focus needed: {context['information_gap']}
- User communication status: {context['user_status']}
- Conversation stage: {context['conversation_stage']}
- Behavioral concerns detected: {context['detected_concerns']}
- Missing assessment category: {context['missing_category']}
- Recent follow-up types asked: {context['asked_followups']}""",
            
            "de": f"""KONTEXT:
- Aktuelle Nutzerantwort: "{context['user_input']}"
- Benötigter Informationsfokus: {context['information_gap']}
- Nutzer-Kommunikationsstatus: {context['user_status']}
- Gesprächsstadium: {context['conversation_stage']}
- Erkannte Verhaltenssorgen: {context['detected_concerns']}
- Fehlende Bewertungskategorie: {context['missing_category']}
- Kürzlich gestellte Follow-up-Typen: {context['asked_followups']}"""
        }[self.language]
        
        if user_exhausted:
            task_instruction = {
                "en": """TASK: Generate ONE perspective-shifting question (15-25 words) that explores a different behavioral dimension.
Focus on gathering NEW information for radicalization stage assessment.""",
                "de": """AUFGABE: Generiere EINE perspektivwechselnde Frage (15-25 Wörter), die eine andere Verhaltensdimension erkundet.
Fokussiere auf das Sammeln NEUER Informationen für Radikalisierungsstufenbewertung."""
            }[self.language]
        else:
            task_instruction = {
                "en": f"""TASK: Generate ONE focused follow-up question (15-25 words) about {context['information_gap']}.
Ask about concrete actions, words, reactions, or observable changes.""",
                "de": f"""AUFGABE: Generiere EINE fokussierte Nachfrage (15-25 Wörter) über {context['information_gap']}.
Frage nach konkreten Handlungen, Worten, Reaktionen oder beobachtbaren Veränderungen."""
            }[self.language]
        
        output_constraints = {
            "en": """OUTPUT REQUIREMENTS:
- Respond ONLY in English with empathy
- Generate exactly ONE question that ends with '?'
- 15-25 words maximum
- Include brief empathetic acknowledgment + specific question""",
            "de": """AUSGABEANFORDERUNGEN:
- Antworte NUR auf Deutsch mit Empathie
- Generiere genau EINE Frage, die mit '?' endet
- 15-25 Wörter maximal
- Füge kurze empathische Anerkennung + spezifische Frage hinzu"""
        }[self.language]
        
        return f"""{role_definition}

{context_section}

{task_instruction}

{output_constraints}

Your empathetic question:"""

    def _validate_and_enhance_response(self, response, user_message, user_exhausted):
        """Validate and enhance the LLM response"""
        if not response:
            return None
        
        cleaned = self._clean_llm_response(response)
        
        if not cleaned:
            return None
        
        validation_checks = {
            "has_content": len(cleaned) > 10,
            "reasonable_length": 15 <= len(cleaned) <= 250,
            "ends_with_question": cleaned.rstrip().endswith('?'),
            "language_consistency": self._check_language_consistency(cleaned),
            "no_processing_artifacts": not any(term in cleaned.lower() for term in 
                                            ['generate', 'strategy', 'analysis', 'processing', 'system', 'task', 'output']),
            "behavioral_focus": any(word in cleaned.lower() for word in 
                                ['what', 'how', 'when', 'where', 'who', 'do', 'does', 'did', 
                                'was', 'wie', 'wann', 'wo', 'wer', 'macht', 'hat', 'war', 'can', 'könnt'])
        }
        
                # 🛡️ Additional hardening: block LLM junk, hallucinations, or broken responses
        forbidden_phrases = [
            "as an ai", "i cannot", "my purpose", "i don't have emotions",
            "as a chatbot", "according to my training", "i'm just a model"
        ]
        if any(bad in cleaned.lower() for bad in forbidden_phrases):
            return None

        # Prevent multiple questions in a single reply
        if cleaned.count("?") > 1:
            return None

        # Avoid stray assistant-format phrases
        if any(token in cleaned.lower() for token in ["your question:", "task:", "instruction:", "output:"]):
            return None

        # Normalize punctuation
        if not cleaned.endswith("?"):
            cleaned = cleaned.rstrip(".! ")
            cleaned += "?"

        # Normalize casing
        if cleaned and not cleaned[0].isupper():
            cleaned = cleaned[0].upper() + cleaned[1:]

        return cleaned if 10 <= len(cleaned) <= 250 else None

    def _select_next_category(self) -> Optional[str]:
        """
        Pick the next unasked follow-up category in priority order.
        Returns the first category from self.followup_categories.keys() that
        has not yet been recorded in self.asked_followups, or None if all have been asked.
        """
        for category in self.followup_categories.keys():
            if category not in self.asked_followups:
                return category
        return None  

    def _check_language_consistency(self, response):
        """Check if response matches expected language"""
        if self.language == "en":
            english_indicators = ['what', 'how', 'when', 'where', 'who', 'do', 'does', 'did', 'can', 'have', 'has', 'would', 'could']
            return any(word in response.lower() for word in english_indicators)
        else:
            german_indicators = ['was', 'wie', 'wann', 'wo', 'wer', 'macht', 'hat', 'kann', 'kannst', 'hast', 'könn', 'werd', 'würd']
            return any(word in response.lower() for word in german_indicators)

    def _repair_response(self, response):
        """Attempt to repair minor response issues"""
        prefixes_to_remove = ['bot:', 'assistant:', 'question:', 'frage:', 'answer:', 'antwort:', 'your empathetic question:']
        for prefix in prefixes_to_remove:
            if response.lower().startswith(prefix):
                response = response[len(prefix):].strip()
        
        if not response.endswith('?'):
            response = response.rstrip('.!') + '?'
        
        if response:
            response = response[0].upper() + response[1:]
        
        if (10 <= len(response) <= 250 and 
            response.endswith('?') and
            self._check_language_consistency(response)):
            return response
        
        return None

    def _clean_llm_response(self, response):
        """Enhanced response cleaning that preserves more valid content"""
        if not response:
            return ""

        lines = response.split('\n')
        content_lines = []

        for line in lines:
            line = line.strip()

            skip_indicators = [
                'def ', 'class ', 'import ', 'if __', 'try:', 'except:',
                'system_prompt', 'generate_', 'processing_', 'strategy:',
                'analysis:', 'instruction:', 'context:', 'task:', 'output:', 'your empathetic question:'
            ]

            if any(line.lower().startswith(indicator.lower()) for indicator in skip_indicators):
                continue

            if line and len(line) > 1:
                content_lines.append(line)

        cleaned = ' '.join(content_lines).strip()

        # 🛡️ Block hallucinated tool/function outputs
        artifact_patterns = ["tool.call", "function.call", "action(", "response =", "tool_response", "generate(", "strategy:"]
        if any(p in cleaned.lower() for p in artifact_patterns):
            return ""

        cleaned = re.sub(r'^(bot:|assistant:|question:|frage:|your empathetic question:)\s*', '', cleaned, flags=re.IGNORECASE)

        question_indicators = ['what', 'how', 'when', 'where', 'who', 'do', 'can', 'have',
                            'was', 'wie', 'wann', 'wo', 'wer', 'kannst', 'hast', 'macht', 'könnt']

        if any(word in cleaned.lower() for word in question_indicators):
            return cleaned

        if len(cleaned) > 250 or any(term in cleaned.lower() for term in ['generate', 'system', 'processing']):
            return ""

        return cleaned

    def _generate_contextual_fallback(self, user_input, has_violence=False):
        """Generate fallback response when primary generation fails"""
        user_messages = [msg.replace("User: ", "") for msg in self.chat_history if msg.startswith("User: ")]
        recent_responses = "; ".join(user_messages[-3:]) if user_messages else ""
        current_risk = self._calculate_risk_level_from_flags()

        user_exhausted = self._user_indicates_no_more_info(user_input)

        fallback_prompt = {
            "en": f"""Generate an empathetic follow-up question for radicalization assessment:

    SITUATION:
    - User response: "{user_input}"
    - Recent conversation: {recent_responses}
    - Risk level: {current_risk}
    - User exhausted current topic: {"Yes" if user_exhausted else "No"}

    Generate ONE clear, empathetic follow-up question (15-25 words) about behavioral observations.

    Respond only with your question.""",
            "de": f"""Generiere eine empathische Nachfrage für Radikalisierungsbewertung:

    SITUATION:
    - Nutzerantwort: "{user_input}"
    - Kürzliches Gespräch: {recent_responses}
    - Risikolevel: {current_risk}
    - Nutzer hat aktuelles Thema erschöpft: {"Ja" if user_exhausted else "Nein"}

    Generiere EINE klare, empathische Nachfrage (15-25 Wörter) über Verhaltensbeobachtungen.

    Antworte nur mit deiner Frage."""
        }[self.language]

        try:
            response = self._get_model().invoke(fallback_prompt).content.strip()
            response = self._clean_llm_response(response)

            # Apply same hardening checks as validate_enhance
            if response and 10 <= len(response) <= 250 and "?" in response:
                if not any(bad in response.lower() for bad in ["as an ai", "output:", "instruction:", "task:"]):
                    if response.count("?") == 1:
                        return response
        except Exception as e:
            self.log_error("ContextualFallbackError", str(e), {"input": user_input})

        # 🧠 If everything failed, use structured last-resort question
        fallback_bank = {
            "en": [
                "That sounds difficult. Have they posted anything unusual online or shared strange videos lately?",
                "Thanks for telling me. Do they use any symbols or numbers that stand out — like 88 or certain emojis?",
                "That’s concerning. Have they made new friends or started spending time with different groups recently?",
                "I understand. Have they expressed anger towards institutions or mentioned phrases like 'system collapse'?",
                "You're doing the right thing by talking about it. Have they ever said violence might be necessary to defend beliefs?"
            ],
            "de": [
                "Das klingt schwierig. Haben sie in letzter Zeit ungewöhnliche Inhalte online geteilt?",
                "Danke fürs Teilen. Verwenden sie auffällige Symbole oder Zahlen wie 88 oder bestimmte Emojis?",
                "Das ist besorgniserregend. Haben sie neue Freunde gefunden oder verbringen Zeit mit anderen Gruppen?",
                "Ich verstehe. Haben sie Wut auf Institutionen geäußert oder von 'Systemkollaps' gesprochen?",
                "Es ist gut, dass du darüber sprichst. Haben sie jemals Gewalt zur Verteidigung ihrer Überzeugungen erwähnt?"
            ]
        }

        fallback_list = fallback_bank[self.language]
        index = min(self.conversation_depth, len(fallback_list) - 1)
        return fallback_list[index]

    def get_response(self, user_input):
        user_input = re.sub(r'\b(\w+) mr\b', r'\1 me', user_input, flags=re.IGNORECASE)
        user_input = re.sub(r'\bwith mr\b', 'with me', user_input, flags=re.IGNORECASE)

        if user_input.lower() == "end_conversation":
            risk_level = self._calculate_risk_level_from_flags()
            conclusion = self.final_decision(risk_level)
            self.log_interaction(user_input, conclusion, risk_level, self.determine_stage())
            return conclusion

        recent_user_inputs = [
            msg.replace("User: ", "") for msg in self.chat_history[-6:]
            if msg.startswith("User: ")
        ]
        is_actual_repetition = user_input.strip().lower() in [
            inp.strip().lower() for inp in recent_user_inputs
        ]

        if self.is_irrelevant(user_input):
            response = self._get_therapeutic_response()
            sentiment, hate = self.analyze_input(user_input)
            result = self.assess_risk(sentiment, hate, user_input, user_input, response)

            if isinstance(result, str) and self.conversation_depth < 10:
               # print("Blocked premature support offer (irrelevant)")
                return self._generate_contextual_fallback(user_input)

            if isinstance(result, str):
                return result

            self.log_interaction(user_input, response, "none")
            return response

        if hasattr(self, 'irrelevant_response_count'):
            self.irrelevant_response_count = 0

        sentiment, hate = self.analyze_input(user_input)

        # Determine next unasked follow-up category
        missing = self._identify_missing_details()
        if missing and missing.lower() in self.asked_followups:
            missing = self._select_next_category()

        # Forcefully pick the next follow-up category if missing is None
        if not missing:
            missing = self._select_next_category()

        if missing:
        #    print(f"Focusing on follow-up category: {missing}")

            llm_reply = self.generate_llm_response(user_input).strip()

        # High-priority override: If user mentions symbols, pamphlets, or codes and category not yet asked
        lower_input = user_input.lower()
        force_symbol_check = any(keyword in lower_input for keyword in ["symbol", "code", "zeichen", "pamphlet", "flyer", "number", "zahl", "emoji", "image"])

        # Force proactive probing if high-risk indicators detected (e.g. stage 2 or 3)
        stage = self.determine_stage()
        if (stage in ["stage_2", "stage_3"] or force_symbol_check) and "symbols_codes" not in self.asked_followups:
            print("Proactively asking about symbols/codes due to stage or input cue")
            self.asked_followups.append("symbols_codes")
            llm_reply = {
                "en": "Can you describe any specific symbols, numbers, or codes that stood out — on paper, clothing, or online?",
                "de": "Gab es bestimmte Symbole, Zahlen oder Codes, die dir aufgefallen sind – auf Papier, Kleidung oder online?"
            }[self.language]

        llm_reply = self._clean_llm_response(llm_reply)

        # Fallback if repeated last bot message
        if hasattr(self, 'last_bot_response') and llm_reply == self.last_bot_response:
            #print("Detected repeated bot message, switching to fallback.")
            llm_reply = self._generate_contextual_fallback(user_input)
        else:
            # Record follow-up category to avoid asking again
            if missing and missing not in self.asked_followups:
                self.asked_followups.append(missing)

        result = self.assess_risk(sentiment, hate, user_input, user_input, llm_reply, is_repetition=is_actual_repetition)

        if isinstance(result, tuple):
            return llm_reply

        if isinstance(result, str):
            if self.conversation_depth < 10 and not self._user_requests_support(user_input):
               # print("Blocked premature support offer (main)")
                return self._generate_contextual_fallback(user_input)
            return result

        return llm_reply


if __name__ == "__main__":
    bot = RadicalizationBot()
    print(f"\n Your anonymous session ID: {bot.session_id}\n")
    print(f"\n Ihre anonyme Sitzungs-ID: {bot.session_id}\n")
    print(INITIAL_PROMPTS["en"]["choose_language"])
    lang_input = input(INITIAL_PROMPTS["en"]["enter_language"]).strip()
    bot.set_language(lang_input)
    
    prompts = INITIAL_PROMPTS[bot.language]
    while True:
        region_input = input(prompts["enter_region"] + " ").strip()
        if bot.validate_region(region_input):
            break
    
    print("\n" + prompts["describe_concern"])
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            farewell = {
                "en": "You've done something important by speaking up. Take care.",
                "de": "Es war wichtig, dass du darüber gesprochen hast. Pass auf dich auf."
            }[bot.language]
            print("Bot:", farewell)
            break
        print("Bot:", bot.get_response(user_input))