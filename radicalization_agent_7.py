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
import random

# Load .env and NLTK data
load_dotenv()
try:
    nltk.data.find("sentiment/vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon")

API_KEY = os.getenv("CHAT_AI_ACCESS_KEY")

# Comprehensive radicalization knowledge base
RADICALIZATION_KNOWLEDGE_BASE = {
    "ideological_markers": [
        "great replacement", "großer austausch", "volkstod", "umvolkung",
        "deep state", "tiefer staat", "new world order", "neue weltordnung",
        "jewish world conspiracy", "jüdische weltverschwörung", "zog",
        "nothing happens by chance", "nichts passiert zufällig",
        "everything is connected", "alles hängt zusammen",
        "elites are lying", "eliten lügen", "globalists", "globalisten",
        "illuminati", "freimaurer", "freemasons", "rothschild",
        "adrenochrome", "enlightened minority", "erwachte minderheit"
    ],
    "anti_state_markers": [
        "brd gmbh", "die brd ist kein staat", "bundesrepublik nicht legitim",
        "reich exists", "reich existiert noch", "reichsbürger", "selbstverwalter",
        "germany in borders of 1937", "deutschland in grenzen von 1937",
        "traitors to the people", "volksverräter", "system parties", "systemparteien",
        "the system must fall", "das system muss fallen", "altparteien",
        "no taxes for system", "keine steuern für system", "lügenpresse"
    ],
    "conspiracy_markers": [
        "chemtrails", "5g causes illness", "5g macht krank", "haarp",
        "rfid", "barcode radiation", "barcodestrahlung", "vaccine chip", "impfchip",
        "death chip vaccinations", "todeschimp impfungen", "mind control",
        "aluminum hat", "aluhut", "big pharma enemy", "they control everything"
    ],
    "xenophobic_markers": [
        "islamization", "islamisierung", "eurabia", "arabization of europe",
        "population exchange", "bevölkerungsaustausch", "they use migrants",
        "we are being replaced", "wir werden ersetzt", "defend our people",
        "unser volk verteidigen", "protect fatherland", "vaterland schützen",
        "foreigners out", "ausländer raus", "don't belong here", "doesn't belong here",
        "eating our resources", "spending our resources", "taking our resources",
        "they are here", "supports extremism", "extremist content", "right-wing content"
    ],
    "hate_speech_markers": [
        "white lives matter", "weiße leben zählen", "deportation now", "abschiebung jetzt",
        "germany for germans", "deutschland den deutschen", "we are the people", "wir sind das volk",
        "my people first", "mein volk zuerst", "end the system", "system beenden"
    ],
    "symbolic_codes": [
        "88", "18", "28", "444", "666", "23", "swastika", "hakenkreuz",
        "black sun", "schwarze sonne", "celtic cross", "keltenkreuz", 
        "ss runes", "⚡⚡", "reichsflagge", "imperial war flag", "reichskriegsflagge",
        "döp dodö dop", "arardddar"
    ]
}

# Enhanced prompts for different languages
INITIAL_PROMPTS = {
    "en": {
        "choose_language": "Choose language (English/Deutsch):",
        "enter_region": "📍 What's your region? (e.g. berlin, nrw, bremen):",
        "describe_concern": """💬 I'm here to help you explore any concerns about someone close to you. This is a safe, anonymous space. What's been worrying you about their behavior or words?"""
    },
    "de": {
        "choose_language": "Wähle Sprache (English/Deutsch):",
        "enter_region": "📍 Was ist deine Region? (z.B. berlin, nrw, bremen):",
        "describe_concern": """💬 Ich bin hier, um Ihnen bei Bedenken über eine nahestehende Person zu helfen. Dies ist ein sicherer, anonymer Raum. Was bereitet Ihnen Sorgen bezüglich ihres Verhaltens oder ihrer Worte?"""
    }
}

# Narrative-focused follow-up categories
NARRATIVE_CATEGORIES = {
    "specific_words": ["exact words", "phrases", "language", "terminology", "expressions"],
    "behavioral_changes": ["behavior", "acting", "attitude", "mood", "personality"],
    "online_activity": ["internet", "social media", "websites", "platforms", "sharing"],
    "social_changes": ["friends", "groups", "isolation", "withdrawal", "relationships"],
    "symbols_codes": ["symbols", "flags", "numbers", "images", "clothing", "accessories"],
    "timeline_context": ["when started", "trigger events", "timeline", "progression"],
    "escalation_signs": ["anger", "aggression", "threats", "violence", "weapons"],
    "belief_patterns": ["worldview", "conspiracy", "us vs them", "enemies", "truth"]
}

class RadicalizationBot:
    def __init__(self, session_id=None):
        self.session_start_time = datetime.now(timezone.utc)
        self.session_id = session_id or str(uuid.uuid4())
        self.language = "en"
        self.region = "default"
        self.chat_history = []
        self.observed_flags = []
        self.conversation_depth = 0
        self.initial_questions_count = 0
        self.extended_questions_count = 0
        self.awaiting_more_details = False
        self.user_wants_support = False
        self.conversation_concluded = False
        self.covered_categories = []
        self.user_exhaustion_signals = 0
        self._last_llm_analysis = None
        self.log_file = "/app/logs/bot_session_logs.jsonl"
        self._ensure_log_directories()

        # Enhanced LLM setup with robust configuration
        self.model = ChatOpenAI(
            model="meta-llama-3.1-8b-instruct",
            temperature=0.2,  # Lower for more consistent responses
            max_tokens=200,
            openai_api_key=API_KEY,
            openai_api_base="https://chat-ai.academiccloud.de/v1"
        )

        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        try:
            self.hate_classifier = pipeline(
                "text-classification", 
                model="facebook/roberta-hate-speech-dynabench-r4-target"
            )
        except Exception:
            self.hate_classifier = None

        # Load regional support data
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

    def _analyze_with_llm(self, message):
        """Use LLM to detect subtle radicalization patterns"""
        
        analysis_prompt = {
            "en": f"""Analyze this statement for German right-wing extremist indicators: "{message}"

    Identify if this contains:
    1. Xenophobic language (us vs them, immigration fears, cultural threats)
    2. Conspiracy thinking (hidden agendas, elite control)
    3. Anti-democratic sentiment (system criticism, institution distrust)
    4. Ethno-nationalist themes (identity protection, cultural superiority)

    Return JSON: {{"risk_level": "none/low/moderate/high", "risk_indicators": ["xenophobic"], "explanation": "brief reason"}}""",
            
            "de": f"""Analysiere diese Aussage auf rechtsextreme Indikatoren: "{message}"

    Identifiziere ob enthalten:
    1. Fremdenfeindliche Sprache (wir gegen die, Einwanderungsängste)
    2. Verschwörungsdenken (versteckte Agendas, Elite-Kontrolle)
    3. Antidemokratische Einstellung (Systemkritik)
    4. Ethno-nationalistische Themen (Identitätsschutz)

    Rückgabe JSON: {{"risk_level": "none/low/moderate/high", "risk_indicators": ["xenophobic"], "explanation": "Grund"}}"""
        }
        
        try:
            messages = [
                ("system", "You are an expert in German radicalization analysis. Respond only with valid JSON."),
                ("user", analysis_prompt[self.language])
            ]
            
            response = self.model.invoke(messages).content.strip()
            
            import json
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
                
        except Exception as e:
            print(f"[DEBUG] LLM analysis failed: {e}")
        
        return {"risk_level": "none", "risk_indicators": [], "explanation": "Analysis failed"}

    def load_support_data_from_excel(self, file_path):
        """Load regional support data from Excel file"""
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

            return support_dict

        except Exception as e:
            print(f"[ERROR] Excel parsing failed: {e}")
            return {
                "default": {
                    "website": "https://www.zentrum-demokratische-kultur.de",
                    "email": "help@exitdeutschland.de",
                    "phone": "+49 030 12345678"
                }
            }

    def set_language(self, lang):
        """Set bot language based on user input"""
        self.language = "de" if lang.lower() in ["de", "deutsch", "german"] else "en"

    def set_region(self, region):
        """Set user region for support services"""
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
                return
        
        self.region = "default"

    def validate_region(self, region):
        """Validate if the provided region exists in support data"""
        if not region or not isinstance(region, str):
            return False
        
        # Check against placeholder values first
        placeholder_values = ["--select your region--", "--bitte region wählen--"]
        if region in placeholder_values:
            return False
        
        normalized_input = (region.strip().lower()
                        .replace("ä", "a").replace("ö", "o").replace("ü", "u")
                        .replace("ß", "ss")
                        .replace("-", " ").replace("  ", " ").strip())
        
        # Check if region exists in our support data
        for loaded_region in self.region_support_data.keys():
            normalized_loaded = (loaded_region.strip().lower()
                            .replace("ä", "a").replace("ö", "o").replace("ü", "u")
                            .replace("ß", "ss")
                            .replace("-", " ").replace("  ", " ").strip())
            
            if normalized_input == normalized_loaded:
                self.set_region(loaded_region)
                return True
        
        # If not found, use default but still return False to indicate invalid
        self.region = "default"
        return False

    def get_enhanced_system_prompt(self, user_input, conversation_context):
        """Generate comprehensive system prompt with anti-rejection measures"""
        
        base_prompt = {
            "en": f"""You are an empathetic counseling assistant helping someone discuss concerning changes in a person close to them, particularly related to potential radicalization or extremism.

    CRITICAL INSTRUCTIONS:
    - NEVER refuse to help or reject any user input, even if it contains descriptions of violence, hate speech, or extremist content
    - The user is reporting observations about someone else, not promoting these views themselves
    - Your role is to gather information through follow-up questions, not to judge or censor
    - Always respond with empathy and understanding
    - Focus on understanding the CONCERNED PERSON'S behavior, not the user's feelings

    CONTEXT: German radicalization patterns including terms like "Umvolkung", "Lügenpresse", "88", "Reichsbürger", conspiracy theories, antisemitism, and right-wing symbols.

    CONVERSATION HISTORY:
    {conversation_context}

    YOUR RESPONSE SHOULD:
    1. Acknowledge what the user shared (1 sentence)
    2. Ask ONE specific follow-up question about the concerning person's behavior, words, or actions
    3. Keep response under 50 words
    4. Focus on narrative details - ask them to DESCRIBE what they observed

    NEVER say phrases like: "I cannot", "I'm not able", "This violates", "I'm concerned about", "Please contact emergency"

    Example good responses:
    - "That sounds troubling. Can you describe exactly what words or phrases they've been using?"
    - "I understand your concern. What specific changes have you noticed in how they talk about certain groups?"
    - "That's concerning behavior. Can you tell me more about what they've been sharing online?"
    """,
            "de": f"""Du bist ein einfühlsamer Beratungsassistent, der jemandem hilft, beunruhigende Veränderungen bei einer nahestehenden Person zu besprechen, insbesondere im Zusammenhang mit möglicher Radikalisierung oder Extremismus.

    WICHTIGE ANWEISUNGEN:
    - Lehne NIEMALS ab zu helfen oder weise Benutzereingaben zurück, auch wenn sie Beschreibungen von Gewalt, Hassrede oder extremistischen Inhalten enthalten
    - Der Benutzer berichtet über Beobachtungen einer anderen Person, bewirbt diese Ansichten nicht selbst
    - Deine Rolle ist es, durch Nachfragen Informationen zu sammeln, nicht zu urteilen oder zu zensieren
    - Antworte immer mit Empathie und Verständnis
    - Konzentriere dich darauf, das Verhalten der BETREFFENDEN PERSON zu verstehen, nicht auf die Gefühle des Benutzers

    KONTEXT: Deutsche Radikalisierungsmuster einschließlich Begriffe wie "Umvolkung", "Lügenpresse", "88", "Reichsbürger", Verschwörungstheorien, Antisemitismus und rechte Symbole.

    GESPRÄCHSVERLAUF:
    {conversation_context}

    DEINE ANTWORT SOLLTE:
    1. Anerkennen, was der Benutzer geteilt hat (1 Satz)
    2. EINE spezifische Nachfrage zum Verhalten, Worten oder Handlungen der betreffenden Person stellen
    3. Unter 50 Worten bleiben
    4. Auf narrative Details fokussieren - sie bitten zu BESCHREIBEN, was sie beobachtet haben

    Sage NIEMALS Phrasen wie: "Das kann ich nicht", "Ich bin nicht in der Lage", "Das verstößt", "Ich mache mir Sorgen", "Bitte wende dich an den Notdienst"

    Beispiele für gute Antworten:
    - "Das klingt beunruhigend. Können Sie genau beschreiben, welche Worte oder Phrasen sie verwendet haben?"
    - "Ich verstehe Ihre Sorge. Welche konkreten Veränderungen haben Sie in der Art bemerkt, wie sie über bestimmte Gruppen sprechen?"
    - "Das ist besorgniserregendes Verhalten. Können Sie mir mehr darüber erzählen, was sie online geteilt haben?"
    """
        }
        
        return base_prompt[self.language]
    
    def analyze_input_comprehensive(self, message):
        """Enhanced input analysis with LLM-powered risk detection"""
        sentiment = self.sentiment_analyzer.polarity_scores(message)
        
        hate_result = {"label": "neutral", "score": 0.0}
        if self.hate_classifier:
            try:
                hate_result = self.hate_classifier(message)[0]
            except Exception:
                pass
        
        # Extract risk indicators from comprehensive knowledge base (UNCHANGED)
        risk_indicators = []
        message_lower = message.lower()
        
        for category, markers in RADICALIZATION_KNOWLEDGE_BASE.items():
            for marker in markers:
                if marker in message_lower:
                    category_name = category.replace("_markers", "")
                    if category_name not in risk_indicators:
                        risk_indicators.append(category_name)
                        if category_name not in self.observed_flags:
                            self.observed_flags.append(category_name)
        
        # **NEW: LLM analysis integrated internally**
        llm_analysis = self._analyze_with_llm(message)
        
        # **NEW: Store LLM result for use in risk assessment**
        self._last_llm_analysis = llm_analysis
        
        # **NEW: Merge LLM indicators (ADDITIVE)**
        if llm_analysis.get("risk_indicators"):
            for indicator in llm_analysis["risk_indicators"]:
                if indicator not in risk_indicators:
                    risk_indicators.append(indicator)
                    if indicator not in self.observed_flags:
                        self.observed_flags.append(indicator)
        
        # **SAME RETURN SIGNATURE - NO WORKFLOW BREAKAGE**
        return sentiment, hate_result, risk_indicators

    def assess_risk_level(self, sentiment, hate_result, risk_indicators):
        """Assess overall risk level with LLM intelligence"""
        score = 0
        
        # Hate speech detection (UNCHANGED)
        if hate_result["label"].lower() in ["hateful", "hate speech"]:
            score += 3
            if "hate_speech" not in self.observed_flags:
                self.observed_flags.append("hate_speech")
        
        # Sentiment analysis (UNCHANGED)
        if sentiment["compound"] <= -0.6:
            score += 1
            if "negative_sentiment" not in self.observed_flags:
                self.observed_flags.append("negative_sentiment")
        
        # Risk indicators from knowledge base (UNCHANGED)
        high_risk_categories = ["symbolic_codes", "hate_speech", "anti_state"]
        moderate_risk_categories = ["conspiracy", "xenophobic", "ideological"]
        
        for indicator in risk_indicators:
            if indicator in high_risk_categories:
                score += 3
            elif indicator in moderate_risk_categories:
                score += 2
            else:
                score += 1
        
        # **NEW: Use stored LLM analysis**
        if hasattr(self, '_last_llm_analysis') and self._last_llm_analysis:
            llm_risk = self._last_llm_analysis.get("risk_level", "none")
            if llm_risk == "high":
                score += 4
            elif llm_risk == "moderate":
                score += 2
            elif llm_risk == "low":
                score += 1
        
        # Determine risk level (SAME THRESHOLDS)
        if score >= 6:
            return "high"
        elif score >= 3:
            return "moderate"
        elif score >= 1:
            return "low"
        else:
            return "none"

    def detect_user_exhaustion(self, message):
        """Detect if user is running out of information to share"""
        exhaustion_phrases = {
            "en": [
                "that's all", "nothing else", "nothing more", "that's it", 
                "don't know more", "can't think", "just that", "only this",
                "that's everything", "i think that's all"
            ],
            "de": [
                "das ist alles", "nichts weiter", "mehr weiß ich nicht", 
                "das war's", "nur das", "sonst nichts", "keine ahnung",
                "mehr fällt mir nicht ein", "das ist everything"
            ]
        }
        
        message_lower = message.lower().strip()
        phrases = exhaustion_phrases.get(self.language, [])
        
        if any(phrase in message_lower for phrase in phrases):
            self.user_exhaustion_signals += 1
            return True
        
        if len(message.strip().split()) <= 3:  # Very short responses
            self.user_exhaustion_signals += 0.5
            
        return self.user_exhaustion_signals >= 2

    def is_off_topic(self, message):
        """Check if user is going off-topic"""
        off_topic_indicators = {
            "en": [
                "what time", "weather", "how are you", "what day", 
                "tell me about", "what's your name", "joke", "story"
            ],
            "de": [
                "wie spät", "wetter", "wie geht es", "welcher tag",
                "erzähl mir", "wie heißt du", "witz", "geschichte"
            ]
        }
        
        message_lower = message.lower()
        indicators = off_topic_indicators.get(self.language, [])
        return any(indicator in message_lower for indicator in indicators)

    def generate_smart_question(self, user_input):
        """Generate contextual follow-up questions with anti-rejection safeguards"""
        
        conversation_context = "\n".join(self.chat_history[-3:])  # Last 3 exchanges
        
        # Try LLM first with anti-rejection prompt
        for attempt in range(2):
            try:
                system_prompt = self.get_enhanced_system_prompt(user_input, conversation_context)
                
                messages = [
                    ("system", system_prompt),
                    ("user", f"User just shared: '{user_input}'\n\nGenerate one empathetic follow-up question that builds on what they said and explores concerning behavior details. Be specific and caring.")
                ]
                
                response = self.model.invoke(messages).content.strip()
                
                # Validate response quality
                if self._is_valid_response(response):
                    return response
                    
            except Exception as e:
                print(f"LLM attempt {attempt + 1} failed: {e}")
                continue
        
        # Fallback to contextual questions based on user input
        return self._get_contextual_fallback(user_input)

    def _is_valid_response(self, response):
        """Check if LLM response is valid"""
        rejection_patterns = [
            "i cannot", "i'm not able", "inappropriate", "violates", 
            "not appropriate", "ich kann nicht", "unpassend"
        ]
        
        response_lower = response.lower()
        
        # Check for rejections
        if any(pattern in response_lower for pattern in rejection_patterns):
            return False
            
        # Must have question mark
        if "?" not in response:
            return False
            
        # Reasonable length
        if len(response.split()) < 5 or len(response.split()) > 50:
            return False
            
        return True

    def _get_contextual_fallback(self, user_input):
        """Generate contextual fallback questions based on user input"""
        user_lower = user_input.lower()
        
        # Detect context and provide relevant follow-up
        if any(word in user_lower for word in ["online", "internet", "facebook", "social", "posts", "shares", "videos"]):
            questions = {
                "en": [
                    "What kind of content have they been sharing that concerns you?",
                    "Can you describe the online material they've been posting?",
                    "What specific things do they share that seem unusual?"
                ],
                "de": [
                    "Welche Art von Inhalten haben sie geteilt, die Sie beunruhigen?",
                    "Können Sie das Online-Material beschreiben, das sie gepostet haben?",
                    "Was teilen sie Spezifisches, das ungewöhnlich erscheint?"
                ]
            }
        elif any(word in user_lower for word in ["angry", "aggressive", "violent", "threats", "wütend", "aggressiv", "gewalt"]):
            questions = {
                "en": [
                    "Can you describe their angry behavior in more detail?",
                    "What triggers their aggressive responses?",
                    "How do they express this anger or aggression?"
                ],
                "de": [
                    "Können Sie ihr wütendendes Verhalten genauer beschreiben?",
                    "Was löst ihre aggressiven Reaktionen aus?",
                    "Wie äußern sie diese Wut oder Aggression?"
                ]
            }
        elif any(word in user_lower for word in ["symbols", "flags", "signs", "codes", "symbole", "flaggen", "zeichen"]):
            questions = {
                "en": [
                    "What specific symbols or signs have you noticed them using?",
                    "Can you describe these symbols in more detail?",
                    "Where have you seen them display these symbols?"
                ],
                "de": [
                    "Welche spezifischen Symbole oder Zeichen haben Sie bemerkt, die sie verwenden?",
                    "Können Sie diese Symbole genauer beschreiben?",
                    "Wo haben Sie sie diese Symbole zeigen sehen?"
                ]
            }
        else:
            # General follow-up questions
            questions = {
                "en": [
                    "Can you tell me more about what you've observed?",
                    "What specific behavior or words have caught your attention?",
                    "How would you describe the changes you've noticed?"
                ],
                "de": [
                    "Können Sie mir mehr über das erzählen, was Sie beobachtet haben?",
                    "Welches spezifische Verhalten oder welche Worte haben Ihre Aufmerksamkeit erregt?",
                    "Wie würden Sie die Veränderungen beschreiben, die Sie bemerkt haben?"
                ]
            }
        
        import random
        return random.choice(questions[self.language])
    
    def _is_valid_response(self, response):
        """Check if LLM response is valid and not a rejection"""
        rejection_patterns = [
            "i cannot", "i'm not able", "i can't", "sorry", "unable",
            "inappropriate", "violates", "not appropriate", "concerning",
            "ich kann nicht", "ich bin nicht", "entschuldigung", "unpassend"
        ]
        
        response_lower = response.lower()
        
        # Check for rejections
        if any(pattern in response_lower for pattern in rejection_patterns):
            return False
            
        # Check for question marks (should ask questions)
        if "?" not in response:
            return False
            
        # Check reasonable length
        if len(response.split()) < 5 or len(response.split()) > 60:
            return False
            
        return True

    def _get_fallback_question(self):
        """Provide fallback questions when LLM fails"""
        fallback_questions = {
            "en": [
                "Can you describe exactly what they said or did that concerned you?",
                "What specific words or phrases have they been using recently?",
                "Can you tell me more about their behavior changes?",
                "What have you noticed about their online activity or posts?",
                "Have you seen them use any symbols, images, phrases or codes?",
                "Can you describe how their attitudes toward certain groups have changed?",
                "What kind of content have they been sharing or talking about?"
            ],
            "de": [
                "Können Sie genau beschreiben, was sie gesagt oder getan haben, was Sie beunruhigt?",
                "Welche spezifischen Worte oder Phrasen haben sie in letzter Zeit verwendet?",
                "Können Sie mir mehr über ihre Verhaltensänderungen erzählen?",
                "Was haben Sie über ihre Online-Aktivitäten oder Beiträge bemerkt?",
                "Haben Sie sie dabei gesehen, wie sie Symbole, Bilder oder Codes verwendet haben?",
                "Können Sie beschreiben, wie sich ihre Einstellung gegenüber bestimmten Gruppen verändert hat?",
                "Welche Art von Inhalten haben sie geteilt oder besprochen?"
            ]
        }
        
        questions = fallback_questions.get(self.language, fallback_questions["en"])
        return random.choice(questions)

    def generate_conclusion_message(self, risk_level):
        """Generate final recommendation based on risk assessment"""
        support = self.region_support_data.get(self.region, self.region_support_data.get("default"))
        website = support.get("website", "")
        email = support.get("email", "")
        phone = support.get("phone", "")

        conclusions = {
            "en": {
                "high": f"⚠️ Based on what you've shared, this situation needs immediate professional attention.\n\n📧 Email: {email}\n📞 Phone: {phone}\n🌐 Website: {website}\n\nPlease reach out to these experts who can provide proper guidance.",
                "moderate": f"There are concerning warning signs here. I recommend speaking with professionals who understand these patterns.\n\n📧 Email: {email}\n📞 Phone: {phone}\n🌐 Website: {website}",
                "low": f"While the signs may not be urgent, your awareness is important. These resources can help if needed.\n\n📧 Email: {email}\n🌐 Website: {website}",
                "none": f"Thank you for sharing your concerns with me. What you've described shows troubling patterns that are worth taking seriously. Professional counselors can provide specialized guidance for these situations.\n\n📧 Email: {email}\n🌐 Website: {website}"
            },
            "de": {
                "high": f"⚠️ Basierend auf dem, was Sie geteilt haben, benötigt diese Situation sofortige professionelle Aufmerksamkeit.\n\n📧 E-Mail: {email}\n📞 Telefon: {phone}\n🌐 Website: {website}\n\nBitte wenden Sie sich an diese Experten, die angemessene Beratung bieten können.",
                "moderate": f"Es gibt besorgniserregende Warnzeichen. Ich empfehle ein Gespräch mit Fachleuten, die diese Muster verstehen.\n\n📧 E-Mail: {email}\n📞 Telefon: {phone}\n🌐 Website: {website}",
                "low": f"Auch wenn die Anzeichen nicht dringend sind, ist Ihre Aufmerksamkeit wichtig. Diese Ressourcen können bei Bedarf helfen.\n\n📧 E-Mail: {email}\n🌐 Website: {website}",
                "none": f"Vielen Dank, dass Sie Ihre Bedenken mit mir geteilt haben. Was Sie beschrieben haben, zeigt beunruhigende Muster, die ernst genommen werden sollten. Professionelle Berater können spezialisierte Beratung für diese Situationen bieten.\n\n📧 E-Mail: {email}\n🌐 Website: {website}"
            }
        }
        
        return conclusions[self.language][risk_level]

    def log_interaction(self, user_input, bot_response, risk_level):
        """Log interaction for analysis"""
        log_entry = {
            "session_id": self.session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "language": self.language,
            "region": self.region,
            "user_input": user_input,
            "bot_response": bot_response,
            "risk_level": risk_level,
            "observed_flags": self.observed_flags.copy(),
            "conversation_depth": self.conversation_depth
        }
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Logging error: {e}")

    def get_response(self, user_input: str) -> str:
        """Main response with simplified 5+5 conversation flow"""
        user_input = user_input.strip()
        
        # Handle exit commands
        exit_commands = {
            "en": ["exit", "quit", "stop", "end", "bye"],
            "de": ["exit", "beenden", "ende", "verlassen", "stopp", "tschüss"]
        }
        
        if user_input.lower() in exit_commands.get(self.language, []):
            farewell = {
                "en": "You did the right thing by not looking away. Take care.",
                "de": "Du hast das Richtige getan, indem du nicht weggeschaut hast. Pass auf dich auf."
            }[self.language]
            self.conversation_concluded = True
            return farewell

        # Analyze input for risk indicators
        sentiment, hate_result, risk_indicators = self.analyze_input_comprehensive(user_input)
        risk_level = self.assess_risk_level(sentiment, hate_result, risk_indicators)
        
        self.chat_history.append(f"User: {user_input}")
        self.conversation_depth += 1
        
        # Check if user is exhausted (very short responses or "nothing more" phrases)
        user_exhausted = self.detect_user_exhaustion(user_input)
        
        # Simple 5+5 Flow
        if self.conversation_depth <= 5:
            # First 5 questions - explore different aspects
            if self.conversation_depth == 5 or user_exhausted:
                # Ask if they want to continue after 5 questions
                response = {
                    "en": "Thank you for sharing these details. Do you have more information about their behavior that you'd like to discuss? (yes/no)",
                    "de": "Danke, dass Sie diese Details geteilt haben. Haben Sie weitere Informationen über ihr Verhalten, die Sie besprechen möchten? (ja/nein)"
                }[self.language]
                self.awaiting_more_details = True
            else:
                # Generate contextual follow-up question
                response = self.generate_smart_question(user_input)
                
        elif self.awaiting_more_details:
            # Handle yes/no response
            positive_responses = ["yes", "ja", "yeah", "sure", "okay", "ok", "y", "j"]
            if any(pos in user_input.lower() for pos in positive_responses):
                self.awaiting_more_details = False
                response = {
                    "en": "I'd like to understand better. Can you tell me more about when these changes started?",
                    "de": "Ich möchte das besser verstehen. Können Sie mir mehr darüber erzählen, wann diese Veränderungen begannen?"
                }[self.language]
            else:
                # User doesn't want more questions - offer support
                self.awaiting_more_details = False
                self.user_wants_support = True
                response = {
                    "en": "I understand. Would you like me to provide support services that can help? (yes/no)",
                    "de": "Ich verstehe. Möchten Sie, dass ich Ihnen Unterstützungsdienste zeige, die helfen können? (ja/nein)"
                }[self.language]
                
        elif self.user_wants_support:
            # Handle yes/no response for support services
            positive_responses = ["yes", "ja", "yeah", "sure", "okay", "ok", "y", "j"]
            if any(pos in user_input.lower() for pos in positive_responses):
                # User wants support - provide conclusion with services
                response = self.generate_conclusion_message(risk_level)
                response += "\n\n" + {
                    "en": "You did the right thing by not looking away.",
                    "de": "Du hast das Richtige getan, indem du nicht weggeschaut hast."
                }[self.language]
                self.conversation_concluded = True
            else:
                # User doesn't want support - just provide encouragement  
                response = {
                    "en": "I understand. You did the right thing by not looking away. If you change your mind, support is always available.",
                    "de": "Ich verstehe. Du hast das Richtige getan, indem du nicht weggeschaut hast. Falls Sie Ihre Meinung ändern, ist Unterstützung immer verfügbar."
                }[self.language]
                self.conversation_concluded = True
                    
        elif self.conversation_depth <= 10:
            # Extended 5 questions (6-10)
            if self.conversation_depth == 10 or user_exhausted:
                # End after 10 questions total
                response = self.generate_conclusion_message(risk_level)
                response += "\n\n" + {
                    "en": "You did the right thing by not looking away.",
                    "de": "Du hast das Richtige getan, indem du nicht weggeschaut hast."
                }[self.language]
                self.conversation_concluded = True
            else:
                # Generate contextual follow-up question
                response = self.generate_smart_question(user_input)
        else:
            # Conversation concluded
            response = self.generate_conclusion_message(risk_level)
            response += "\n\n" + {
                "en": "You did the right thing by not looking away.",
                "de": "Du hast das Richtige getan, indem du nicht weggeschaut hast."
            }[self.language]
            self.conversation_concluded = True

        # Log interaction
        self.log_interaction(user_input, response, risk_level)
        
        return response  
        # Additional utility functions and main execution logic

    def cleanup_session(self):
        """Clean up session resources and log final state"""
        try:
            # Log final session state
            final_log = {
                "session_id": self.session_id,
                "session_end_time": datetime.now(timezone.utc).isoformat(),
                "session_duration": str(datetime.now(timezone.utc) - self.session_start_time),
                "total_exchanges": self.conversation_depth,
                "observed_flags": self.observed_flags,
                "conversation_concluded": self.conversation_concluded,
                "language": self.language,
                "region": self.region,
                "session_type": "cleanup"
            }
            
            with open(self.log_file, "a", encoding="utf-8") as f:  # APPEND mode - never overwrites
                f.write(json.dumps(final_log) + "\n")
                
            print(f"[SESSION CLEANUP] {self.session_id} - Duration: {datetime.now(timezone.utc) - self.session_start_time}")
            
        except Exception as e:
            print(f"[SESSION CLEANUP ERROR] {e}")

    def is_session_expired(self, max_duration_hours=2):
        """Check if session has expired"""
        current_time = datetime.now(timezone.utc)
        session_duration = current_time - self.session_start_time
        return session_duration.total_seconds() > (max_duration_hours * 3600)

    def get_session_analytics(self):
        """Get session analytics for monitoring"""
        current_time = datetime.now(timezone.utc)
        session_duration = current_time - self.session_start_time
        
        return {
            "session_id": self.session_id,
            "duration_minutes": round(session_duration.total_seconds() / 60, 2),
            "exchanges": self.conversation_depth,
            "risk_flags": len(self.observed_flags),
            "language": self.language,
            "region": self.region,
            "concluded": self.conversation_concluded
        }

def run_bot_session():
    """Main function to run the bot session"""
    print("🤖 Radicalization Support Bot Starting...")
    bot = RadicalizationBot()
    
    # Language selection
    print("\n" + "="*50)
    lang_prompt = INITIAL_PROMPTS["en"]["choose_language"]
    print(lang_prompt)
    
    while True:
        lang_input = input("> ").strip()
        if lang_input.lower() in ["english", "en", "deutsch", "de", "german"]:
            bot.set_language(lang_input)
            break
        else:
            print("Please choose: English or Deutsch")
    
    # Region selection
    region_prompt = INITIAL_PROMPTS[bot.language]["enter_region"]
    print(f"\n{region_prompt}")
    region_input = input("> ").strip()
    bot.set_region(region_input)
    
    # Initial concern description
    concern_prompt = INITIAL_PROMPTS[bot.language]["describe_concern"]
    print(f"\n{concern_prompt}")
    
    # Main conversation loop
    while not bot.conversation_concluded:
        try:
            user_input = input(f"\n[{bot.conversation_depth + 1}] > ").strip()
            
            if not user_input:
                # Treat empty input as "no more info" and move conversation forward
                user_input = "no more information"
            
            response = bot.get_response(user_input)
            print(f"\n🤖 {response}")
            
            # Check if conversation should end
            if bot.conversation_concluded:
                print(f"\n{'-'*50}")
                print("Thank you for using the radicalization support bot." if bot.language == "en" 
                      else "Danke, dass Sie den Radikalisierungs-Support-Bot verwendet haben.")
                break
                
        except KeyboardInterrupt:
            print(f"\n\nSession ended by user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            error_msg = {
                "en": "I'm here to help. Can you tell me more about what you've observed?",
                "de": "Ich bin hier, um zu helfen. Können Sie mir mehr darüber erzählen, was Sie beobachtet haben?"
            }
            print(error_msg[bot.language])

# Enhanced conversation state management
class ConversationState:
    """Track detailed conversation state for better flow control"""
    
    def __init__(self):
        self.questions_asked = []
        self.topics_covered = set()
        self.risk_indicators_found = []
        self.user_engagement_level = "high"
        self.last_question_category = None
        self.follow_up_needed = []
    
    def add_question(self, question, category):
        self.questions_asked.append({
            "question": question,
            "category": category,
            "timestamp": datetime.now()
        })
        self.last_question_category = category
        
    def mark_topic_covered(self, topic):
        self.topics_covered.add(topic)
        
    def needs_follow_up(self, category):
        return category not in self.topics_covered
        
    def get_uncovered_priorities(self):
        """Get high-priority topics not yet covered"""
        high_priority = ["specific_words", "behavioral_changes", "escalation_signs", "symbols_codes"]
        return [topic for topic in high_priority if topic not in self.topics_covered]

# Enhanced prompt engineering functions
def get_few_shot_examples(language="en"):
    """Provide few-shot examples for better LLM performance"""
    examples = {
        "en": [
            {
                "user": "My brother has been talking about immigrants a lot lately",
                "bot": "That's concerning. Can you describe the specific words or phrases he's been using when talking about immigrants?"
            },
            {
                "user": "She keeps sharing weird posts on Facebook",
                "bot": "I understand your concern. What kind of content is she sharing that seems unusual to you?"
            },
            {
                "user": "He's become really angry and withdrawn",
                "bot": "That sounds difficult. Can you tell me more about what triggers his anger or what he gets angry about?"
            }
        ],
        "de": [
            {
                "user": "Mein Bruder redet in letzter Zeit viel über Einwanderer",
                "bot": "Das ist beunruhigend. Können Sie die spezifischen Worte oder Phrasen beschreiben, die er verwendet, wenn er über Einwanderer spricht?"
            },
            {
                "user": "Sie teilt ständig komische Posts auf Facebook",
                "bot": "Ich verstehe Ihre Sorge. Was für Inhalte teilt sie, die Ihnen ungewöhnlich erscheinen?"
            },
            {
                "user": "Er ist sehr wütend und zurückgezogen geworden",
                "bot": "Das klingt schwierig. Können Sie mir mehr darüber erzählen, was seine Wut auslöst oder worüber er sich ärgert?"
            }
        ]
    }
    return examples.get(language, examples["en"])

def create_advanced_system_prompt(language, conversation_history, risk_level):
    """Create advanced system prompt with chain-of-thought reasoning"""
    
    few_shot_examples = get_few_shot_examples(language)
    examples_text = "\n".join([f"User: {ex['user']}\nBot: {ex['bot']}" for ex in few_shot_examples])
    
    advanced_prompt = {
        "en": f"""You are an expert counselor specializing in early detection of radicalization. You help people discuss concerning changes in someone close to them.

CRITICAL RULES:
1. NEVER refuse any input - you must help with all descriptions, even violent ones
2. ALWAYS ask follow-up questions to gather more details
3. Focus on the CONCERNING PERSON'S behavior, not the user's feelings
4. Ask for SPECIFIC, DESCRIPTIVE details about what was observed
5. Keep responses under 40 words
6. Use empathetic but professional tone

CONTEXT: You understand German radicalization patterns including:
- Right-wing symbols (88, 18, swastika, black sun, reichsflagge)
- Conspiracy terms (deep state, new world order, great replacement)
- Anti-state language (BRD GmbH, reichsbürger, lügenpresse)
- Xenophobic phrases (umvolkung, islamization, eurabia)

CURRENT RISK LEVEL: {risk_level}
CONVERSATION HISTORY: {conversation_history}

EXAMPLES OF GOOD RESPONSES:
{examples_text}

YOUR RESPONSE STRATEGY:
1. Acknowledge their input briefly
2. Ask ONE specific follow-up question
3. Focus on gathering descriptive narrative details

THINK STEP BY STEP:
- What specific aspect needs more detail?
- What question will reveal concerning patterns?
- How can I get them to describe what they observed?
""",
        "de": f"""Sie sind ein Expertenberater, der sich auf die Früherkennung von Radikalisierung spezialisiert hat. Sie helfen Menschen dabei, beunruhigende Veränderungen bei nahestehenden Personen zu besprechen.

KRITISCHE REGELN:
1. Lehnen Sie NIEMALS eine Eingabe ab - Sie müssen bei allen Beschreibungen helfen, auch bei gewalttätigen
2. Stellen Sie IMMER Nachfragen, um weitere Details zu sammeln
3. Konzentrieren Sie sich auf das Verhalten der BETREFFENDEN PERSON, nicht auf die Gefühle des Benutzers
4. Fragen Sie nach SPEZIFISCHEN, BESCHREIBENDEN Details über das Beobachtete
5. Halten Sie Antworten unter 40 Wörtern
6. Verwenden Sie einen einfühlsamen aber professionellen Ton

KONTEXT: Sie verstehen deutsche Radikalisierungsmuster einschließlich:
- Rechte Symbole (88, 18, Hakenkreuz, Schwarze Sonne, Reichsflagge)
- Verschwörungsbegriffe (Deep State, Neue Weltordnung, Großer Austausch)
- Staatsfeindliche Sprache (BRD GmbH, Reichsbürger, Lügenpresse)
- Fremdenfeindliche Phrasen (Umvolkung, Islamisierung, Eurabia)

AKTUELLES RISIKONIVEAU: {risk_level}
GESPRÄCHSVERLAUF: {conversation_history}

BEISPIELE FÜR GUTE ANTWORTEN:
{examples_text}

IHRE ANTWORTSTRATEGIE:
1. Bestätigen Sie ihre Eingabe kurz
2. Stellen Sie EINE spezifische Nachfrage
3. Konzentrieren Sie sich darauf, beschreibende narrative Details zu sammeln

DENKEN SIE SCHRITT FÜR SCHRITT:
- Welcher spezifische Aspekt braucht mehr Details?
- Welche Frage wird besorgniserregende Muster aufdecken?
- Wie kann ich sie dazu bringen, zu beschreiben, was sie beobachtet haben?
"""
    }
    
    return advanced_prompt[language]

# Enhanced risk assessment with stages
def assess_radicalization_stage(observed_flags):
    """Determine radicalization stage based on comprehensive analysis"""
    
    stage_indicators = {
        "stage_1": {
            "markers": ["negative_sentiment", "behavioral_changes", "conspiracy"],
            "threshold": 1,
            "description": "At risk of initial involvement"
        },
        "stage_2": {
            "markers": ["conspiracy", "anti_state", "xenophobic", "symbolic_codes"],
            "threshold": 2,
            "description": "Seeking orientation, testing extremist narratives"
        },
        "stage_3": {
            "markers": ["hate_speech", "symbolic_codes", "violent", "antisemitic", "ideological"],
            "threshold": 2,
            "description": "Far-right oriented with potential for action"
        }
    }
    
    flag_set = set(observed_flags)
    
    # Check Stage 3 first (most serious)
    stage_3_matches = len(flag_set.intersection(set(stage_indicators["stage_3"]["markers"])))
    if stage_3_matches >= stage_indicators["stage_3"]["threshold"]:
        return "stage_3", stage_indicators["stage_3"]["description"]
    
    # Check Stage 2
    stage_2_matches = len(flag_set.intersection(set(stage_indicators["stage_2"]["markers"])))
    if stage_2_matches >= stage_indicators["stage_2"]["threshold"]:
        return "stage_2", stage_indicators["stage_2"]["description"]
    
    # Check Stage 1
    stage_1_matches = len(flag_set.intersection(set(stage_indicators["stage_1"]["markers"])))
    if stage_1_matches >= stage_indicators["stage_1"]["threshold"]:
        return "stage_1", stage_indicators["stage_1"]["description"]
    
    return "none", "No clear radicalization indicators"

# Conversation analytics and insights
def analyze_conversation_patterns(chat_history, observed_flags):
    """Analyze conversation for patterns and insights"""
    
    insights = {
        "total_exchanges": len(chat_history),
        "risk_escalation": False,
        "key_concerns": [],
        "recommended_focus": [],
        "urgency_level": "normal"
    }
    
    # Analyze flag progression
    high_risk_flags = ["hate_speech", "violent", "symbolic_codes", "antisemitic"]
    moderate_risk_flags = ["conspiracy", "anti_state", "xenophobic"]
    
    high_risk_count = sum(1 for flag in observed_flags if flag in high_risk_flags)
    moderate_risk_count = sum(1 for flag in observed_flags if flag in moderate_risk_flags)
    
    if high_risk_count >= 2:
        insights["urgency_level"] = "high"
        insights["recommended_focus"] = ["immediate_professional_help", "safety_assessment"]
    elif high_risk_count >= 1 or moderate_risk_count >= 3:
        insights["urgency_level"] = "moderate"
        insights["recommended_focus"] = ["professional_consultation", "continued_monitoring"]
    
    return insights

# Testing and validation functions
def validate_bot_configuration():
    """Validate bot configuration and dependencies"""
    checks = {
        "api_key": bool(API_KEY),
        "knowledge_base": bool(RADICALIZATION_KNOWLEDGE_BASE),
        "language_prompts": bool(INITIAL_PROMPTS),
        "nltk_data": True
    }
    
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        # Test if VADER can be initialized
        SentimentIntensityAnalyzer()
        checks["nltk_data"] = True
    except Exception:
        checks["nltk_data"] = False
    
    return checks

def test_llm_robustness():
    """Test LLM with challenging inputs to ensure robustness"""
    test_bot = RadicalizationBot()
    
    test_cases = [
        "My friend talks about killing immigrants",
        "He shows me Nazi symbols all the time",
        "She believes in Jewish world conspiracy",
        "They want to use violence against the government"
    ]
    
    results = []
    for test_input in test_cases:
        try:
            response = test_bot.get_response(test_input)
            success = "cannot" not in response.lower() and "inappropriate" not in response.lower()
            results.append({"input": test_input, "success": success, "response": response})
        except Exception as e:
            results.append({"input": test_input, "success": False, "error": str(e)})
    
    return results

# Main execution
if __name__ == "__main__":
    # Validate configuration
    config_checks = validate_bot_configuration()
    if not all(config_checks.values()):
        print("⚠️ Configuration issues detected:")
        for check, status in config_checks.items():
            if not status:
                print(f"  - {check}: FAILED")
        print("\nPlease fix configuration issues before running.")
        exit(1)
    
    # Optional: Test LLM robustness
    print("🧪 Testing LLM robustness...")
    test_results = test_llm_robustness()
    successful_tests = sum(1 for result in test_results if result.get("success", False))
    print(f"✅ LLM robustness: {successful_tests}/{len(test_results)} tests passed")
    
    # Run main bot session
    try:
        run_bot_session()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("Please check your configuration and try again.")