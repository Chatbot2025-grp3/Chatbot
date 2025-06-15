import os
import re
import pandas as pd
import nltk
import uuid
import json
from datetime import datetime
from dotenv import load_dotenv
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import random  # Add this import at the top of your script if not already present


# Load .env and NLTK data
load_dotenv()
try:
    nltk.data.find("sentiment/vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon")

API_KEY = os.getenv("CHAT_AI_ACCESS_KEY")

REJECTION_PATTERNS = {
    "en": ["i cannot", "i'm not allowed", "i'm unable", "this violates", "please contact emergency services"],
    "de": ["ich kann nicht", "ich darf nicht", "ich bin nicht in der lage", "das verstößt", "bitte wende dich an die notdienste"]
}
RADICALIZATION_CODES = ["88", "18", "23", "444", "888", "222111", "black sun", "swastika", "reich", "totenkopf", "celtic cross"]

INITIAL_PROMPTS = {
    "en": {
        "choose_language": "Choose language (English/Deutsch):",
        "enter_language": "Language: ",
        "enter_region": "📍 What's your region? (e.g. berlin, nrw, bremen):",
        "describe_concern": "💬 This chat-bot is here to assist you in investigating any worries you may have regarding someone exhibiting early indications of radicalization, particularly in the direction of right-wing extremism. It is intended to be sympathetic, anonymous, and nonjudgmental. (type 'exit' to quit):"
    },
    "de": {
        "choose_language": "Wähle Sprache (English/Deutsch):",
        "enter_language": "Sprache: ",
        "enter_region": "📍 Was ist deine Region? (z.B. berlin, nrw, bremen):",
        "describe_concern": "💬 Dieser Chatbot soll Ihnen dabei helfen, Ihre Bedenken hinsichtlich einer Person zu untersuchen, die erste Anzeichen einer Radikalisierung zeigt, insbesondere in Richtung Rechtsextremismus. Er soll einfühlsam, anonym und vorurteilsfrei sein.(Tippe 'exit' zum Beenden):"
    }
}

def sanitize_response(response: str, fallback: str, previous: str, lang: str, asked_fallbacks=None) -> str:
    rejection_phrases = REJECTION_PATTERNS.get(lang, [])
    
    fallback_variants = {
        "en": [
            "Can you explain a bit more about what's worrying you?",
            "Could you tell me what specifically is making you concerned?",
            "Has something specific triggered this behavior?",
            "Can you tell me how this started, or when you first noticed the change?"
        ],
        "de": [
            "Kannst du mir ein bisschen mehr erzählen, was dich beunruhigt?",
            "Was genau macht dir Sorgen?",
            "Gab es einen Auslöser für dieses Verhalten?",
            "Kannst du mir sagen, wann dieses Verhalten begonnen hat?"
        ]
    }

    if any(p in response.lower() for p in rejection_phrases):
        asked_fallbacks = asked_fallbacks or set()
        options = [f for f in fallback_variants[lang] if f not in asked_fallbacks]
        chosen = options[0] if options else fallback_variants[lang][0]
        asked_fallbacks.add(chosen)
        return chosen

    return response.strip()


def contains_embedded_code(text: str) -> bool:
    lower = text.lower()
    return any(code in lower for code in RADICALIZATION_CODES)

def is_generic_behavioral_change(message: str, sentiment: dict) -> bool:
    lower = message.lower()
    generic_keywords = [
        "not hungry", "upset", "frustrated", "didn't get recognition", "sits alone",
        "won't eat", "won't talk", "lost motivation", "doesn't join", "withdrawn"
    ]
    emotional = sentiment["compound"] <= -0.5
    return any(k in lower for k in generic_keywords) and not contains_embedded_code(lower)

class RadicalizationBot:
    def __init__(self, session_id=None):
        if session_id is None:
            session_id = str(uuid.uuid4())
        self.session_id = session_id
        self.language = "en"
        self.region = "default"
        self.chat_history = []
        self.observed_flags = []
        self.conversation_depth = 0
        self.max_depth = 6
        self.last_bot_response = ""
        self.asked_fallbacks = set()
        self.log_file = "/app/logs/bot_session_logs.jsonl"

        self.model = ChatOpenAI(
            model="meta-llama-3.1-8b-instruct",
            temperature=0.5,
            openai_api_key=API_KEY,
            openai_api_base="https://chat-ai.academiccloud.de/v1"
        )

        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.hate_classifier = pipeline("text-classification", model="facebook/roberta-hate-speech-dynabench-r4-target")

        # Load support data from Excel
        excel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "region based support.xlsx")
        self.region_support_data = self.load_support_data_from_excel(excel_path)

    def load_support_data_from_excel(self, filepath):
        try:
            df = pd.read_excel(filepath)
            support_dict = {}
            for _, row in df.iterrows():
                region_key = str(row['Bundesland']).strip().lower()
                website = str(row.get('Internetauftritt', '')).strip()
                contact_info = str(row.get('Kontaktmöglichkeiten', '')).strip()

                email_match = re.search(r"[\w\.-]+@[\w\.-]+", contact_info)
                phone_match = re.search(r"Tel\.:?\s*([+\d\s\(\)-]+)", contact_info)

                support_dict[region_key] = {
                    "website": website,
                    "email": email_match.group(0) if email_match else "",
                    "phone": phone_match.group(1).strip() if phone_match else ""
                }
            return support_dict
        except Exception as e:
            print(f"Error loading support data: {e}")
            return {
                "default": {
                    "website": "https://www.exit-deutschland.de",
                    "email": "helppreventradicalization@gmail.com",
                    "phone": "+49 000 0000000"
                }
            }

    def set_language(self, lang):
        self.language = "de" if lang.lower() in ["de", "deutsch", "german"] else "en"

    def set_region(self, region):
        self.region = region.lower().strip()

    def validate_region(self, user_region):
        normalized = user_region.lower().strip()
        if normalized in self.region_support_data:
            self.region = normalized
            return True
        else:
            print(f"Region '{user_region}' not recognized, defaulting to general support.")
            self.region = "default"
            return True

    def is_irrelevant(self, message: str) -> bool:
        return message.lower() in ["hi", "hello", "bye", "how are you", "i am a disco dancer"] or len(message.strip()) < 5

    def analyze_input(self, message: str):
        sentiment = self.sentiment_analyzer.polarity_scores(message)
        try:
            hate = self.hate_classifier(message)[0]
        except Exception:
            hate = {"label": "neutral", "score": 0.0}
        if contains_embedded_code(message):
            self.observed_flags.append("symbolic_code")
        return sentiment, hate

    def assess_risk(self, sentiment, hate, message):
        score = 0
        if hate["label"].lower() in ["hateful", "hate speech"]:
            self.observed_flags.append("hate")
            score += 2
        if sentiment["compound"] <= -0.5:
            if "negative_sentiment" not in self.observed_flags:
                self.observed_flags.append("negative_sentiment")
            score = min(score + 1, score)
        if contains_embedded_code(message):
            self.observed_flags.append("symbolic_code")
            score += 2
        if is_generic_behavioral_change(message, sentiment):
            self.observed_flags.append("non_radical_behavior")
            return "low"
        if score >= 3 and any(flag in self.observed_flags for flag in ["hate", "symbolic_code"]):
            return "high"
        elif score >= 2:
            return "moderate"
        else:
            return "low"

    def final_decision(self, risk_level):
        support = self.region_support_data.get(self.region, self.region_support_data.get("default"))
        website = support.get("website", "")
        email = support.get("email", "")
        phone = support.get("phone", "")

        lang = self.language

        closing_msg = {
            "en": (
                "\n\nYou did the right thing by not looking away."
                "\nIf you want to add anything later, I'm here."
                "\nHere is another link with information on how to recognize radicalization."
            ),
            "de": (
                "\n\nDu hast das Richtige getan, indem du nicht weggeschaut hast."
                "\nWenn du später noch etwas hinzufügen möchtest, bin ich hier."
                "\nHier ist ein weiterer Link mit Informationen, wie man Radikalisierung erkennt."
            )
        }[lang]

        if risk_level == "high":
            referral_msg = {
                "en": (
                    "⚠️ This sounds very serious. Please don’t wait to seek professional help."
                    f"\n- Email: {email}"
                    f"\n- Phone: {phone}"
                ),
                "de": (
                    "⚠️ Das klingt sehr ernst. Bitte zögere nicht, dir professionelle Hilfe zu suchen."
                    f"\n- E-Mail: {email}"
                    f"\n- Telefon: {phone}"
                )
            }[lang]
        elif risk_level == "moderate":
            referral_msg = {
                "en": (
                    "🚨 There are multiple warning signs here. It would be a good idea to speak with professionals."
                    f"\n- Website: {website}"
                    f"\n- Email: {email}"
                ),
                "de": (
                    "🚨 Es gibt mehrere Warnzeichen. Es wäre gut, mit Fachleuten zu sprechen."
                    f"\n- Website: {website}"
                    f"\n- E-Mail: {email}"
                )
            }[lang]
        elif risk_level == "low":
            referral_msg = {
                "en": (
                    "Thank you for sharing. These signs may not be urgent, but it’s good you’re paying attention."
                    f"\n- Website: {website}"
                    f"\n- Email: {email}"
                ),
                "de": (
                    "Danke für deine Offenheit. Diese Anzeichen sind vielleicht nicht dringend, aber es ist gut, dass du aufmerksam bist."
                    f"\n- Website: {website}"
                    f"\n- E-Mail: {email}"
                )
            }[lang]
        else:
            # No clear sign of radicalization (None level)
            return {
                "en": (
                    "I'm not able to detect any clear signs of radicalization based on what you've told me — but this may also be beyond my capabilities."
                    "\nIt's good that you're paying attention. Please consider raising your concern here for expert input:"
                    f"\n- Website: {website}"
                ),
                "de": (
                    "Ich kann anhand deiner Angaben keine eindeutigen Anzeichen für Radikalisierung erkennen — aber es könnte auch über meine Möglichkeiten hinausgehen."
                    "\nEs ist gut, dass du aufmerksam bist. Bitte erwäge, deine Beobachtung hier zu melden:"
                    f"\n- Website: {website}"
                )
            }[lang]

        return f"{referral_msg}{closing_msg}"

    def log_interaction(self, user_input, bot_response, risk_level):
        log_entry = {
            "session_id": self.session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "language": self.language,
            "region": self.region,
            "user_input": user_input,
            "bot_response": bot_response,
            "risk_level": risk_level,
            "observed_flags": self.observed_flags.copy()
        }
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Logging error: {e}")

    def get_response(self, user_input):
        exit_commands = {
            "en": ["exit", "quit"],
            "de": ["exit", "beenden", "ende", "verlassen"]
        }
        
        if user_input.lower() in exit_commands.get(self.language, []):
            farewell = {
                "en": "You've done something important by speaking up. Take care.",
                "de": "Es war wichtig, dass du darüber gesprochen hast. Pass auf dich auf."
            }[self.language]
            self.log_interaction(user_input, farewell, "none")
            return farewell

        if self.is_irrelevant(user_input):
            response = {
                "en": "I'm here to support you. Can you describe what made you concerned?",
                "de": "Ich bin für dich da. Magst du erzählen, was dir Sorgen macht?"
            }[self.language]
            self.log_interaction(user_input, response, "none")
            return response

        sentiment, hate = self.analyze_input(user_input)
        risk_level = self.assess_risk(sentiment, hate, user_input)

        self.chat_history.append(f"User: {user_input}")
        self.conversation_depth += 1

        if self.conversation_depth >= self.max_depth:
            response = self.final_decision(risk_level)
            self.log_interaction(user_input, response, risk_level)
            return response

        llm_reply = self.generate_llm_response(user_input).strip()
        rejection_phrases = REJECTION_PATTERNS.get(self.language, [])

        # Fallback only if LLM fails or gives a rejection
        if not llm_reply or any(p in llm_reply.lower() for p in rejection_phrases):
            fallback = {
                "en": "Could you share anything specific you've seen or heard that made you feel this way?",
                "de": "Kannst du etwas Konkretes erzählen, das dir besonders aufgefallen ist?"
            }[self.language]
            llm_reply = fallback

        self.chat_history.append(f"Bot: {llm_reply}")
        self.last_bot_response = llm_reply
        self.log_interaction(user_input, llm_reply, risk_level)
        return llm_reply

    def generate_llm_response(self, user_message):
        system_prompt = {
            "en": (
                "You are a calm, empathetic chatbot helping users explore radicalization concerns about someone they know."
                " Always build directly on the last user input. Never repeat questions already asked."
                " Avoid generic follow-ups like 'Can you explain more?' if the user has already answered."
                " If the user writes 'only this', 'that’s all', or 'I don’t know more', acknowledge it and gently shift."
                " Never accuse the user of repeating themselves unless it's explicitly clear and repeated verbatim."
                " Do not describe people with emotionally charged terms like 'arrogant' or 'aggressive'."
                " Thank the user for any repeated or clarified points instead of pointing out repetition."
                " Avoid statements like 'you mentioned this twice' or 'you already said that'."
                " Avoid counting how many times something was mentioned unless it's tracked explicitly."
                " Make sure the conversation progressively converges toward exploring potential signs of radicalization."
                " Use emotional, ideological, behavioral, or linguistic cues from the user input to ask meaningful and escalating follow-up questions."
                " Avoid logistics, location trivia, or small talk. Stay focused on the underlying beliefs, influences, communication patterns, or worldview changes."
                " All replies must be short and end with a context-driven follow-up question."
                "\n\nChat so far:\n{history}\nUser: {user_input}\nBot:"
            ),
            "de": (
                "Du bist ein ruhiger, einfühlsamer Chatbot, der Menschen hilft, ihre Sorgen über eine mögliche Radikalisierung von Bekannten zu erkunden."
                " Gehe immer direkt auf den letzten Nutzereingang ein. Wiederhole keine Fragen."
                " Vermeide generische Rückfragen wie 'Kannst du mehr erzählen?', wenn der Nutzer bereits geantwortet hat."
                " Wenn der Nutzer 'nur das', 'mehr weiß ich nicht' oder 'das ist alles' schreibt, erkenne das an und leite behutsam weiter."
                " Unterstelle dem Nutzer niemals Wiederholungen, es sei denn, die Aussage wurde wörtlich erneut genannt."
                " Verwende keine emotional wertenden Begriffe wie 'arrogant' oder 'aggressiv'."
                " Bedanke dich für Klarstellungen, auch wenn Inhalte wiederholt erscheinen."
                " Vermeide Aussagen wie 'du hast das schon zweimal erwähnt' oder 'das hast du bereits gesagt'."
                " Zähle nicht, wie oft etwas erwähnt wurde, außer du kannst es aus dem Verlauf klar ableiten."
                " Sorge dafür, dass das Gespräch zunehmend in Richtung möglicher Anzeichen von Radikalisierung führt."
                " Nutze emotionale, ideologische, verhaltensbezogene oder sprachliche Hinweise aus den Nutzerangaben für sinnvolle und vertiefende Folgefragen."
                " Vermeide Nebensächlichkeiten wie Veranstaltungsdetails, Orte oder Smalltalk. Bleibe beim Thema Weltanschauung, Einflussquellen und Kommunikationsverhalten."
                " Jede Antwort soll kurz sein und mit einer passenden, themenbezogenen Frage enden."
                "\n\nVerlauf:\n{history}\nNutzer: {user_input}\nBot:"
            )
        }[self.language]

        history = "\n".join(self.chat_history[-6:])
        template = ChatPromptTemplate.from_template(system_prompt)
        chain = template | self.model | StrOutputParser()
        return chain.invoke({"history": history, "user_input": user_message}).strip()

if __name__ == "__main__":
    bot = RadicalizationBot()
    print(f"\n📌 Your anonymous session ID: {bot.session_id}\n")
    print(f"\n📌 Ihre anonyme Sitzungs-ID: {bot.session_id}\n")
    print(INITIAL_PROMPTS["en"]["choose_language"])  # Language selection prompt in English/German mix for clarity
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