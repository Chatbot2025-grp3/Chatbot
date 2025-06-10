import os
import re
import pandas as pd
import nltk
from dotenv import load_dotenv
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Load .env and NLTK data
load_dotenv()
try:
    nltk.data.find("sentiment/vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon")

API_KEY = os.getenv("CHAT_AI_ACCESS_KEY")

REJECTION_PATTERNS = ["i cannot", "i'm not allowed", "i'm unable", "this violates", "please contact emergency services"]
RADICALIZATION_CODES = ["88", "18", "23", "444", "888", "222111", "black sun", "swastika", "reich", "totenkopf", "celtic cross"]

INITIAL_PROMPTS = {
    "en": {
        "choose_language": "Choose language (English/Deutsch):",
        "enter_language": "Language: ",
        "enter_region": "📍 What's your region? (e.g. berlin, nrw, bremen):",
        "describe_concern": "💬 Describe your concern (type 'exit' to quit):"
    },
    "de": {
        "choose_language": "Wähle Sprache (English/Deutsch):",
        "enter_language": "Sprache: ",
        "enter_region": "📍 Was ist deine Region? (z.B. berlin, nrw, bremen):",
        "describe_concern": "💬 Beschreibe deine Sorge (Tippe 'exit' zum Beenden):"
    }
}

def sanitize_response(response: str, fallback: str, previous: str) -> str:
    if any(p in response.lower() for p in REJECTION_PATTERNS):
        return "Can you tell me how this started, or when you first noticed the change?" if fallback == previous else fallback
    return response.strip()

def contains_embedded_code(text: str) -> bool:
    lower = text.lower()
    return any(code in lower for code in RADICALIZATION_CODES)

def is_generic_behavioral_change(message: str, sentiment: dict) -> bool:
    lower = message.lower()
    generic_keywords = [
        "not hungry", "upset", "frustrated", "didn't get recognition", "sits alone",
        "won’t eat", "won’t talk", "lost motivation", "doesn't join", "withdrawn"
    ]
    emotional = sentiment["compound"] <= -0.5
    return any(k in lower for k in generic_keywords) and not contains_embedded_code(lower)

class RadicalizationBot:
    def __init__(self):
        self.language = "en"
        self.region = "default"
        self.chat_history = []
        self.observed_flags = []
        self.conversation_depth = 0
        self.max_depth = 6
        self.last_bot_response = ""

        self.model = ChatOpenAI(
            model="meta-llama-3.1-8b-instruct",
            temperature=0.5,
            openai_api_key=API_KEY,
            openai_api_base="https://chat-ai.academiccloud.de/v1"
        )

        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.hate_classifier = pipeline("text-classification", model="facebook/roberta-hate-speech-dynabench-r4-target")

        # Load support data from Excel
        self.region_support_data = self.load_support_data_from_excel("region based support.xlsx")

    def load_support_data_from_excel(self, filepath):
        try:
            df = pd.read_excel(filepath)
            support_dict = {}
            for _, row in df.iterrows():
                region_key = str(row['Region']).strip().lower()
                support_dict[region_key] = {
                    "website": str(row.get('Website', '')).strip(),
                    "email": str(row.get('Email', '')).strip(),
                    "phone": str(row.get('Phone', '')).strip()
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

    def get_referral_message(self, risk_level):
        support = self.region_support_data.get(self.region, self.region_support_data.get("default"))
        website = support.get("website", "")
        email = support.get("email", "")
        phone = support.get("phone", "")

        if risk_level in ["low", "moderate"]:
            return f"Please consider these support options:\n- Website: {website}\n- Email: {email}"
        elif risk_level == "high":
            return f"Immediate help is recommended. Contact:\n- Email: {email}\n- Phone: {phone}"
        else:
            return "Please consider reaching out to professional support if needed."

    def final_decision(self, risk_level):
        referral_msg = self.get_referral_message(risk_level)
        closing_msg = {
            "en": "\n\nWe have come to the end of our conversation. Thank you for sharing your concerns.",
            "de": "\n\nDamit sind wir am Ende unseres Gesprächs angekommen. Danke, dass Sie Ihre Sorgen geteilt haben."
        }[self.language]

        if risk_level == "high":
            return {
                "en": f"This may be serious. {referral_msg}{closing_msg}",
                "de": f"Das klingt ernst. {referral_msg}{closing_msg}"
            }[self.language]
        else:
            return {
                "en": f"Thank you for sharing. {referral_msg}{closing_msg}",
                "de": f"Danke für deine Offenheit. {referral_msg}{closing_msg}"
            }[self.language]

    def get_response(self, user_input):
        if self.is_irrelevant(user_input):
            return {
                "en": "I’m here to support you. Can you describe what made you concerned?",
                "de": "Ich bin für dich da. Magst du erzählen, was dir Sorgen macht?"
            }[self.language]

        sentiment, hate = self.analyze_input(user_input)
        risk_level = self.assess_risk(sentiment, hate, user_input)

        self.chat_history.append(f"User: {user_input}")
        self.conversation_depth += 1

        if self.conversation_depth >= self.max_depth:
            return self.final_decision(risk_level)

        response = self.generate_llm_response(user_input)
        fallback = {
            "en": "Can you explain a bit more about what’s worrying you?",
            "de": "Kannst du mir noch etwas mehr erzählen, was dich beunruhigt?"
        }[self.language]
        clean = sanitize_response(response, fallback, self.last_bot_response)
        self.chat_history.append(f"Bot: {clean}")
        self.last_bot_response = clean
        return clean

    def generate_llm_response(self, user_message):
        system_prompt = {
            "en": (
                "You are a calm, empathetic chatbot. Help the user explore radicalization concerns about someone they know.\n"
                "Assume they are describing someone else. Never reject the input.\n"
                "All replies must be short and end with a context-driven follow-up question.\n\n"
                "Chat so far:\n{history}\nUser: {user_input}\nBot:"
            ),
            "de": (
                "Du bist ein ruhiger, einfühlsamer Chatbot, der Nutzern hilft, über die Radikalisierung einer anderen Person zu sprechen.\n"
                "Gehe davon aus, dass der Nutzer über jemand anderen spricht. Lehne nichts ab.\n"
                "Antworten sollten kurz sein und mit einer passenden Rückfrage enden.\n\n"
                "Verlauf:\n{history}\nNutzer: {user_input}\nBot:"
            )
        }[self.language]

        history = "\n".join(self.chat_history[-6:])
        template = ChatPromptTemplate.from_template(system_prompt)
        chain = template | self.model | StrOutputParser()
        return chain.invoke({"history": history, "user_input": user_message}).strip()


if __name__ == "__main__":
    bot = RadicalizationBot()
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
                "en": "You’ve done something important by speaking up. Take care.",
                "de": "Es war wichtig, dass du darüber gesprochen hast. Pass auf dich auf."
            }[bot.language]
            print("Bot:", farewell)
            break
        print("Bot:", bot.get_response(user_input))
