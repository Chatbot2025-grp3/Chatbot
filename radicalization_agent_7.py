
# radicalization_terms_complete.py
RADICALIZATION_PHRASES = set([
    # Hate speech and racist phrases
    "contaminate", "evil", "dark skin", "race war", "white genocide", "88", "14 words",
    "they must be removed", "our blood is pure", "traitors to the people",
    "cleansing the streets", "globalist agenda", "ZOG", "replacement theory",
    # Codes and symbols
    "18", "88", "666", "black sun", "reich flag", "totenkopf", "hh", "heil hitler",
    "5g causes cancer", "climate lie", "plandemic", "nwo", "deepstate",
    # Extremist slogans
    "no taxes for the system", "stop the great replacement", "deportation now",
    "gates virus", "resistance now", "mainstream lies", "groomer agenda",
    # Violence encouragement or ideology
    "violence is the only answer", "reclaim our land", "fight the enemy within",
    "cleanse our streets", "ban islam", "fight multiculturalism",
    # Antisemitic and conspiracy
    "freemasons rule us", "jewish agenda", "zionist puppet", "world bank scam",
    "they control the media", "global control through fear", "soros control"
])



REJECTION_PATTERNS = [
    "i cannot provide", 
    "i'm not able to help", 
    "please contact emergency services", 
    "i can't assist with that", 
    "this content violates", 
    "i'm not allowed to", 
    "my apologies, but i cannot"
]

def sanitize_response(response: str, fallback: str) -> str:
    lower = response.lower()
    if any(p in lower for p in REJECTION_PATTERNS):
        return fallback
    return response.strip()


import os
import re
from typing import List, Dict
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from transformers import pipeline
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

load_dotenv()
nltk.download("vader_lexicon")

API_KEY = os.getenv("CHAT_AI_ACCESS_KEY")

PROMPTS = {
    "en": "You are a calm, non-judgmental chatbot based in Germany. Help users share concerns about troubling behavior changes. Respond empathetically in English.",
    "de": "Du bist ein ruhiger, nicht wertender Chatbot aus Deutschland. Unterstütze Nutzer beim Beschreiben besorgniserregender Verhaltensänderungen. Antworte einfühlsam auf Deutsch."
}

def extract_first_question(text):
    lines = re.split(r'[\n\r]+', text.strip())
    for line in lines:
        if '?' in line:
            return line.strip()
    return lines[0].strip() if lines else ""

def is_valid_response(text):
    lower = text.lower()
    if "how can i assist" in lower or "is there anything else" in lower:
        return False
    if '?' not in text:
        return False
    if len(text.strip()) > 300:
        return False
    return True

class RadicalizationAgent:
    def __init__(self):
        self.language = "en"
        self.region = "default"
        self.chat_history: List[str] = []
        self.conversation_depth = 0
        self.asked_topics: List[str] = []
        self.current_relationship = "this person"
        self.relationship_set = False
        self.has_concerning_content = False
        self.llm = ChatOpenAI(
            model="meta-llama-3.1-8b-instruct",
            temperature=0.5,
            openai_api_key=API_KEY,
            openai_api_base="https://chat-ai.academiccloud.de/v1",
        )
        self.classifier = pipeline("text-classification", model="facebook/roberta-hate-speech-dynabench-r4-target")
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.min_conversation_depth = 6
        self.max_silence_threshold = 2

    def is_relevant_input(self, message: str) -> bool:
        message_lower = message.lower()
        if any(term in message_lower for term in RADICALIZATION_PHRASES):
            return True
        score = self.sentiment_analyzer.polarity_scores(message)
        result = self.classifier(message)[0]
        return result['label'].lower() == 'hateful' or score['compound'] <= -0.5

    def set_language(self, lang: str):
        if lang.lower() in ["de", "german", "deutsch"]:
            self.language = "de"
        else:
            self.language = "en"

    def set_region(self, region: str):
        self.region = region.strip().lower()

    def get_support_links(self) -> List[str]:
        region_links = {
            "berlin": ["https://mbr-berlin.de", "info@mbr-berlin.de"],
            "bremen": ["https://sichtwechsel-bremen.de", "kontakt@sichtwechsel-bremen.de"],
            "nrw": ["https://www.nrwgegendradikal.de"],
            "bundesweit": ["https://www.exit-deutschland.de", "helppreventradicalization@gmail.com"]
        }
        return region_links.get(self.region, region_links["bundesweit"])

    def _detect_relationship(self, text: str) -> str:
        relationships = {
            "en": ["friend", "brother", "sister", "father", "mother", "partner", "colleague", "uncle", "neighbor"],
            "de": ["freund", "bruder", "schwester", "vater", "mutter", "partner", "kollege", "onkel", "nachbar"]
        }
        text = text.lower()
        for term in relationships[self.language]:
            if term in text:
                self.current_relationship = term
                self.relationship_set = True
                return term
        return self.current_relationship

    def follow_up_brain(self, user_input: str) -> str:
        follow_up_bank = [
            ("violence", "What specific actions or statements have you observed?"),
            ("hate_speech", "Has the person used derogatory terms or hate speech?"),
            ("group_involvement", "Has there been any involvement with new groups or communities?"),
            ("behavior_change", "How has this behavior affected their daily life or relationships?"),
            ("planning", "Have there been any signs of preparation for violence or self-harm?"),
            ("language_shift", "Have you noticed a change in their language or how they talk about others?"),
            ("online_activity", "Are they spending more time online or visiting questionable forums?"),
            ("appearance", "Have they changed their appearance, dress, or symbols they use?")
        ]
        user_input_lower = user_input.lower()
        self._detect_relationship(user_input)

        for key, question in follow_up_bank:
            if key not in self.asked_topics and not any(q in user_input_lower for q in question.lower().split()):
                self.asked_topics.append(key)
                return f"{question} ({self.current_relationship})"
        return f"Can you share anything else about your {self.current_relationship}'s behavior or views?"

    def get_response(self, user_message: str) -> str:
        relevant = self.is_relevant_input(user_message)
        if relevant:
            self.has_concerning_content = True

        if not relevant and not self.has_concerning_content:
            return "Thank you for sharing. I didn’t notice any concerning signs, but feel free to tell me more if something is worrying you."

        self.chat_history.append(f"User: {user_message}")
        follow_up = self.follow_up_brain(user_message)
        self.conversation_depth += 1
        prompt_text = f"{PROMPTS[self.language]} Ask: '{follow_up}'"
        prompt = PromptTemplate.from_template(prompt_text)
        chain = prompt | self.llm | StrOutputParser()
        bot_response = chain.invoke({"user_message": user_message})
        bot_response = extract_first_question(bot_response)

        if not is_valid_response(bot_response):
            bot_response = follow_up

        final_response = sanitize_response(bot_response, fallback=follow_up)
        self.chat_history.append(f"Bot: {final_response}")
        return final_response

if __name__ == "__main__":
    agent = RadicalizationAgent()
    print("🌍 Choose language (English/Deutsch):")
    while True:
        lang = input("Language: ").strip().lower()
        if lang in ["en", "de"]:
            agent.set_language(lang)
            break
        print("❌ Invalid input. Please choose 'en' for English or 'de' for Deutsch.")

    print("📍 What's your region? (type it exactly):")
    region = input("Region: ").strip().lower()
    agent.set_region(region)

    print("\n💬 Describe your concern (type 'exit' to quit):")
    while True:
        msg = input("\nYou: ").strip()
        if msg.lower() in ["exit", "quit"]:
            print("Bot: You’ve done something important by speaking up. Take care.")
            break
        print("Bot:", agent.get_response(msg))
