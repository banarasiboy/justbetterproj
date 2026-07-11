"""
CodSoft ML Internship - Task 1
Chatbot with Rule-Based Responses

The idea here is pretty simple: the bot doesn't "understand" language the
way a real NLP model would. Instead, it looks for keywords or patterns in
whatever the user types and matches them against a set of predefined rules.
It's a good first step before moving into anything statistical or deep
learning based, because it forces you to think about how conversation is
actually structured (greetings, questions, small talk, goodbyes, etc.)
"""

import re
import random
from datetime import datetime


class RuleBasedChatbot:
    def __init__(self, bot_name="CodBot"):
        self.name = bot_name
        self.user_name = None

        # Each rule is a (regex pattern, list of possible responses) pair.
        # Using a list of responses instead of a single string so the bot
        # doesn't sound exactly the same every time -> feels less robotic.
        self.rules = [
            (r"\b(hi|hello|hey|hola)\b",
             ["Hey there! How can I help you today?",
              "Hello! What's on your mind?",
              "Hi! Good to see you."]),

            (r"\bmy name is (\w+)|i am (\w+)|i'm (\w+)\b",
             ["Nice to meet you, {name}!",
              "Got it, I'll remember you, {name}."]),

            (r"\bhow are you\b",
             ["I'm just a program, but I'm running smoothly! How about you?",
              "Doing great, thanks for asking!"]),

            (r"\b(what is your name|who are you)\b",
             [f"I'm {self.name}, a simple rule-based chatbot.",
              f"You can call me {self.name}."]),

            (r"\b(time)\b",
             ["It's currently {time}."]),

            (r"\b(date|today)\b",
             ["Today's date is {date}."]),

            (r"\b(thank you|thanks)\b",
             ["You're welcome!", "No problem at all!", "Anytime!"]),

            (r"\b(help|what can you do)\b",
             ["I can chat about basic things - greetings, the time, "
              "the date, or just have a simple conversation. "
              "Try asking me something!"]),

            (r"\b(bye|goodbye|exit|quit)\b",
             ["Goodbye! Have a great day.", "See you later!", "Bye bye!"]),

            (r"\b(joke)\b",
             ["Why do programmers prefer dark mode? Because light attracts bugs!",
              "I would tell you a UDP joke, but you might not get it."]),
        ]

        # Anything that doesn't match a rule falls back to one of these.
        self.fallback_responses = [
            "I'm not sure I understand. Could you rephrase that?",
            "Hmm, I don't have a rule for that one yet.",
            "Sorry, can you say that differently?",
        ]

    def _apply_placeholders(self, response):
        response = response.replace("{time}", datetime.now().strftime("%I:%M %p"))
        response = response.replace("{date}", datetime.now().strftime("%B %d, %Y"))
        return response

    def get_response(self, user_input):
        text = user_input.lower().strip()

        for pattern, responses in self.rules:
            match = re.search(pattern, text)
            if match:
                response = random.choice(responses)

                # Special case: capture the user's name if they introduced themselves
                if "{name}" in response:
                    name = next((g for g in match.groups() if g), None)
                    if name:
                        self.user_name = name.capitalize()
                        response = response.replace("{name}", self.user_name)

                return self._apply_placeholders(response)

        return random.choice(self.fallback_responses)

    def is_exit(self, user_input):
        return bool(re.search(r"\b(bye|goodbye|exit|quit)\b", user_input.lower()))


def run_chat():
    bot = RuleBasedChatbot()
    print(f"{bot.name}: Hi! I'm {bot.name}. Type 'bye' whenever you want to end the chat.")

    while True:
        user_input = input("You: ")
        if not user_input.strip():
            continue

        reply = bot.get_response(user_input)
        print(f"{bot.name}: {reply}")

        if bot.is_exit(user_input):
            break


if __name__ == "__main__":
    run_chat()
