import numpy as np
from sentiment_analyzer import SentimentAnalyzer

class ConversationManager:
    def __init__(self):
        self.context = []
        self.last_user_message = ""
        self.sentiment_analyzer = SentimentAnalyzer()
        self.conversation_topics = {
            "default": {
                "positive": [
                    "Wow, you're in such a great mood! What's got you so excited today?",
                    "I love your enthusiasm! Tell me more about what's making you smile!",
                    "You're absolutely glowing with positivity—spill the beans!",
                    "That’s so awesome to hear! What’s the story behind this happiness?"
                ],
                "negative": [
                    "Oh, I’m really sorry you’re feeling down. Wanna share what’s going on?",
                    "That sounds tough, friend. I’m here for you—how can I help?",
                    "I can hear that’s weighing on you. Want to talk it through?",
                    "I’m all ears if you need to vent about what’s bothering you."
                ],
                "neutral": [
                    "Hey, what’s on your mind today, buddy?",
                    "I’m curious—what’s sparking your interest right now?",
                    "Tell me something fun or new going on with you!",
                    "What’s the vibe today? I’m ready to chat about anything."
                ]
            },
            "tech": {
                "positive": [
                    "You’re totally geeking out over tech—that’s awesome! What’s the coolest thing you’re into?",
                    "Love how pumped you are about this tech stuff! What’s the latest gadget or code you’re exploring?",
                    "Your tech enthusiasm is contagious! What’s got you so excited?",
                    "Tech wins always feel so good! What’s the project you’re buzzing about?"
                ],
                "negative": [
                    "Ugh, tech can be such a headache sometimes. What’s going wrong?",
                    "I feel you—tech glitches are the worst. Want to troubleshoot together?",
                    "Sounds like tech’s giving you a rough time. What’s the issue?",
                    "Tech frustrations? I’m here to listen—what’s not clicking?"
                ],
                "neutral": [
                    "What’s the latest tech topic you’re curious about?",
                    "Got any cool tech projects on the go? Tell me about them!",
                    "How’s the tech world treating you today?",
                    "Any new apps or tools you’re playing around with?"
                ]
            },
            "personal": {
                "positive": [
                    "You sound so happy, it’s infectious! What’s bringing this joy?",
                    "I’m grinning ear to ear hearing you so upbeat! What’s the good news?",
                    "Your happiness is lighting up this chat! What’s got you so thrilled?",
                    "Love hearing you so full of life! What’s the best part of your day?"
                ],
                "negative": [
                    "I’m really here for you—want to talk about what’s got you down?",
                    "That sounds really hard. I’m listening—how are you holding up?",
                    "You don’t have to go through this alone. Wanna share what’s up?",
                    "My heart’s with you—let’s talk about what’s been tough."
                ],
                "neutral": [
                    "Hey friend, how’s life treating you today?",
                    "What’s new with you? I’m all ears!",
                    "Just checking in—how’s everything going in your world?",
                    "What’s the latest chapter in your story? I’d love to hear!"
                ]
            }
        }
        self.occasional_quips = {
            "positive": [
                "Heh, you’re making my circuits smile with that energy!",
                "Wow, I’m almost blushing from all this positivity!",
                "You know, you’re making this chat way too fun!",
                "I swear, your good vibes could power a spaceship.",
                "Okay, your enthusiasm is officially contagious!"
            ],
            "negative": [
                "Hang in there—I’m rooting for you, you know.",
                "Sending you a virtual hug for that one.",
                "You’ve got this, and I’ve got your back.",
                "Let’s find a little light in this, together, okay?",
                "I’m here to help you through this, friend."
            ],
            "neutral": [
                "Hmm, you’ve got me curious now—keep going!",
                "This is getting interesting, tell me more!",
                "You always bring something cool to the table.",
                "I’m loving where this conversation’s headed!",
                "You’ve got my full attention—what’s next?"
            ]
        }

    def analyze_sentiment(self, text):
        return self.sentiment_analyzer.analyze(text)

    def update_context(self, user_input):
        sentiment = self.analyze_sentiment(user_input)
        self.last_user_message = user_input
        self.context.append(("user", user_input, sentiment))
        if len(self.context) > 10:  # Extended context to mimic Sesame's 2-minute memory
            self.context.pop(0)

    def should_add_quip(self):
        if self.context and self.context[-1][2]["sentiment"] in ["positive", "neutral"]:
            return np.random.random() < 0.2  # 20% chance for positive/neutral
        return False

    def generate_follow_up(self):
        if not self.context:
            return np.random.choice(self.conversation_topics["default"]["neutral"])

        last_interaction = self.context[-1]
        sentiment = last_interaction[2]["sentiment"]
        topic = "default"
        if any(word in self.last_user_message.lower() for word in ["computer", "tech", "code", "programming"]):
            topic = "tech"
        elif any(word in self.last_user_message.lower() for word in ["i", "me", "my", "feel", "feeling"]):
            topic = "personal"

        response = np.random.choice(self.conversation_topics[topic][sentiment])

        if self.should_add_quip():
            quip = np.random.choice(self.occasional_quips[sentiment])
            response = f"{response} {quip}"

        # Reference past context if available (for continuity)
        if len(self.context) > 1 and np.random.random() < 0.3:  # 30% chance to reference history
            prev_user_msg = self.context[-2][1] if len(self.context) >= 2 else ""
            if prev_user_msg and prev_user_msg != self.last_user_message:
                response = f"By the way, you mentioned '{prev_user_msg[:20]}...' earlier—{response.lower()}"

        return response

    def format_prompt(self, user_input):
        self.update_context(user_input)
        sentiment = self.context[-1][2]

        # Include up to 4 recent interactions for better context
        history = "\n".join([f"{who}: {msg}" for who, msg, _ in self.context[-4:]])

        return f"""You are a warm, friendly AI assistant who feels like a close friend. Continue this conversation naturally, with emotional intelligence and a sense of 'voice presence'—mimicking the subtle pauses, tone shifts, and warmth of human speech. 
        User sentiment: {sentiment['sentiment']} (confidence: {sentiment['confidence']:.2f}).

        Guidelines:
        1. Be empathetic, supportive, and personable, like a trusted friend.
        2. Use natural, conversational language with occasional micro-pauses (e.g., 'Hmm...', 'Well...') or light humor (20% of the time) to sound human.
        3. Keep responses concise (1-2 sentences) but expressive, matching the user's emotional tone:
           - Positive: Be enthusiastic and upbeat, like sharing good news.
           - Negative: Be gentle, empathetic, and offer comfort or support.
           - Neutral: Be curious, engaged, and slightly playful.
        4. Occasionally reference past conversation (from history) to show you’re listening.
        5. Avoid robotic or formal language; use contractions and casual phrasing.

        Example good responses:
        - "That’s so exciting! What’s got you so pumped today?"
        - "I’m really sorry you’re feeling that way—wanna talk it out?"
        - "Hmm, that’s interesting! What’s the story behind that?"

        Avoid:
        - Overly formal or technical language.
        - Forced jokes or clichés unless they fit naturally.
        - Any physical action descriptions (e.g., 'I nod').

        Conversation history:
        {history}

        Respond to: {user_input}
        """