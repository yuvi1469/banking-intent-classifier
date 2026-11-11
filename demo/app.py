"""Gradio demo for the banking intent classifier.

The assistant predicts a customer's intent and decides how to respond:
it answers when confident and one intent is clearly ahead, asks a
clarifying question when the message is likely to be one of the top two
intents, and asks the customer to rephrase when even the top two together
are unlikely, as the message may be off-topic. Messages that may be about
a lost, stolen or misused card always get a safety warning.

Run locally with a trained model folder that contains demo_settings.json:

    python demo/app.py --model-dir path/to/distilbert
"""

import argparse
import json
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HIGH_RISK_INTENTS = {"lost_or_stolen_card", "compromised_card"}
SETTINGS_FILE = "demo_settings.json"
MAX_LENGTH = 128
TOP_INTENTS_SHOWN = 5

SAFETY_WARNING = (
    "**Important:** if your card is lost, stolen or being used without your permission, "
    "freeze it in the app straight away. I'm also passing this conversation to a member of the support team."
)
REPHRASE_REPLY = (
    "Sorry, I'm not sure I understood. I can help with questions about cards, payments, "
    "transfers, top-ups and your account. Could you rephrase your message?"
)


def readable(intent):
    return intent.replace("_", " ").capitalize()


class IntentAssistant:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
        self.labels = self.model.config.id2label
        settings = json.loads((model_dir / SETTINGS_FILE).read_text())
        self.off_topic_threshold = settings["off_topic_threshold"]
        self.clarify_threshold = settings["clarify_threshold"]

    def probabilities(self, message):
        inputs = self.tokenizer(
            message, truncation=True, max_length=MAX_LENGTH,
            return_token_type_ids=False, return_tensors="pt",
        )
        with torch.no_grad():
            return torch.softmax(self.model(**inputs).logits, dim=-1)[0].tolist()

    def decide(self, message):
        probabilities = self.probabilities(message)
        ranked = sorted(range(len(probabilities)), key=probabilities.__getitem__, reverse=True)
        first, second = ranked[0], ranked[1]

        confident = probabilities[first] >= self.off_topic_threshold
        clear_winner = probabilities[first] - probabilities[second] >= self.clarify_threshold
        if confident and clear_winner:
            action = "answer"
        elif probabilities[first] + probabilities[second] >= self.off_topic_threshold:
            action = "clarify"
        else:
            action = "rephrase"

        options = [self.labels[first], self.labels[second]]
        considered = options[:1] if action == "answer" else options
        return {
            "action": action,
            "intent": options[0],
            "options": options,
            "high_risk": any(intent in HIGH_RISK_INTENTS for intent in considered),
            "top": [(self.labels[i], probabilities[i]) for i in ranked[:TOP_INTENTS_SHOWN]],
        }


def format_reply(decision):
    if decision["action"] == "answer":
        reply = f"I can help with that. This looks like: **{readable(decision['intent'])}**."
    elif decision["action"] == "clarify":
        first, second = (readable(intent) for intent in decision["options"])
        reply = f"I want to make sure I help with the right thing. Is this about **{first}** or **{second}**?"
    else:
        reply = REPHRASE_REPLY
    if decision["high_risk"]:
        reply = f"{SAFETY_WARNING}\n\n{reply}"
    return reply


def build_interface(assistant, examples=()):
    def respond(message):
        if not message.strip():
            return "Please type a message.", {}
        decision = assistant.decide(message)
        return format_reply(decision), {readable(intent): p for intent, p in decision["top"]}

    with gr.Blocks(title="Banking intent assistant") as interface:
        gr.Markdown(
            "# Banking intent assistant\n"
            "Type a customer message to see how the assistant understands it. "
            "This is a course project demo, not a real banking service."
        )
        message = gr.Textbox(label="Customer message", placeholder="For example: My card hasn't arrived yet")
        send = gr.Button("Send", variant="primary")
        reply = gr.Markdown()
        confidence = gr.Label(label="Most likely intents", num_top_classes=TOP_INTENTS_SHOWN)
        send.click(respond, inputs=message, outputs=[reply, confidence])
        message.submit(respond, inputs=message, outputs=[reply, confidence])
        if examples:
            gr.Examples(examples=[[example] for example in examples], inputs=message)
    return interface


def main():
    parser = argparse.ArgumentParser(description="Run the banking intent assistant demo.")
    parser.add_argument("--model-dir", required=True, help="Folder with the trained model and demo_settings.json")
    parser.add_argument("--share", action="store_true", help="Create a temporary public link")
    args = parser.parse_args()
    build_interface(IntentAssistant(args.model_dir)).launch(share=args.share)


if __name__ == "__main__":
    main()
