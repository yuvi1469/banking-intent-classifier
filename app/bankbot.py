"""BankBot Assistant: a chat app built on the banking intent classifier.

For each customer message, BankBot predicts the intent and decides how to respond:
it answers when it is confident and one intent is clearly ahead, asks a
clarifying question when the message is likely to be one of the top two
intents, and asks the customer to rephrase when even the top two together are
unlikely, as the message may be off-topic. Messages that may be about a lost,
stolen or misused card always get a safety warning.

After a clarifying question, the customer can reply with the option they mean
("the first one", "2", or the option's name) and BankBot continues from there.

Run locally with a trained model folder that contains demo_settings.json:

    python app/bankbot.py --model-dir path/to/distilbert
"""

import argparse
import json
import re
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

APP_NAME = "BankBot Assistant"
HIGH_RISK_INTENTS = {"lost_or_stolen_card", "compromised_card"}
SETTINGS_FILE = "demo_settings.json"
MAX_LENGTH = 128
TOP_INTENTS_SHOWN = 5

WELCOME = (
    "Hi, I'm BankBot. I can help with questions about your cards, payments, transfers, "
    "top-ups and your account. What can I help you with today?"
)
SAFETY_WARNING = (
    "**Important:** if your card is lost, stolen or being used without your permission, "
    "freeze it in the app straight away. I'm also passing this conversation to a member of the support team."
)
REPHRASE_REPLY = (
    "Sorry, I'm not sure I understood. I can help with questions about cards, payments, "
    "transfers, top-ups and your account. Could you rephrase your message?"
)
RESPONSE_NAMES = {
    "answer": "Answered",
    "clarify": "Asked a clarifying question",
    "rephrase": "Asked to rephrase",
    "chosen": "Answered after the customer chose an option",
}
FIRST_OPTION = {"1", "one", "first", "the first", "first one", "the first one", "a", "option 1", "option a"}
SECOND_OPTION = {"2", "two", "second", "the second", "second one", "the second one", "b", "option 2", "option b"}

CSS = """
.gradio-container {max-width: 1100px !important; margin: 0 auto;}
#bankbot-header h1 {margin-bottom: 0;}
#bankbot-header p {color: var(--body-text-color-subdued); margin-top: 4px;}
"""


def readable(intent):
    return intent.replace("_", " ").capitalize()


def simplify(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


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
    if decision["action"] in ("answer", "chosen"):
        reply = f"Thanks. I understand this is about **{readable(decision['intent'])}**. I'll take you to the right help for that."
    elif decision["action"] == "clarify":
        first, second = (readable(intent) for intent in decision["options"])
        reply = (
            "I want to make sure I help with the right thing. Is this about:\n\n"
            f"1. **{first}**\n2. **{second}**\n\nReply with 1 or 2, or tell me more."
        )
    else:
        reply = REPHRASE_REPLY
    if decision["high_risk"]:
        reply = f"{SAFETY_WARNING}\n\n{reply}"
    return reply


def chosen_option(message, options):
    """Returns the option the customer picked after a clarifying question, or None."""
    text = simplify(message)
    if text in FIRST_OPTION:
        return options[0]
    if text in SECOND_OPTION:
        return options[1]
    named = [option for option in options if simplify(readable(option)) in text]
    return named[0] if len(named) == 1 else None


def chat_turn(assistant, message, history, pending_options):
    """Handles one customer message. Returns the new history, the pending options and the decision."""
    choice = chosen_option(message, pending_options) if pending_options else None
    if choice:
        decision = {
            "action": "chosen", "intent": choice, "options": pending_options,
            "high_risk": choice in HIGH_RISK_INTENTS, "top": None,
        }
    else:
        decision = assistant.decide(message)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": format_reply(decision)},
    ]
    pending_options = decision["options"] if decision["action"] == "clarify" else None
    return history, pending_options, decision


def build_interface(assistant, examples=()):
    def respond(message, history, pending_options, last_top):
        if not message.strip():
            return "", history, pending_options, last_top, gr.skip(), gr.skip()
        history, pending_options, decision = chat_turn(assistant, message, history, pending_options)
        top = {readable(intent): p for intent, p in decision["top"]} if decision["top"] else last_top
        status = f"**Response:** {RESPONSE_NAMES[decision['action']]}"
        if decision["high_risk"]:
            status += "  \n**Safety warning shown**"
        return "", history, pending_options, top, top, status

    def new_chat():
        return [{"role": "assistant", "content": WELCOME}], None, {}, {}, ""

    with gr.Blocks(title=APP_NAME) as interface:
        pending_options = gr.State(None)
        last_top = gr.State({})
        gr.Markdown(
            f"# {APP_NAME}\nYour banking help, understood. "
            "A course project on intent classification, not a real banking service.",
            elem_id="bankbot-header",
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                chat = gr.Chatbot(
                    value=[{"role": "assistant", "content": WELCOME}],
                    height=480, show_label=False, layout="bubble",
                )
                with gr.Row():
                    message = gr.Textbox(
                        placeholder="Type your message, for example: My card hasn't arrived yet",
                        show_label=False, scale=5, autofocus=True, submit_btn=False, lines=1, max_lines=4,
                    )
                    send = gr.Button("Send", variant="primary", scale=1, min_width=90)
                clear = gr.Button("New conversation", variant="secondary", size="sm")
            with gr.Column(scale=2):
                confidence = gr.Label(label="What BankBot understood", num_top_classes=TOP_INTENTS_SHOWN)
                status = gr.Markdown()
                if examples:
                    gr.Examples(examples=[[example] for example in examples], inputs=message, label="Try a message")

        outputs = [message, chat, pending_options, last_top, confidence, status]
        inputs = [message, chat, pending_options, last_top]
        send.click(respond, inputs=inputs, outputs=outputs)
        message.submit(respond, inputs=inputs, outputs=outputs)
        clear.click(new_chat, outputs=[chat, pending_options, last_top, confidence, status])
    return interface


def launch(assistant, examples=(), **launch_options):
    """Builds and starts the app with BankBot's theme."""
    interface = build_interface(assistant, examples)
    interface.launch(theme=gr.themes.Soft(primary_hue="blue"), css=CSS, **launch_options)
    return interface


def main():
    parser = argparse.ArgumentParser(description=f"Run {APP_NAME}.")
    parser.add_argument("--model-dir", required=True, help="Folder with the trained model and demo_settings.json")
    parser.add_argument("--share", action="store_true", help="Create a temporary public link")
    args = parser.parse_args()
    launch(IntentAssistant(args.model_dir), share=args.share)


if __name__ == "__main__":
    main()
