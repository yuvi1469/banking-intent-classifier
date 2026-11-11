# Banking Intent Classifier

An intent classifier for banking chatbots. It takes a customer message, such as
"My card hasn't arrived yet", and predicts which of 77 banking intents it belongs to.
**BankBot Assistant**, the chat app built on it, answers when confident, asks when
unsure, and warns customers about lost or misused cards.

[Open the notebook in Colab](https://colab.research.google.com/github/yuvi1469/banking-intent-classifier/blob/main/notebooks/banking_intent_classifier.ipynb) · [Read the report](docs/report.md) · [Model card](docs/model_card.md)

## Results

Test set of 3,080 messages from BANKING77.

| Model | Accuracy | Precision | Recall | Macro F1 | Median CPU time | Size |
|---|---|---|---|---|---|---|
| TF-IDF baseline | 0.897 | 0.902 | 0.897 | 0.898 | 1.1 ms | 1.4 MB |
| **DistilBERT (chosen)** | **0.929** | **0.932** | **0.929** | **0.929** | **57 ms** | **269 MB** |
| RoBERTa | 0.931 | 0.934 | 0.931 | 0.931 | 109 ms | 502 MB |

- Both transformers beat the baseline in all 1,000 bootstrap resamples.
- RoBERTa and DistilBERT are effectively tied, so DistilBERT was chosen for being faster and smaller.
- Asking a clarifying question on about 10% of messages cuts DistilBERT's wrong answers from 219 to 151.
- With BankBot's full response rules, wrong answers fall from 7.1% to 2.8% of banking messages.

The project also tests off-topic detection, typos, speed and limited training data. See the [report](docs/report.md) for these results and the limitations.

## Project structure

| Path | Contents |
|---|---|
| `notebooks/banking_intent_classifier.ipynb` | The full project, from problem definition to the app |
| `app/bankbot.py` | BankBot Assistant, the Gradio chat app |
| `docs/report.md` | Project report |
| `docs/model_card.md` | Model card |
| `docs/demo_script.md` | Presentation plan and video checklist |
| `results/figures/` | Key charts |
| `requirements.txt` | Pinned package versions |

## How to run

### The notebook

1. Open the notebook using the Colab link above.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Select **Runtime > Run all**, and allow access to Google Drive when asked.
4. When section 12.4 prints a `gradio.live` link, open it to use BankBot.

The first run trains all the models and saves them to `MyDrive/banking-intent-classifier`, which needs a GPU and takes about an hour. Later runs load the saved models, as `RETRAIN = False`, and also work on a CPU, more slowly. Set `RETRAIN = True` to train again.

### BankBot on your own computer

This needs a trained model folder from Google Drive. Section 12.1 of the notebook adds the `demo_settings.json` file that the app needs.

```bash
pip install -r requirements.txt
```

```bash
python app/bankbot.py --model-dir path/to/distilbert
```

## Data and credits

- **BANKING77:** Casanueva et al. (2020), *Efficient Intent Detection with Dual Sentence Encoders*. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Two label names were corrected, and near-duplicates were removed from the training data.
- **CLINC150:** Larson et al. (2019), *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*. Licensed under [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). Its out-of-scope messages are used for off-topic evaluation, with messages containing banking words removed.
- **Models:** [DistilBERT](https://huggingface.co/distilbert-base-uncased) (Sanh et al., 2019) and [RoBERTa](https://huggingface.co/roberta-base) (Liu et al., 2019), using Hugging Face Transformers.

This is a course project, not a real banking service.
