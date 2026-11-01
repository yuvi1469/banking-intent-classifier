# Banking Intent Classifier

An intent classifier for banking chatbots. It takes a customer message, such as
"My card hasn't arrived yet", and predicts which of 77 banking intents it belongs to.

**Status:** in progress. Results will be added here once the models are evaluated.

## Approach

- **Dataset:** BANKING77
- **Baseline:** TF-IDF with logistic regression
- **Models:** DistilBERT and RoBERTa
- **Metrics:** accuracy, precision, recall and macro F1
- **Demo:** Gradio app

## Project structure

| Folder | Contents |
|---|---|
| `notebooks/` | Main notebook |
| `src/` | Reusable code |
| `demo/` | Gradio app |
| `results/` | Charts and results tables |
| `docs/` | Report, model card and demo notes |

## How to run

1. Open the notebook from `notebooks/` in Google Colab.
2. Select a GPU runtime.
3. Run all cells from top to bottom.

Full instructions will be added when the notebook is complete.
