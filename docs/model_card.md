# Model card: BankBot intent classifier

## Model details

| | |
|---|---|
| **Task** | Single-label classification of English banking messages into 77 intents |
| **Architecture** | `distilbert-base-uncased`, fine-tuned with a classification head |
| **Developer** | Yuvaraj Guliani, as an individual course project |
| **Date** | November 2026 |
| **Training** | Learning rate 5e-5, best at epoch 9 of 10, batch size 32, weight decay 0.01, one epoch of warmup, mixed precision, seed 42 |
| **Input limit** | 128 tokens |
| **Size** | 269 MB |
| **Output** | A probability for each of the 77 intents |

The trained weights are not stored in the repository because of their size. The notebook produces them and saves them to Google Drive.

## Intended use

- Learning about and demonstrating intent classification for chatbots.
- Research on when a chatbot should answer, ask or hand over.

## Out-of-scope use

- **Real customer support without human oversight.** This is not a production banking system.
- Making or triggering financial decisions or actions, such as blocking cards or moving money.
- Languages other than English, or domains other than retail banking.
- Messages containing several requests at once.

## Training data

BANKING77 (Casanueva et al., 2020), CC BY 4.0.

| Split | Messages |
|---|---|
| Training | 8,982 |
| Validation (stratified 10%) | 998 |
| Test (official, unchanged) | 3,080 |

Text cleaning covers whitespace only. 12 near-duplicates within the training data and 11 near-copies of test messages were removed from the training data.

## Performance

Test set, 3,080 messages, 40 per intent. The 95% range comes from 1,000 bootstrap resamples.

| Metric | Value |
|---|---|
| Accuracy | 0.929 |
| Macro precision | 0.932 |
| Macro recall | 0.929 |
| Macro F1 | 0.929 (95% range 0.919 to 0.937) |

| High-risk intent | Precision | Recall | F1 |
|---|---|---|---|
| `lost_or_stolen_card` | 0.854 | 0.875 | 0.864 |
| `compromised_card` | 0.902 | 0.925 | 0.914 |

| Other test | Result |
|---|---|
| Macro F1 with 1 typo per message | 0.814 |
| Macro F1 with 2 typos per message | 0.714 |
| Off-topic messages flagged (CLINC150) | 94.8% |
| Banking messages wrongly flagged | 9.6% |
| Off-topic AUROC | 0.978 |
| Median CPU time per message | 57 ms (96 ms for the slowest 5%; varies between runs) |

## Decision thresholds

Both were chosen on the validation split and are saved in `demo_settings.json`.

| Threshold | Value | Use |
|---|---|---|
| Clarify | 0.90 | Answer only if the top probability leads the second by at least this much |
| Off-topic | 0.939 | Answer only if the top probability reaches this. Below it, ask a clarifying question if the top two together reach it, otherwise ask to rephrase |

With these thresholds, on the banking test set: 87.2% answered correctly, 2.8% answered wrongly, 5.8% asked a clarifying question and 4.1% asked to rephrase. These figures assume that customers pick the right option when asked.

## Limitations

- **High-risk misses.** 8 of 80 lost, stolen or compromised card messages were misclassified. A safety warning depends on a high-risk intent being among the top predictions, so an unusually worded report can miss it.
- **Typos.** Macro F1 falls by 0.21 with two typos per message.
- **Confident mistakes.** High confidence does not guarantee a correct prediction.
- **Similar intents.** Most errors are between closely related intents, especially transfers and top-ups.
- **Label noise.** Some BANKING77 labels look questionable, which affects both training and measured scores.
- **Off-topic estimate.** CLINC150 messages differ in style from BANKING77, so the off-topic results may be optimistic.

## Ethical considerations

- **Customer harm.** A missed report of a lost or stolen card could lead to financial loss. Any real use should send these messages to a person, with safeguards that do not depend on the model alone.
- **Unequal performance.** The model was only tested on English text. People who write in dialects, in a second language or with spelling difficulties may get worse results, as the typo test suggests. This was not measured directly.
- **Privacy.** Customer messages can contain personal and financial details. A real deployment would need secure handling, retention limits and a lawful basis for processing. The public app link should not be used with real customer data.
- **Transparency.** Customers should know that they are talking to an automated system, and be able to reach a person.

## Licenses and credits

- BANKING77: Casanueva et al. (2020), CC BY 4.0.
- CLINC150: Larson et al. (2019), CC BY 3.0, used for off-topic evaluation only.
- DistilBERT: Sanh et al. (2019). RoBERTa: Liu et al. (2019).
