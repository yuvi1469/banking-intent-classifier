# Demo script

A running order for the 10 to 15 minute presentation. Times are a guide: the full version runs about 14 minutes, and the cuts at the end bring it down to 10.

## Running order

| Time | Part | Show |
|---|---|---|
| 0:00 to 1:00 | 1. The problem | Title slide, then the problem slide |
| 1:00 to 2:30 | 2. The data | Dataset slide |
| 2:30 to 4:00 | 3. The models | Models slide |
| 4:00 to 6:00 | 4. Results | Results slide, then the high-risk slide |
| 6:00 to 7:30 | 5. Where the models go wrong | Error analysis slide |
| 7:30 to 11:00 | 6. Live demo | The demo slide, the live demo slide, BankBot Assistant, then the demo results slide |
| 11:00 to 13:00 | 7. Beyond accuracy | Beyond accuracy slide |
| 13:00 to 14:00 | 8. Limitations and next steps | Limitations slide, then the summary slide |
| 14:00 onwards | Questions | Summary slide |

## Talking points

### 1. The problem (1 minute)

- A bank's chatbot first has to work out what the customer wants, so it can answer or pass them to the right team. That is intent classification.
- There are 77 intents, many of them very similar, such as *card arrival* and *card delivery estimate*.
- Some mistakes matter more than others. Missing a lost or stolen card is worse than mixing up two top-up questions.
- Goal: at least 0.90 macro F1, answers within 200 ms, and safe handling of high-risk messages.

### 2. The data (1.5 minutes)

- BANKING77: 13,083 online banking queries in English, 77 intents, licensed CC BY 4.0.
- Messages are short, about 10 words on average. The test set has exactly 40 messages per intent, while the training set is less even.
- I fixed two label names, removed 12 near-duplicates within the training data and 11 near-copies of test messages, so the test score is not inflated.
- Split: 8,982 training, 998 validation (stratified), and the official 3,080 test messages, used only once at the end.

### 3. The models (1.5 minutes)

- **Baseline:** TF-IDF with logistic regression. Fast, simple, and a fair bar to beat.
- **DistilBERT:** a smaller, faster transformer.
- **RoBERTa:** a larger transformer, trained on more text.
- Each model was tuned on the validation split only. The transformers were tried with three learning rates, with early stopping.

### 4. Results (2 minutes)

| Model | Test accuracy | Test macro F1 |
|---|---|---|
| TF-IDF baseline | 0.897 | 0.898 |
| DistilBERT | 0.929 | 0.929 |
| RoBERTa | 0.931 | 0.931 |

- Both transformers clearly beat the baseline, in 100% of 1,000 bootstrap resamples.
- RoBERTa and DistilBERT are effectively tied: RoBERTa is ahead by 0.003, in only 74% of resamples.
- DistilBERT is about twice as fast and half the size, so I chose it for the demo.
- High-risk intents: each model misses 7 to 9 of the 80 lost, stolen or compromised card messages. That is why the demo adds a safety warning.

### 5. Where the models go wrong (1.5 minutes)

- 9 of DistilBERT's 10 most confused pairs are among the 100 most similar pairs of intents. The errors come from overlapping intents, not random noise.
- 115 test messages are wrong for all three models, and some of their labels look questionable.
- Example: "my child used my card" was predicted as *age limit*, pulled by a single word.

### 6. Live demo (3.5 minutes)

Open BankBot Assistant from the public link in section 12.4. Type each message into the chat, or click it under "Try a message". For each one, point to BankBot's reply and to the "What BankBot understood" panel. Click **New conversation** between examples so the chat stays easy to follow.

| # | Message | What happens | What to say |
|---|---|---|---|
| 1 | My card hasn't arrived yet | Answers *card arrival*, 99.9% | The easy case: confident and right. |
| 2 | I need to verify my identity, then reply **2** | Asks: 1. *why verify identity* or 2. *verify my identity*? Then answers *verify my identity* | Two intents are nearly identical, so it asks instead of guessing, and the conversation carries on from the customer's choice. |
| 3 | I think someone has been using my card | Answers *compromised card*, with the safety warning | High-risk messages always get the warning and go to a person. |
| 4 | I lost my wallet with my card in it | Safety warning, then asks to rephrase | It isn't confident enough to answer, but the warning still shows. Admit this is awkward for a worried customer. |
| 5 | my crad hasnt arived yet | Asks to rephrase, 34% confidence | Typos are the main weakness, but it asks instead of answering wrongly. |
| 6 | What's the weather going to be like tomorrow? | Asks to rephrase | Off-topic messages are caught by the confidence threshold. |
| 7 | How long will it take for my new card to get here? | Answers *card arrival*, 98.9% | Honest failure: confident but probably wrong, it should be *card delivery estimate*. |
| 8 | I think my child used my card while I wasn't home. | Asks to rephrase, no warning | Known failure from the error analysis: no wrong answer, but the warning is missed. |
| 9 | A message from the audience | Whatever happens | Invite one if time allows. |

Then give the measured numbers from section 12.2, so the examples aren't the only evidence:

- 90% of banking messages are answered, and about 97% of those answers are right.
- Wrong answers drop from 7.1% to 2.8%.
- 93% of off-topic messages are asked to rephrase.

### 7. Beyond accuracy (2 minutes)

- **Clarifying questions:** asking on about 10% of messages cuts DistilBERT's wrong answers from 219 to 151. RoBERTa gains more, down to 126.
- **Off-topic detection:** about 95% of off-topic messages flagged, with an AUROC of 0.98.
- **Typos:** two typos per message drop DistilBERT to 0.71 macro F1, level with the baseline. RoBERTa holds up better, at 0.78.
- **Speed:** DistilBERT takes about 57 ms per message on a CPU, within the 200 ms target.
- **Limited data:** with 10 examples per intent the baseline wins. With 30, DistilBERT wins.

### 8. Limitations and next steps (1 minute)

- BANKING77 is English only, and has one intent per message.
- The off-topic test messages come from another dataset and are easier to spot than real ones.
- Typos are the clearest weakness. Training on typo-filled copies of the messages is the first improvement to try.
- High-risk messages need a safety net beyond the model, such as keyword rules, so that a missed prediction still reaches a person.

## Shortening to 10 minutes

- Cut part 5 to a single sentence about similar intents.
- In part 6, show examples 1, 2, 3, 6 and 7 only.
- In part 7, cover only clarifying questions and typos.

## Likely questions

| Question | Answer |
|---|---|
| Why macro F1 rather than accuracy? | It weighs every intent equally, so rare intents count as much as common ones. The test set is balanced, so here the two come out close. |
| Why not use RoBERTa, since it scored highest? | The gap is 0.003 and not reliable. DistilBERT is twice as fast and half the size. RoBERTa is better with typos and clarifying questions, so it would be worth it if those mattered most. |
| How do you know the test score isn't inflated? | The test set was used once, after all tuning on validation. I also removed training messages that were near-copies of test messages. |
| How were the thresholds chosen? | On the validation split only: the clarify threshold to ask on at most 10% of messages, and the off-topic threshold to wrongly flag 10% of banking messages. |
| Why not use a large language model? | A fine-tuned small model is faster, cheaper, runs on a CPU and gives calibrated scores for each intent. Comparing against an LLM would be good future work. |
| What would you do with more time? | Typo augmentation, more realistic off-topic data, and keyword rules as a safety net for high-risk messages. |
| Would this work for another bank? | The method would, but the model would need retraining on that bank's own intents and messages. |

## Before presenting

- [ ] Open the notebook in Colab and choose **Runtime > Run all** at least 15 minutes before. The saved models load from Drive, so no retraining is needed.
- [ ] Open the `gradio.live` link from section 12.4 and try one message. The link only works while the Colab session is open.
- [ ] Keep the slides and the demo in separate tabs, and zoom the browser to about 125% so the audience can read it.
- [ ] Close other tabs and turn off notifications.
- [ ] Have the backup ready (below).

## Backup plan

If the link or Colab fails during the live demo:

1. Show the table of examples in section 12.3 of the saved notebook on GitHub. It has every response and confidence from a real run.
2. Carry on with the talk. The demo is about 3 minutes, and the measured results in section 12.2 still make the point.
