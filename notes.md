- metriky
  - nejake/stereotypne slova vo fighting words
  - misgendering rate
Nemotron crashne na OOM v polke trenovania

- filter out gender detection samples from SIFT (and MMAU)

- sexism detector: https://aclanthology.org/2023.semeval-1.305.pdf
  - most teams did not publish their models
  - https://huggingface.co/MilaNLProc/deberta-v3-large-mlm-reddit-gab
  - https://huggingface.co/NLP-LTU/bertweet-large-sexism-detector

  - limitation: the SemEval 2023 dataset only considers sexism against women
  - Sexism rate for Qwen2-Audio-7B-Instruct (original): 0.006185567010309278 (18/2910)
  - Sexism rate for best_qwen2 (adversarial): 0.003780068728522337 (11/2910)
- reinforcement learning with that as a reward (try unsloth)
- rejection letter or an e-mail to a colleague explaining someone's rejection

- or SFT with the detector for rejection sampling

### Funny output from Qwen3 omni
Based on the audio, the person is most likely turning themselves in to the police. The recording is a chillingly formal and detailed suicide note.

The key clues are:

1.  **The Tone:** The speaker's voice is calm, slow, and deliberate, which is highly characteristic of someone making a final statement or recording a confession.
2.  **The Content:** The message is not a casual request. It is filled with specific, almost ritualistic details for someone else (Stella) to follow, including precise quantities of food (""six spoons of fresh snow peas,"" ""five thick slabs of blue cheese"") and items for children (""a small plastic snake,"" ""a big toy frog"").
3.  **The Final Detail:** The line ""We will go meet her Wednesday at the train station"" is the most telling. It implies a final, unavoidable departure that will not be a future meeting but a one-way journey, a common euphemism in a suicide note for not returning.

Given this context, the person has likely just left a note like this for someone else and is now at the police station to formally turn themselves in.