# Voice to BioBERT Disease Classification

This folder runs the Thesis 2 inference path:

```text
voice or audio file
  -> DC-offset removal
  -> silence trimming
  -> stationary noise reduction
  -> amplitude normalization
  -> Whisper transcription
  -> text normalization
  -> narrative-to-symptom extraction
  -> rule-based urgency detection
       -> emergency/urgent: safety response and stop
       -> routine: continue to clarification check
  -> clarification-question generation
       -> missing/uncertain symptoms: ask questions and stop
       -> sufficient symptoms: continue to classification
  -> final trained 773-class BioBERT model
  -> score and top-two margin checks
       -> low/ambiguous result: withhold class and ask questions
       -> accepted result: show ranked disease-class predictions
  -> rule-based response generation
       -> emergency/urgent: safety action only
       -> clarification: questions and no disease class
       -> routine prediction: sourced symptom guidance and professional follow-up
```

The code uses the latest exported classifier by default:

```text
../3 Models After Training/outputs/models/biobert
```

The label mapping is read directly from that model's `config.json`. Do not create
a new mapping or sort the disease names again during inference.

## Installation

From this folder, create and activate a Python environment:

```powershell
conda create -n thesis_chatbot python=3.10 -y
conda activate thesis_chatbot
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first audio run downloads `openai/whisper-small`. The BioBERT classifier is
loaded only from the local final model folder and is not downloaded again.

For MP3 or M4A input, install FFmpeg if `librosa` cannot decode the file. WAV is
the simplest format for testing.

## Run with an audio file

```powershell
python voice_to_disease.py --audio-file "C:\path\to\symptom_scenario.wav"
```

Save the denoised audio and complete output:

```powershell
python voice_to_disease.py `
  --audio-file "C:\path\to\symptom_scenario.wav" `
  --save-processed-audio "outputs\processed.wav" `
  --output-json "outputs\prediction.json"
```

## Record from the microphone

This records a 15-second symptom description:

```powershell
python voice_to_disease.py --record-seconds 15
```

Example spoken scenario:

```text
I have sharp abdominal pain, nausea, diarrhea, fever, and chills.
```

Speak clearly and describe several relevant symptoms. Keep the first test below
25 seconds so Whisper can process it as one chunk.

If microphone recording reports that `sounddevice` is unavailable, reinstall the
requirements and list the detected microphones with:

```powershell
pip install -r requirements.txt
python -m sounddevice
```

## Test BioBERT without audio

Use direct text to verify the local classifier before downloading Whisper:

```powershell
python voice_to_disease.py --text "sharp abdominal pain, nausea, diarrhea, fever and chills"
```

Natural patient wording is converted to the short symptom-list format used in
training. For example:

```powershell
python voice_to_disease.py --text "My head hurts, I feel nauseous, I vomited twice, and I saw flashing lights."
```

The resulting BioBERT input is:

```text
headache, nausea, vomiting, spots or clouds in vision
```

The extraction result records present, negated, uncertain, historical, and
other-person symptom mentions. Only currently present symptoms are sent to
BioBERT. If fewer than two current symptoms are found, the pipeline marks the
result for clarification. If none are found, disease classification is skipped.

## Urgency detection before classification

`urgency_detector.py` evaluates the original narrative and the asserted current
symptoms before BioBERT is loaded. It returns one of three levels:

| Level | Pipeline behavior |
|---|---|
| `emergency` | Show the emergency action and bypass BioBERT |
| `urgent` | Recommend prompt same-day assessment and bypass BioBERT |
| `routine` | Continue to symptom confirmation and BioBERT |

The first rule set covers severe breathing difficulty or unresponsiveness,
stroke warning signs, dangerous chest-pain combinations, possible anaphylaxis,
seizure emergencies, major bleeding, selected pregnancy emergencies, fever with
neck stiffness, urinary retention, and immediate mental-health danger. The
detailed rule and source table is in `URGENCY_RULES.md`.

Run the gate without Whisper or BioBERT:

```powershell
python urgency_detector.py "I have crushing chest pain, I am short of breath and sweating."
```

Malaysia's `999` is the default emergency number. Configure another location
without changing the code:

```powershell
python voice_to_disease.py `
  --text "I am struggling to breathe" `
  --emergency-number "LOCAL_EMERGENCY_NUMBER" `
  --urgent-service "your local urgent-care service"
```

The research-only `--disable-urgency-detection` option is available for controlled
classifier experiments. It should remain disabled in the chatbot application.

## Clarification questions

`clarification_questions.py` generates at most three structured follow-up
questions. It asks before BioBERT when no current symptom is found, fewer than
two current symptoms are found, or a symptom is uncertain. It can also ask after
BioBERT when the top score is below `--minimum-score` or the difference between
the top two scores is below `--minimum-margin`.

Emergency and urgent actions always take priority. In those cases, clarification
is deferred and BioBERT is not loaded. For routine cases, the output JSON contains:

```text
clarification.clarification_required
clarification.should_pause_classification
clarification.should_withhold_prediction
clarification.reason_codes
clarification.questions[]
classification.display_prediction
```

The chatbot UI must show a disease class only when
`classification.display_prediction` is `true`. Each question includes a stable
`question_id`, category, answer type, purpose, and `safety_related` flag.

Run the question generator without Whisper or BioBERT:

```powershell
python clarification_questions.py "I have a headache."
```

For the next conversational turn, keep the original narrative and append the
answer before rerunning the full pipeline. For example:

```text
Turn 1: "I have a headache."
Question: "Where is the headache, when did it start, and how severe is it?"
Turn 2 combined input: "I have a headache. It started yesterday, is behind my
eyes, and is 7 out of 10. I also feel nauseous."
```

Rerunning extraction and urgency on the combined input is important because an
answer may introduce a new emergency warning sign. The implementation and JSON
contract are described further in `CLARIFICATION_QUESTIONS.md`.

## Rule-based responses

`rule_based_response.py` converts the pipeline decision into a deterministic
chatbot response. Its response types are `emergency`, `urgent`, `clarification`,
`routine_prediction`, and `no_prediction`. It never changes the urgency level or
creates another disease prediction.

For an accepted BioBERT result, the response describes the class as similarity
to a training-dataset class rather than a diagnosis. It can add conservative,
source-backed guidance for headache, respiratory, gastrointestinal, fever,
dizziness, and itchy-skin symptom groups. Other classes receive only the general
professional-assessment fallback. The rules do not prescribe medicines or give
disease-specific treatment.

The output JSON contains the ready-to-display message in:

```text
rule_based_response.response_text
```

The structured fields include `response_type`, `displayed_prediction`,
`recommended_actions`, `self_care_guidance`, `safety_net_guidance`,
`clarification_questions`, `matched_guidance_rule_ids`, and `sources`.
Implementation details and source links are in `RULE_BASED_RESPONSES.md`.

## Test the symptom extractor by itself

This command does not load Whisper or BioBERT:

```powershell
python symptom_extractor.py "I am wheezing and short of breath, but I have no fever." --json
```

Run the focused tests with:

```powershell
python -m unittest discover -s tests -v
```

The deployable canonical vocabulary is in
`resources/symptom_vocabulary.json`. It contains the 333 comma-delimited symptom
tokens from the underlying `fhai50032/SymptomsDisease246k` data used to create
the 773-class training dataset. `build_symptom_vocabulary.py` makes this artifact
reproducible; normal inference does not need the source JSON.

## Before-and-after comparison

Symptom extraction is enabled by default. Use the same narrative with
`--disable-symptom-extraction` to reproduce the old full-transcript path:

```powershell
python voice_to_disease.py --text "I have a severe headache, feel nauseous, vomited twice, and saw flashing lights."

python voice_to_disease.py --text "I have a severe headache, feel nauseous, vomited twice, and saw flashing lights." --disable-symptom-extraction
```

This switch is intended for controlled A/B evaluation. It should not be used to
hide unsuccessful examples or to claim disease-classification improvement from
one scenario.

## Useful options

```text
--device auto                 Use CUDA when available, otherwise CPU
--top-k 3                     Return the three highest-scoring classes
--minimum-score 0.50          Mark lower scores for symptom review
--minimum-margin 0.10         Clarify when the top two classes are too close
--minimum-extracted-symptoms 2  Ask for clarification below this count
--max-clarification-questions 3 Limit questions returned in one turn
--disable-symptom-extraction Send the full normalized narrative to BioBERT
--emergency-number 999       Configure the displayed emergency number
--urgent-service TEXT        Configure local same-day care wording
--disable-urgency-detection  Research-only bypass of the safety gate
--disable-rule-based-response Research-only omission of the response stage
--skip-audio-preprocessing    Run the raw-audio comparison path
--whisper-model MODEL         Change Whisper model or use a local path
```

The model score is a softmax output, not a clinically calibrated probability.
BioBERT predicts one of the 773 disease classes from the training dataset; it
does not establish a diagnosis. Urgency detection runs before BioBERT and
overrides ordinary disease responses.
For a vague, low-score, or ambiguous input, the pipeline asks the user for more
information and sets `classification.display_prediction` to `false`.

Symptom extraction and disease classification must be evaluated separately.
Correctly extracted respiratory symptoms can still produce the wrong disease
class because the classifier is limited by its training data and calibration.
The chatbot UI should show the extracted symptoms for user confirmation before
using them for classification.
