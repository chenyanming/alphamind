# AlphaMind

AlphaMind gives foreign residents real-time intelligence for Japanese phone
calls. It listens to the caller and explains the meaning in Simplified Chinese.
It then provides two relevant Japanese reply options. It flags dates,
prices, appointments, and other commitments before the user answers.

The project is one Python application with two independent Agents. It targets
the Everyday Agents track of the Agents for Humans Hackathon.

![AlphaMind architecture](docs/architecture.png)

## The two-Agent design

```python
import os

from vifu import LocalWhisper, OpenAICompatible, Vifu
from japanese_reply_agent import JapaneseReplyAgent
from reasoning_config import ReasoningConfig
from speaker_identification import LocalSpeakerIdentifier
from voice_agent import JapaneseCallListenerAgent

app = Vifu("AlphaMind")

call_listener = JapaneseCallListenerAgent(
    language="ja-JP",
    handoff="japanese-reply-agent",
    transcriber=app.provider(
        "local-whisper",
        LocalWhisper(model="ggml-small.bin", language="ja"),
    ),
    speaker_identifier=LocalSpeakerIdentifier(),
)

reasoning_provider = app.provider(
    "openai-compatible",
    OpenAICompatible(
        url=(
            f"{os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1').rstrip('/')}"
            "/chat/completions"
        ),
        model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=30,
    ),
)
reply_agent = JapaneseReplyAgent(
    ReasoningConfig.app_provider(reasoning_provider)
)

app.agent(
    "japanese-call-listener",
    call_listener,
    implementation="livekit-agents",
    providers={"transcription": app.providers["local-whisper"]},
)
app.agent(
    "japanese-reply-agent",
    reply_agent,
    implementation="strands-agents",
    providers={"reasoning": reasoning_provider},
)
app.run()
```

- `japanese-call-listener` owns the LiveKit audio session, VAD, speech-to-text,
  local speaker identification, speaker routing, final-turn ordering, handoff,
  and result delivery. It enrolls the listener's voice, sends only caller
  turns to Strands, and keeps uncertain turns out of reasoning. It drops known non-speech
  markers and suppresses repeated final transcripts for three seconds per
  session and speaker. Fragment-level speaker decisions are combined into one
  turn-level role, and completed turns wait in capture order while reasoning is
  busy so no intermediate speech is silently replaced.
- `japanese-reply-agent` is a Strands specialist. It explains the caller's
  Japanese, provides distinct response options, identifies commitments and
  risks, and publishes one schema-validated Assist Card.
- The `vifu` Python package registers both Agents, routes their typed handoff,
  and records each invocation as a separate trace.

The Call Listener never performs the specialist reasoning. The Reply Agent
never owns the audio room. Provider errors fail visibly and do not switch the
application to another model.

## Local quickstart

The verified development environments use Python 3.12 and 3.13 with `uv`.

Place these models in `~/.vifu/models/`:

```text
~/.vifu/models/ggml-small.bin
~/.vifu/models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
```

- `ggml-small.bin` is the multilingual
  [whisper.cpp small model](https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-small.bin).
  The English-only model cannot transcribe Japanese.
- `3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx` is the
  [sherpa-onnx speaker model](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models).
  It identifies the listener and multiple callers locally.

Install the locked dependencies from PyPI:

```bash
uv sync --frozen
```

The lock file pins all public Python dependencies. `uv` manages the project
environment, so you do not need to create or activate a virtual environment.

Configure any service that implements the OpenAI Chat Completions API. For
example, to use OpenAI:

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4.1-mini"
export OPENAI_API_KEY="<your-api-key>"
```

`OPENAI_BASE_URL` is the API base URL, without `/chat/completions`. Replace the
base URL and model to use another compatible provider. `OPENAI_API_KEY` may be
omitted when the selected provider does not require authentication. The key is
used only by the App-private reasoning Provider. It is not written to the App
manifest, trace metadata, or repository.

To exercise one deterministic two-Agent handoff without opening the microphone,
run:

```bash
uv run --frozen python main.py demo
```

This command sends one Japanese turn through both Agents and prints the Assist
Card. `uv` manages the project environment, so no virtual environment activation
step is required.

Then start the microphone experience:

```bash
uv run --frozen python main.py
```

AlphaMind first asks you to say one complete sentence. That voice becomes the
listener profile for the current session. Later listener speech is shown but
does not trigger Strands. Caller voices get stable `caller-N` labels. The app
skips short or ambiguous speech instead of guessing the speaker.

The default console shows the microphone meter, final Japanese transcript,
analysis status, and Assist Card. Framework debug logs, raw tool calls, and
native model-loading messages stay out of the user view. For development
diagnostics, run `uv run --frozen python main.py --debug`.

[LiveKit console mode](https://docs.livekit.io/agents/server/startup-modes/)
uses the computer audio device. The voice pipeline runs in the local process.
`main.py` configures local Whisper, local speaker identification, and an
OpenAI-compatible reasoning Provider directly in Python. Each Provider is
declared once in the App Provider list and bound to the Agent that uses it. The
application does not read `.env.local` or `~/.vifu/providers.json`, and it does
not join a hosted LiveKit room.

## Tests

Run the App suite:

```bash
uv run --frozen python -m unittest discover -s tests -v
```

The tests cover the two-Agent registration, the real in-process handoff,
non-speech and duplicate-turn filtering, Strands tool schema, result validation,
interruption ordering, and deterministic retry of unsafe model output.

## Model and safety behavior

The Strands Agent must call its `card` tool exactly once per attempt. The tool
requires two response options. Pydantic and deterministic checks
reject duplicate responses, copied caller requests, unsafe commitments, and
incorrect dates. The application uses deterministic decoding and retries a
rejected card once with the same model. If the result is still unusable, the
Agent returns a safe clarification card so the voice session can continue.
When the caller asks the listener for personal or booking information, the
model translates the request while the application supplies safe listener-side
responses. It never fills unknown names, dates, amounts, or numbers for the
user.

## Hackathon fit

The specialist is a real Strands Agent, not a prompt-only wrapper. It uses the
Strands `Agent` loop and a structured `@tool`. A deterministic validation
boundary checks each result before the application shows an Assist Card.

The [official requirements](https://agentsforhumans.devpost.com/rules) state
that Amazon Bedrock AgentCore deployment can strengthen the Technical
Implementation score, but is not required. This project therefore keeps cloud
deployment outside the verified submission path. A complete entry still needs
the public repository, architecture diagram, and a public demo video of no more
than five minutes. It also needs the submitter's AWS Builder ID.

## Project provenance

AlphaMind was created during the hackathon submission period. It uses the
public Strands Agents, LiveKit Agents, Pydantic, and `vifu` Python packages.
The `vifu` package supplies Provider adapters and Agent routing. Local model
files are separate downloads and are not part of this repository.

The project is licensed under Apache License 2.0. See [LICENSE](LICENSE).
