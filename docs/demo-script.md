# Demo video script

Target length: 3 minutes. The final video must be public, no longer than five
minutes, and include English narration or English subtitles.

## 0:00–0:25 — Problem and audience

Show an incoming Japanese phone call and the empty Assist Card area.

Narration:

> Foreign residents in Japan often receive calls about deliveries,
> appointments, housing, utilities, or schools. When they cannot understand the
> caller, they also cannot decide how to answer. AlphaMind explains the latest
> message and provides relevant Japanese reply options.

## 0:25–0:55 — One App, two Agents

Show `main.py` and focus on the two `app.agent(...)` lines.

Narration:

> This is one Python app with two independent Agents. The Japanese Call Listener
> owns live audio and speech recognition. The Japanese Reply Agent uses Strands
> Agents for translation, reply options, and commitment checks. The app routes
> a typed handoff between them and traces both turns.

## 0:55–1:35 — Deterministic local demo

Run:

```bash
uv run --frozen python main.py demo
```

Show the Japanese source, Chinese translation, reply suggestions, and deadline
confirmation. Then show the two Agent traces.

## 1:35–2:15 — Live voice

Run:

```bash
uv run --frozen python main.py
```

Speak one ordinary Japanese sentence, then one sentence containing a date or
price. Show the final transcript becoming an Assist Card. Keep audio and screen
recording synchronized.

## 2:15–2:40 — Strands implementation

Show `japanese_reply_agent.py`: `Agent`, `@tool`, `publish_call_assist`, and the
deterministic validation/retry boundary.

Narration:

> Strands owns the specialist Agent loop and structured tool call. It produces
> several response options for the caller's actual request. The application
> rejects copied, duplicate, or unsafe answers and retries the same model.

## 2:40–3:00 — Impact

Show the architecture diagram and final working card.

Narration:

> The result helps a foreign resident understand a Japanese call and choose a
> reply while the conversation continues. The human stays responsible for
> appointments, prices, and other commitments.

## Recording checklist

- Use the 3B local model and multilingual Whisper model.
- Capture one complete successful run before recording the final take.
- Show the working product, not only code or slides.
- Include the problem, audience, and why it matters.
- Keep secrets, local paths, notifications, and unrelated windows off-screen.
- If any narration or UI explanation is not English, add English subtitles.
