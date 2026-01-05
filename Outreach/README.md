# AI Referral Pitch Agent

A LangGraph-powered agentic workflow that generates personalized referral pitches by analyzing LinkedIn profiles.

## Features

- **Profile Scraping**: Extracts raw text from LinkedIn profiles
- **AI-Powered Extraction**: Uses Qwen3:8b to extract structured information
- **Smart Pitch Writing**: Generates tailored cold outreach messages

## Workflow Architecture

The agent uses **parallel execution** for optimal performance:

```
       ┌─────────────────────────────┐
       │           START             │
       └──────────┬────────┬─────────┘
                  │        │
     ┌────────────▼┐      ┌▼────────────┐
     │ get_candidate│      │ get_receiver │  ← Run in parallel
     │ profile      │      │ profile      │
     └─────────┬───┘      └───┬─────────┘
               │              │
     ┌─────────▼───┐      ┌───▼─────────┐
     │extract_cand │      │extract_recv │  ← Run in parallel
     │info         │      │info         │
     └─────────┬───┘      └───┬─────────┘
               │              │
               └──────┬───────┘
                      │
              ┌───────▼───────┐
              │write_a_referral│  ← Waits for both
              │pitch           │
              └───────┬───────┘
                      │
                     END
```

## Requirements

- Python 3.9+
- Ollama with `qwen3:8b` model
- LangChain & LangGraph

## Usage

```bash
python agent.py
```

## License

MIT
