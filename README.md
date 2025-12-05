# ES Options

Options analytics for ES (E-mini S&P 500 futures).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Add your EOD API key to `.env`:
```
EOD_API_KEY=your_key
```

## Usage

```bash
# Fetch chain
python -m scripts.fetch_eod_es_chain --symbol ES

# Build RND surface
python -m scripts.build_daily_rnd_surface --symbol ES --plot

# Debug/inspect data
python -m scripts.debug_chain --symbol ES
```

## Breeden-Litzenberger

Risk-neutral density from call prices:

```
f_Q(K) = e^(rT) * d²C/dK²
```
