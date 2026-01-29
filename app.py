import os
from pathlib import Path

import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# ---------------------------
# 1) Paths + API Key
# ---------------------------
BASE_DIR = Path(__file__).parent  # F1-Race-Intelligence-Copilot folder
DATA_DIR = BASE_DIR              # your CSVs are in the same folder

load_dotenv()  # will read .env from current working directory
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Put it in a .env file as OPENAI_API_KEY=...")

client = OpenAI(api_key=OPENAI_API_KEY)


# ---------------------------
# 2) Load data
# ---------------------------
races = pd.read_csv(DATA_DIR / "races.csv")
results = pd.read_csv(DATA_DIR / "results.csv")
lap_times = pd.read_csv(DATA_DIR / "lap_times.csv")
circuits = pd.read_csv(DATA_DIR / "circuits.csv")
constructors_df = pd.read_csv(DATA_DIR / "constructors.csv")

# Lookups
circuit_map = circuits.set_index("circuitId")["name"].to_dict()
constructor_map = constructors_df.set_index("constructorId")["name"].to_dict()

def get_circuit_name(circuitId: int) -> str:
    return circuit_map.get(int(circuitId), f"Circuit {circuitId}")

def get_constructor_name(constructorId: int) -> str:
    return constructor_map.get(int(constructorId), f"Constructor {constructorId}")

constructor_choices = [(row["name"], int(row["constructorId"])) for _, row in constructors_df.iterrows()]


# ---------------------------
# 3) Build ML table
# ---------------------------
races_small = races[["raceId", "year", "round", "circuitId", "name"]].copy()
results_small = results[["raceId", "driverId", "constructorId", "grid", "positionOrder"]].copy()
laps_small = lap_times[["raceId", "driverId", "lap", "milliseconds"]].copy()

df = laps_small.merge(races_small, on="raceId", how="left").merge(results_small, on=["raceId", "driverId"], how="left")

df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
df["constructorId"] = pd.to_numeric(df["constructorId"], errors="coerce")
df = df.dropna(subset=["constructorId", "grid"])

df["grid"] = df["grid"].astype(int)
df["constructorId"] = df["constructorId"].astype(int)

# ---------------------------
# 4) Train model (baseline)
# ---------------------------
features = ["year", "round", "circuitId", "constructorId", "grid", "lap"]
X = df[features]
y = df["milliseconds"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
print(f"✅ Model trained. MAE ≈ {mae/1000:.3f} seconds")


# ---------------------------
# 5) Strategy simulation + LLM explain
# ---------------------------
def simulate_strategy(model, base_input, pit_lap_change):
    base = pd.DataFrame([base_input])
    base_pred = model.predict(base)[0]

    new = base_input.copy()
    new["lap"] = new["lap"] + pit_lap_change
    new_df = pd.DataFrame([new])
    new_pred = model.predict(new_df)[0]

    delta = (new_pred - base_pred) / 1000

    return {
        "original_time_s": float(base_pred / 1000),
        "new_time_s": float(new_pred / 1000),
        "delta_seconds": float(delta)
    }

def explain_markdown(title, resA, resB, teamA_name, teamB_name, circuit_name):
    prompt = f"""
You are an F1 performance engineer. Answer in clean Markdown.

### {title}

Circuit: {circuit_name}

Team A: {teamA_name}
- Original: {resA['original_time_s']:.3f}s
- New: {resA['new_time_s']:.3f}s
- Delta: {resA['delta_seconds']:.3f}s

Team B: {teamB_name}
- Original: {resB['original_time_s']:.3f}s
- New: {resB['new_time_s']:.3f}s
- Delta: {resB['delta_seconds']:.3f}s

Explain:
- Which team is predicted faster after the change and why (use the numbers)
- What this implies for strategy
- 1 limitation (tyres/traffic/weather not modeled)
Keep it concise.
"""
    resp = client.responses.create(model="gpt-4.1-mini", input=prompt)
    return resp.output[0].content[0].text


# ---------------------------
# 6) Gradio compare function
# ---------------------------
def compare_two_teams(year, round_no, circuitId, grid, lap, pit_change, teamA, teamB):
    # Cast safely (Gradio Numbers)
    year = int(float(year))
    round_no = int(float(round_no))
    circuitId = int(float(circuitId))
    grid = int(float(grid))
    lap = int(float(lap))
    pit_change = int(float(pit_change))
    teamA = int(float(teamA))
    teamB = int(float(teamB))

    cname = get_circuit_name(circuitId)
    A = get_constructor_name(teamA)
    B = get_constructor_name(teamB)

    baseA = {"year": year, "round": round_no, "circuitId": circuitId, "constructorId": teamA, "grid": grid, "lap": lap}
    baseB = {"year": year, "round": round_no, "circuitId": circuitId, "constructorId": teamB, "grid": grid, "lap": lap}

    resA = simulate_strategy(model, baseA, pit_change)
    resB = simulate_strategy(model, baseB, pit_change)

    winner = A if resA["new_time_s"] < resB["new_time_s"] else B

    summary = f"""
## 🆚 Team vs Team Comparison
**Circuit:** {cname}  
**Scenario:** Year {year}, Round {round_no}, Grid {grid}, Lap {lap}, Δlap {pit_change}

| Team | Original (s) | New (s) | Delta (s) |
|---|---:|---:|---:|
| {A} | {resA['original_time_s']:.3f} | {resA['new_time_s']:.3f} | {resA['delta_seconds']:.3f} |
| {B} | {resB['original_time_s']:.3f} | {resB['new_time_s']:.3f} | {resB['delta_seconds']:.3f} |

✅ **Predicted faster team after change:** **{winner}**
"""

    title = f"Comparison: {A} vs {B} | Year {year} Round {round_no} | Grid {grid} Lap {lap} | Δlap {pit_change}"
    explanation = explain_markdown(title, resA, resB, A, B, cname)

    return summary, explanation


# ---------------------------
# 7) UI
# ---------------------------
gr.close_all()

with gr.Blocks() as app:
    gr.Markdown("# 🏁 F1 Race Intelligence Copilot\nCompare two constructors with ML predictions + LLM explanations.")

    with gr.Row():
        year = gr.Number(value=2011, label="Year")
        round_no = gr.Number(value=1, label="Round")
        circuitId = gr.Number(value=1, label="Circuit ID")
        grid = gr.Number(value=1, label="Grid Position")
        lap = gr.Number(value=20, label="Lap")

    pit_change = gr.Slider(-10, 10, value=-3, step=1, label="Pit timing change (laps)")

    teamA = gr.Dropdown(choices=constructor_choices, value=9, label="Team A (Constructor)")
    teamB = gr.Dropdown(choices=constructor_choices, value=6, label="Team B (Constructor)")

    btn = gr.Button("Compare Teams")

    out1 = gr.Markdown()
    out2 = gr.Markdown()

    btn.click(
        fn=compare_two_teams,
        inputs=[year, round_no, circuitId, grid, lap, pit_change, teamA, teamB],
        outputs=[out1, out2]
    )

if __name__ == "__main__":
    app.launch(share=True, show_error=True)
