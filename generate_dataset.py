from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from scenarios import generate_hiring_dataset, SCENARIOS

scenario = "Balanced"
df = generate_hiring_dataset(scenario=scenario, n=1200, seed=42)

out = Path(__file__).resolve().parent / "data" / "hiring_synthetic.csv"
df.to_csv(out, index=False)

print(f"Created {len(df)} rows at {out}")
print("Scenario:", scenario)
print("Available scenarios:", ", ".join(SCENARIOS))
