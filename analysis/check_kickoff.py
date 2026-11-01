"""Concordância slot a slot entre o time do XML e a metade ocupada no saque inicial.

Usa o tracking já orientado (Home atacando em x crescente), então no saque o Home
deve estar em x < metade do campo e o Away em x > metade, nos dois tempos.
Saída: só contagens e proporções.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from preprocessing.params import PITCH_L
from preprocessing.prep_tracking import _kickoff_frames

WINDOW = 25        # frames de cada lado do saque (~1 s de cada lado)
MARGIN = 3.0       # ignora slots a menos de 3 m da linha do meio
MID = PITCH_L / 2

data = pd.read_parquet("data/processed/tracking_clean.parquet")
home = pd.read_parquet("data/processed/slot_teams.parquet").is_home.to_numpy()
kickoffs = _kickoff_frames()  # kickoffs.json se existir; senão o fallback publicado

print("slots Home no XML:", int(home.sum()), "| Away:", int((~home).sum()))

for half in (1, 2):
    k = kickoffs[half]
    w = data[(data.video_frame >= k - WINDOW) & (data.video_frame <= k + WINDOW)]
    x = np.array([w[f"s{i:02d}_x"].median() for i in range(1, len(home) + 1)])

    observed = np.isfinite(x)
    usable = observed & (np.abs(x - MID) > MARGIN)
    in_low_half = x < MID                      # metade de x menor
    agree = in_low_half == home                # Home deve estar em x menor

    n = int(usable.sum())
    print(f"\ntempo {half}")
    print("  frames na janela:", len(w))
    print("  slots observados:", int(observed.sum()), "| avaliáveis (fora da margem):", n)
    if n:
        print("  concordância geral:", round(float(agree[usable].mean()), 3))
        for name, mask in (("Home", home), ("Away", ~home)):
            m = usable & mask
            if m.sum():
                print(f"  concordância {name}: {round(float(agree[m].mean()), 3)} (n={int(m.sum())})")