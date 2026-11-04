"""Compare kloppy per-slot team with team_of_each_track (aggregates only)."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from kloppy import metrica

from games import asset_paths
from starter.baseline import team_of_each_track

paths = asset_paths(source="tactical")
ds = metrica.load_tracking_epts(meta_data=str(paths["atd_meta"]),
                                raw_data=str(paths["atd_raw"]))

ours = np.asarray(team_of_each_track(paths["atd_meta"]))  # 0 = Home, 1 = Away

# kloppy: numeric player id -> 0 (home) or 1 (away)
kloppy_team = {}
for team in ds.metadata.teams:
    code = 0 if "home" in str(team.ground).lower() else 1
    for pl in team.players:
        m = re.search(r"(\d+)", str(pl.player_id))
        if m:
            kloppy_team[int(m.group(1))] = code

ids = sorted(kloppy_team)
n = len(ours)
print("slots no nosso vetor:", n)
print("jogadores no kloppy:", len(ids))
print("ids do kloppy cobrem 1..%d:" % n, ids == list(range(1, n + 1)))

if ids == list(range(1, n + 1)):
    kl = np.array([kloppy_team[i] for i in ids])
    print("slots que divergem:", int((kl != ours).sum()))
    print("home no kloppy:", int((kl == 0).sum()), "| home no nosso:", int((ours == 0).sum()))
else:
    print("ids não mapeiam 1..N; não comparei (reporte esta linha).")

cols = ds.to_df().columns
extra = sorted({re.sub(r"^.*?_", "", c) for c in cols if "_" in c} - {"x", "y", "s"})
print("sufixos de coluna além de x, y, s:", extra[:10])