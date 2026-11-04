"""Preprocessing parameters (single source of truth). See docs/preprocessamento_v2.md, section 7.

Starting values: confirm against the unlabeled tracking distribution itself or against
other matches. NEVER tune on the public 5-minute window or pin values to specific frames.
"""

SCHEMA_VERSION = "1.0"

N_FRAMES = 199_319          # rows in every intermediate file (video_frame 0..199318)
FPS = 25

# Pitch size in metres: prefer games.GAMES["BAR-ATH"]["pitch"] (105.3, 68.0); this is the fallback.
PITCH_L, PITCH_W = 105.3, 68.0

# Ball
BALL_VMAX = 38.0            # m/s, physical upper bound for a very hard shot (range 35-38)
BALL_EPS = 1.0              # m, tracker position noise (sensitivity 0.5 / 1 / 2)
BALL_K = 30                 # prior observations considered by the jump filter (stable for 20..60)
MAX_GAP_FRAMES = 12         # max gap length to interpolate (0.5 s); compare 6 / 12 / 25
MIN_BALL_SPEED_DIR = 1.0    # m/s, minimum ball speed to compute direction change

# Players
PLAYER_VMAX = 12.0          # m/s, above human sprint; jumps above this become slot_break
DEDUP_DIST = 1.0            # m, same-team players closer than this are treated as duplicates
STILL_SPEED = 0.5           # m/s, below this the player counts in n_still_*

# Orientation
ORIENT_WIN = 25             # frames after kickoff used to find the attack direction
# Published FAQ/DATA values in the tactical video_frame reference (0-based);
# not derived from tuning on the public five-minute window.
KICKOFF_FALLBACK = {1: 11_185, 2: 113_742}   # V0 / sanity check only

# Proximity and pitch
RADII = (2.0, 3.0, 5.0)     # m, counts n_*_r2/r3/r5
PRESSURE_TAU = 3.0          # m, exponential pressure decay sum(exp(-d / tau))
PITCH_MARGIN = 1.0          # m, out_of_pitch margin
