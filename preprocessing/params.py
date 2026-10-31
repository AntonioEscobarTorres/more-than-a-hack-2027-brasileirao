"""Parametros do pre-processamento (fonte unica). Ver docs/preprocessamento_v2.md, secao 7.

Valores de partida: confirmar na distribuicao dos proprios dados de tracking (sem rotulos) ou em
outros jogos. NUNCA ajustar nos 5 minutos publicos e nunca atrelar a frames especificos.
"""

SCHEMA_VERSION = "1.0"

N_FRAMES = 199_319          # linhas de todo arquivo intermediario (video_frame 0..199318)
FPS = 25

# Dimensoes do campo em metros: usar games.GAMES["BAR-ATH"]["pitch"] (105.3, 68.0); aqui so o fallback.
PITCH_L, PITCH_W = 105.3, 68.0

# Bola
BALL_VMAX = 38.0            # m/s, limite fisico de um chute muito forte (faixa 35-38)
BALL_EPS = 1.0              # m, ruido de posicao do tracker (sensibilidade 0.5 / 1 / 2)
BALL_K = 30                 # observacoes anteriores consideradas no filtro (estavel de 20 a 60)
MAX_GAP_FRAMES = 12         # lacuna maxima para interpolar (0.5 s); comparar 6 / 12 / 25
MIN_BALL_SPEED_DIR = 1.0    # m/s, velocidade minima da bola para calcular mudanca de direcao

# Jogadores
PLAYER_VMAX = 12.0          # m/s, acima do sprint humano; saltos acima viram slot_break
DEDUP_DIST = 1.0            # m, jogadores do mesmo time a menos disso sao duplicata
STILL_SPEED = 0.5           # m/s, abaixo disso o jogador conta em n_still_*

# Orientacao
ORIENT_WIN = 25             # frames apos o pontape inicial usados para achar o sentido de ataque
# Valores publicados no FAQ/DATA, no referencial video_frame tático (índice zero);
# não foram derivados de ajustes nos cinco minutos públicos.
KICKOFF_FALLBACK = {1: 11_185, 2: 113_742}   # so para V0/conferencia

# Proximidade e campo
RADII = (2.0, 3.0, 5.0)     # m, contagens n_*_r2/r3/r5
PRESSURE_TAU = 3.0          # m, decaimento exponencial de pressao sum(exp(-d / tau))
PITCH_MARGIN = 1.0          # m, margem de out_of_pitch
