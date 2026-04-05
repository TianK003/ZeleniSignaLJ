# Zeleni SignaLJ

**Optimizacija semaforjev z umetno inteligenco za Ljubljano**

Arnes HackathON 2026 | Ekipa Ransomware

## Pregled

Agenti s spodbujevalnim ucenjem (neodvisni PPO), uceni v simulatorju SUMO, za optimizacijo krmiljenja semaforjev v ljubljanskem Bleiweisovem trikotniku — obmocju zlivanja Trzaske, Celovske in Dunajske ceste.

## Ciljno obmocje

Tri kljucna krizisca, ki tvorijo trikotnik:
- **Bleiweisova / Trzaska / Askerceva** (jugozahod)
- **Bleiweisova / Celovska / Tivolska** (severozahod)
- **Tivolska / Dunajska / Slovenska** (severovzhod)

## Podatki o obmocju

Podatki o cestnem omrezju so pridobljeni iz OpenStreetMap (ODbL licenca).

**Okvir (bounding box):**
| Stran | Koordinata |
|-------|------------|
| Sever | 46.05840 |
| Jug | 46.04540 |
| Zahod | 14.49385 |
| Vzhod | 14.50687 |

**Vir:** [OpenStreetMap Export](https://www.openstreetmap.org/export)

Podatke je mogoce ponovno prenesti z Overpass API:
```bash
wget -O data/osm/bleiweisova.osm \
  "https://overpass-api.de/api/map?bbox=14.49385,46.04540,14.50687,46.05840"
```

## Hiter zacetek

```bash
# 1. Namestitev SUMO
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update && sudo apt-get install -y sumo sumo-tools sumo-doc
export SUMO_HOME="/usr/share/sumo"

# 2. Ustvarjanje Python okolja
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Preverjanje namestitve
python -c "import sumo_rl; print('sumo-rl', sumo_rl.__version__)"

# 4. Ucenje (lokalno, majhen obseg)
python src/train.py

# 5. Evalvacija
python src/evaluate.py
```

## Struktura projekta

```
zeleni-signalj/
├── data/
│   ├── osm/          # Surovi OSM izvlecki
│   ├── networks/     # SUMO .net.xml datoteke
│   ├── routes/       # Prometno povprasevanje .rou.xml
│   └── gtfs/         # LPP avtobusni podatki (neobvezno)
├── src/
│   ├── train.py      # PPO ucni skript
│   ├── evaluate.py   # Evalvacija modela in primerjava KPI
│   └── custom_reward.py  # Prilagojene nagradne funkcije
├── hpc/
│   ├── traffic_rl.def    # Apptainer definicija vsebnika
│   └── submit_train.sh   # SLURM skripta za Vego
├── models/           # Shranjeni modeli (kontrolne tocke)
├── logs/             # Dnevniki ucenja
└── results/          # CSV datoteke z rezultati evalvacije
```

## Tehnologije

- **SUMO** — mikroskopski prometni simulator (DLR)
- **sumo-rl** — Gymnasium/PettingZoo ovoj za SUMO
- **stable-baselines3** — implementacija PPO algoritma
- **HPC Vega** (IZUM) — supracunalnik za ucenje z 32 paralelnimi okolji

## Ekipa

- Nik Jenic
- Tian Kljucanin
- Nace Omahen
- Masa Uhan

## Licenca

MIT
