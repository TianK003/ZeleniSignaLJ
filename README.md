# Zeleni SignaLJ

**Optimizacija semaforjev z umetno inteligenco za Ljubljano**

Arnes HackathON 2026 | Ekipa Ransomware

## Pregled

Agenti s spodbujevalnim ucenjem (neodvisni PPO z deljeno politiko), uceni v simulatorju SUMO, za optimizacijo krmiljenja semaforjev v ljubljanskem Bleiweisovem trikotniku — obmocju zlivanja Trzaske, Celovske in Dunajske ceste.

## Ciljno obmocje

Zaradi kompleksnosti optimizacije vecjega obmocja smo se osredotocili na optimizacijo le 5 krizisc. Zelimo narediti prototip, ki dokaze, da se promet lahko optimizira, seveda bi pa v realnosti naredili optimizacijo na nivoju obmocja.

Opazujemo sledeca krizisca:
1. **Tivolska / Slovenska / Dunajska / Trg OF** (Kolodvor)
2. **Bleiweisova / Tivolska / Celovska / Gosposvetska** (Pivovarna)
3. **Slovenska / Gosposvetska / Dalmatinova** (Slovenska)
4. **Askerceva / Presernova / Groharjeva** (Trzaska)
5. **Askerceva / Zoisova / Slovenska / Barjanska** (Askerceva)

![Slika izbranih krizisc](./data/media/Observed_intersections.png)

Preostalih 32 semaforjev v omrezju ohranja izvorne SUMO programe (fiksne cikle iz .net.xml), kar zagotavlja posteno primerjavo z bazno linijo.

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
wget -O data/osm/map.osm \
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

# 4. Generiranje prometa
python src/generate_demand.py --profile uniform --duration 3600 --peak_vph 800

# 5. Ucenje (lokalno, 50 epizod ~ 4 min)
python src/experiment.py --episode_count 50 --tag local_50ep

# 6. Primerjava rezultatov in nadzorna plosca
python src/experiment.py --compare_only
python src/dashboard.py
# Odpri results/dashboard.html v brskalniku

# 7. Evalvacija specificnega modela
python src/evaluate.py --model results/experiments/XXXXX/ppo_shared_policy.zip
```

## Parametri simulacije

| Parameter | Vrednost | Opis |
|-----------|----------|------|
| `num_seconds` | 3600 | Trajanje ene epizode simulacije (1 ura) |
| `delta_time` | 5 | Sekunde med odlocitvami agenta (frekvenca akcij) |
| `yellow_time` | 2 | Trajanje rumene faze med preklopom zelene |
| `min_green` | 10 | Minimalen cas zelene faze pred preklopom |
| `max_green` | 90 | Maksimalen cas zelene faze pred ponovnim odlocanjem |
| `reward_fn` | "queue" | Nagrada = negativno stevilo ustavljenih vozil na korak |

## PPO hiperparametri

| Parameter | Vrednost | Opis |
|-----------|----------|------|
| `learning_rate` | 0.001 | Korak gradientnega posodabljanja |
| `n_steps` | 720 | Korakov na agenta pred PPO posodobitvijo (= 1 celotna epizoda) |
| `batch_size` | 180 | Velikost mini-serije za gradient (3600 / 180 = 20 serij) |
| `n_epochs` | 10 | Stevilo prehodov cez zbiralnik na posodobitev |
| `gamma` | 0.99 | Diskontni faktor (0=kratkovidno, 1=neskoncen horizont) |
| `gae_lambda` | 0.95 | GAE glajenje med pristranskostjo in varianco |
| `ent_coef` | 0.05 | Entropijski bonus — spodbuja raziskovanje |
| `clip_range` | 0.2 | PPO obrezovanje: omejuje spremembo politike na posodobitev |

## Razumevanje korakov (timesteps)

En "korak" (timestep) v SB3 = 5 sekund simuliranega prometa za 1 semafor. Ker je 5 semaforjev vektoriziranih prek SuperSuit, SB3 pristeje 5 korakov na vsak SUMO korak.

```
1 epizoda = num_seconds / delta_time * num_agents
         = 3600 / 5 * 5 = 3600 SB3 korakov

n_steps = 720 → 720 * 5 agentov = 3600 korakov = tocno 1 epizoda na PPO posodobitev
```

| `--episode_count` | SB3 koraki | Polnih epizod | PPO posodobitev |
|-------------------|------------|---------------|-----------------|
| 10 | 36.000 | 10 | 10 |
| 50 | 180.000 | 50 | 50 |
| 100 | 360.000 | 100 | 100 |
| 500 | 1.800.000 | 500 | 500 |

## Struktura projekta

```
zeleni-signalj/
├── data/
│   ├── osm/              # Surovi OSM izvlecki
│   ├── networks/         # SUMO .net.xml datoteke
│   ├── routes/           # Prometno povprasevanje .rou.xml
│   └── gtfs/             # LPP avtobusni podatki (neobvezno)
├── src/
│   ├── experiment.py     # Celoten eksperiment (bazna linija → ucenje → evalvacija)
│   ├── train.py          # PPO ucni skript (samostojno)
│   ├── evaluate.py       # Evalvacija modela in primerjava KPI
│   ├── config.py         # ID-ji krizisc in imena
│   ├── agent_filter.py   # PettingZoo ovoj za filtriranje na 5 krizisc
│   ├── tls_programs.py   # Obnovitev izvornih SUMO programov za neopazovana krizisca
│   ├── custom_reward.py  # Prilagojene nagradne funkcije
│   ├── generate_demand.py # Generiranje prometnega povprasevanja
│   ├── analyze_sim.py    # Analiza SUMO izhoda (teleporti, pretoki)
│   └── dashboard.py      # Generiranje HTML nadzorne plosce
├── hpc/
│   ├── traffic_rl.def    # Apptainer definicija vsebnika
│   └── submit_train.sh   # SLURM skripta za Vego
├── models/               # Shranjeni modeli (kontrolne tocke)
├── logs/                 # Dnevniki ucenja
└── results/
    ├── experiments/      # Rezultati po eksperimentih (meta.json, results.csv, model)
    └── dashboard.html    # Interaktivna nadzorna plosca
```

## Tehnologije

- **SUMO 1.26.0** — mikroskopski prometni simulator (DLR)
- **sumo-rl 1.4.5** — PettingZoo ovoj za SUMO (vec-agentno)
- **stable-baselines3 2.8.0** — implementacija PPO algoritma
- **SuperSuit 3.9+** — vektorizacija PettingZoo okolja za SB3
- **HPC Vega** (IZUM) — supracunalnik za ucenje z vecjim stevilom epizod

## Ekipa

- Nik Jenic
- Tian Kljucanin
- Nace Omahen
- Masa Uhan
