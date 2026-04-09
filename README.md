# Zeleni SignaLJ

**Optimizacija semaforjev z umetno inteligenco za Ljubljano**

Arnes HackathON 2026 | Ekipa Ransomware

[![DOI](https://zenodo.org/badge/1202274690.svg)](https://doi.org/10.5281/zenodo.19462105)

## Pregled

Agenti s spodbujevalnim učenjem (neodvisni PPO z deljeno politiko), učeni v simulatorju SUMO, za optimizacijo krmiljenja semaforjev v ljubljanskem Bleiweisovem trikotniku — območju zlivanja Tržaške, Celovške in Dunajske ceste.

## Ciljno območje

Zaradi kompleksnosti optimizacije večjega območja smo se osredotočili na optimizacijo le 5 križišč. Želimo narediti prototip, ki dokaže, da se promet lahko optimizira, seveda bi pa v realnosti naredili optimizacijo na nivoju območja.

Opazujemo sledeča križišča:
1. **Tivolska / Slovenska / Dunajska / Trg OF** (Kolodvor)
2. **Bleiweisova / Tivolska / Celovška / Gosposvetska** (Pivovarna)
3. **Slovenska / Gosposvetska / Dalmatinova** (Slovenska)
4. **Aškerčeva / Prešernova / Groharjeva** (Tržaška)
5. **Aškerčeva / Zoisova / Slovenska / Barjanska** (Aškerčeva)

![Slika izbranih križišč](./data/media/Observed_intersections.png)

Preostalih 32 semaforjev v omrežju ohranja izvorne SUMO programe (fiksne cikle iz .net.xml), kar zagotavlja pošteno primerjavo z bazno linijo.

## Podatki o območju

Podatki o cestnem omrežju so pridobljeni iz OpenStreetMap (ODbL licenca).

**Okvir (bounding box):**
| Stran | Koordinata |
|-------|------------|
| Sever | 46.05840 |
| Jug | 46.04540 |
| Zahod | 14.49385 |
| Vzhod | 14.50687 |

**Vir:** [OpenStreetMap Export](https://www.openstreetmap.org/export)

Podatke je mogoče ponovno prenesti z Overpass API:
```bash
wget -O data/osm/map.osm \
  "https://overpass-api.de/api/map?bbox=14.49385,46.04540,14.50687,46.05840"
```

## Hiter začetek

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

# 3b. Pospešitev SUMO (5-8x hitrejše — uporabi libsumo namesto TraCI socketa)
export LIBSUMO_AS_TRACI=1

# 4a. Generiranje enakomerne prometne obremenitve (za testiranje)
python src/generate_demand.py --profile uniform --duration 3600 --peak_vph 800

# 4b. Generiranje prometnih scenarijev koničnih ur (za resno učenje)
python src/generate_demand.py --scenario all
# Ustvari: routes_morning_rush.rou.xml, routes_evening_rush.rou.xml, routes_offpeak.rou.xml

# 5. Učenje — enaka prometna obremenitev (lokalno, 50 epizod ~ 4 min)
python src/experiment.py --episode_count 50 --tag local_50ep

# 5b. Učenje — scenarij jutranje konice (priporočeno za produkcijo)
python src/experiment.py --scenario morning_rush --episode_count 50 --tag jutro_50ep

# 5c. Napredno učenje po celem dnevu (Curriculum)
python src/experiment.py --episode_count 50 --curriculum --tag napredno_ucenje

# 6. Primerjava rezultatov in nadzorna plošča
python src/experiment.py --compare_only
python src/dashboard.py
# Odpri results/dashboard.html v brskalniku

# 7. Evalvacija modela po vseh scenarijih koničnih ur
python src/evaluate.py --model results/experiments/XXXXX/ppo_shared_policy.zip
python src/evaluate.py --model models/ppo_morning_rush_final.zip --scenario morning_rush
```

## Generiranje prometnega povpraševanja (`generate_demand.py`)

Dva načina uporabe: `--profile` za enakomerno testno obremenitev ali `--scenario` za realistične scenarije koničnih ur iz dvokoničnega 24h matematičnega modela (8:00 jutranja + 16:00 večerna konica). Scenariji modelirajo tudi smer prometnega toka: jutranja konica ima 70 % prometa usmerjenega v center, večerna konica pa 70 % iz centra.

```bash
# Enakomerna obremenitev (za dimne teste)
python src/generate_demand.py --profile uniform --duration 3600 --peak_vph 800

# Vsi scenariji koničnih ur naenkrat (priporočeno za resno učenje)
python src/generate_demand.py --scenario all

# Posamezni scenariji
python src/generate_demand.py --scenario morning_rush   # 06:00-10:00, 4h, bimodalna krivulja
python src/generate_demand.py --scenario evening_rush   # 14:00-18:00, 4h, bimodalna krivulja
python src/generate_demand.py --scenario offpeak        # 12:00-13:00, 1h, referenčni scenarij
```

| Parameter | Opis |
|-----------|------|
| `--profile uniform` | Enakomerna obremenitev (za dimne teste) |
| `--scenario` | `morning_rush` / `evening_rush` / `offpeak` / `all` |
| `--duration` | Trajanje simulacije v sekundah (samo `--profile` način) |
| `--peak_vph` | Koničen pretok v vozilih/uro (samo `--profile` način) |
| `--fringe_factor` | Verjetnost izvorov na robovih (privzeto 5.0) |

| Scenarij | Okno | Trajanje | Datoteka |
|----------|------|----------|---------|
| `morning_rush` | 06:00-10:00 | 4 ure | `routes_morning_rush.rou.xml` |
| `evening_rush` | 14:00-18:00 | 4 ure | `routes_evening_rush.rou.xml` |
| `offpeak` | 12:00-13:00 | 1 ura | `routes_offpeak.rou.xml` |

## Učenje (`experiment.py`)

Zažene bazno linijo → učenje → evalvacija → shranjevanje v enem klicu.

```bash
# Osnovna uporaba
python src/experiment.py --episode_count 50 --tag local_50ep

# Scenariji koničnih ur (zahteva routes_morning_rush.rou.xml)
python src/experiment.py --scenario morning_rush --episode_count 100 --tag jutro_100ep
python src/experiment.py --scenario evening_rush --episode_count 100 --tag vecer_100ep

# Nadaljevanje iz kontrolne točke
python src/experiment.py --episode_count 100 --resume results/experiments/XXXXX/ppo_shared_policy.zip

# Curriculum learning — naključni urni rezini čez cel dan
python src/experiment.py --episode_count 200 --curriculum --tag curriculum_200ep

# Curriculum z beleženjem napredka (primerja RL vs bazna linija vsako epizodo)
python src/experiment.py --episode_count 100 --curriculum --log_curriculum --tag curriculum_log

# Vzporedno učenje na več CPE (za HPC)
python src/experiment.py --episode_count 500 --num_cpus 4 --tag hpc_500ep

# Časovna omejitev (npr. 1 ura učenja)
python src/experiment.py --max_hours 1.0 --tag 1h_local

# Surovi timesteps (namesto epizod)
python src/experiment.py --total_timesteps 180000 --tag raw_50ep

# Samo primerjava obstoječih eksperimentov
python src/experiment.py --compare_only
```

| Zastavica | Opis |
|-----------|------|
| `--episode_count N` | Število polnih epizod (1 epizoda = 3600 SB3 korakov) |
| `--total_timesteps N` | Surovo število SB3 korakov |
| `--max_hours H` | Zaustavitev po H urah (stenski čas) |
| `--scenario` | `uniform` / `morning_rush` / `evening_rush` / `offpeak` |
| `--curriculum` | Naključni urni rezini 24h krivulje |
| `--log_curriculum` | Beleženje RL vs. bazna linija vsako epizodo |
| `--num_cpus N` | Vzporedni SUMO procesi (za HPC) |
| `--resume PATH` | Nadaljevanje iz obstoječe kontrolne točke |
| `--tag OZNAKA` | Oznaka eksperimenta (za identifikacijo) |

## Evalvacija

Primerja naučen model z bazno linijo (fiksni časi) po treh scenarijih.

```bash
# Evalvacija čez vse tri scenarije
python src/evaluate.py --model results/experiments/XXXXX/ppo_shared_policy.zip

# Samo en scenarij
python src/evaluate.py --model models/ppo_morning_rush_final.zip --scenario morning_rush

# Z vizualizacijo SUMO GUI
python src/evaluate.py --model models/ppo_morning_rush_final.zip --gui

# Samo bazna linija (brez modela)
python src/evaluate.py
```

Izhod: `results/rush_hour_comparison.csv` (primerjava po scenarijih) + `results/comparison_summary.csv` (po križiščih).

## Načrtovalec (Schedule Controller)

`schedule_controller.py` implementira produkcijsko strategijo: RL agent se aktivira samo v koničnih urah, sicer tečejo izvorni SUMO programi.

| Čas | Način |
|-----|-------|
| 00:00-06:00 | Fiksni čas (noč) |
| 06:00-10:00 | RL agent (jutranja konica) |
| 10:00-14:00 | Fiksni čas (poldnevi) |
| 14:00-18:00 | RL agent (večerna konica) |
| 18:00-24:00 | Fiksni čas (večer/noč) |

```python
from schedule_controller import ScheduleController
ctrl = ScheduleController(
    model_morning="models/ppo_morning_rush_final.zip",
    model_evening="models/ppo_evening_rush_final.zip"
)
ctrl.print_schedule()
mode = ctrl.get_mode(hour=7.5)   # -> "rl_morning"
```

## Parametri simulacije

| Parameter | Vrednost | Opis |
|-----------|----------|------|
| `num_seconds` | 3600 + 600 | Trajanje epizode: 600s ogrevanje + 1h RL |
| `delta_time` | 5 | Sekunde med odločitvami agenta (frekvenca akcij) |
| `yellow_time` | 2 | Trajanje rumene faze med preklopom zelene |
| `min_green` | 10 | Minimalen čas zelene faze pred preklopom |
| `max_green` | 90 | Maksimalen čas zelene faze pred ponovnim odločanjem |
| `reward_fn` | "queue" | Nagrada = negativno število ustavljenih vozil na korak |
| `WARMUP_SECONDS` | 600 | 10 min mehanske SUMO simulacije pred RL prevzemom |

## PPO hiperparametri

| Parameter | Vrednost | Opis |
|-----------|----------|------|
| `learning_rate` | 0.001 | Korak gradientnega posodabljanja |
| `n_steps` | 720 | Korakov na agenta pred PPO posodobitvijo (= 1 celotna epizoda) |
| `batch_size` | 180 | Velikost mini-serije za gradient (3600 / 180 = 20 serij) |
| `n_epochs` | 10 | Število prehodov čez zbiralnik na posodobitev |
| `gamma` | 0.99 | Diskontni faktor (0=kratkovidno, 1=neskončen horizont) |
| `gae_lambda` | 0.95 | GAE glajenje med pristranskostjo in varianco |
| `ent_coef` | 0.05 | Entropijski bonus — spodbuja raziskovanje |
| `clip_range` | 0.2 | PPO obrezovanje: omejuje spremembo politike na posodobitev |

## Napredno učenje (Curriculum Learning)

Z uporabo zastavice `--curriculum` algoritem naključno vzorči različne ure dneva. Sistem izračuna prometni pretok iz dvokoničnega matematičnega modela (`demand_math.get_vph`): konici ob 8:00 in 16:00, ponoči prazne ceste. Skupno 40.000 vozil/dan (nastavljivo v `config.py` kot `TOTAL_DAILY_CARS`).

Model vsako epizodo vidi drugačen scenarij in se tako nauči splošnih pravil za različne prometne obremenitve. Z `--log_curriculum` dobimo podroben zapis napredka (primerjava RL vs. bazna linija za vsako epizodo) v `curriculum_progress.txt`.

## Razumevanje korakov (timesteps)

En "korak" (timestep) v SB3 = 5 sekund simuliranega prometa za 1 semafor. Ker je 5 semaforjev vektoriziranih prek SuperSuit, SB3 prišteje 5 korakov na vsak SUMO korak.

```
1 epizoda = num_seconds / delta_time * num_agents
         = 3600 / 5 * 5 = 3600 SB3 korakov

n_steps = 720 → 720 * 5 agentov = 3600 korakov = točno 1 epizoda na PPO posodobitev
```

| `--episode_count` | SB3 koraki | Polnih epizod | PPO posodobitev |
|-------------------|------------|---------------|-----------------|
| 10 | 36.000 | 10 | 10 |
| 50 | 180.000 | 50 | 50 |
| 100 | 360.000 | 100 | 100 |
| 500 | 1.800.000 | 500 | 500 |

## Arhitektura cevovoda

```mermaid
flowchart TD
    subgraph DATA ["Podatki"]
        OSM["data/osm/bleiweisova.osm\nOpenStreetMap izvoz"]
        NET["data/networks/ljubljana.net.xml\nSUMO omrežje"]
        CFG["data/networks/ljubljana.sumocfg\nSUMO konfiguracija"]
        ROU["data/routes/routes.rou.xml\nPrometno povpraševanje"]
        ROURUSHHOUR["data/routes/routes_*_rush.rou.xml\nKonični scenariji"]
    end

    subgraph SRC_SETUP ["Priprava okolja"]
        CONFIG["src/config.py\nTLS IDs + imena križišč"]
        DEMANDMATH["src/demand_math.py\ndvokonična 24h krivulja"]
        GENDEM["src/generate_demand.py\nuniform + scenariji koničnih ur"]
        TLSPROG["src/tls_programs.py\nobnovitev ne-ciljnih programov"]
        AGENTFIL["src/agent_filter.py\nPettingZoo ovoj → 5 križišč"]
        CUSTOMREW["src/custom_reward.py\nprilagojena nagradna funkcija"]
    end

    subgraph SRC_TRAIN ["Učenje"]
        EXPERIMENT["src/experiment.py\nceloten cevovod"]
    end

    subgraph SRC_EVAL ["Evalvacija & Analiza"]
        EVALUATE["src/evaluate.py\nKPI primerjava po scenarijih"]
        EVALHELPER["src/eval_helper.py\npomočnik za podprocese"]
        ANALYZE["src/analyze_sim.py\nteleporti, pretoki, statistika"]
        DASHBOARD["src/dashboard.py\nHTML nadzorna plošča"]
        SCHEDCTRL["src/schedule_controller.py\načrtovalec konic RL/fiksni"]
    end

    subgraph HPC ["HPC — Vega superračunalnik"]
        DEF["hpc/traffic_rl.def\nApptainer vsebnik"]
        SLURM["hpc/sweep/submit_train.sh\nSLURM opravilo"]
    end

    subgraph RESULTS ["Rezultati"]
        MODEL["results/experiments/ID/\nppo_shared_policy.zip"]
        TRAINLOG["results/experiments/ID/\ntraining_log.csv"]
        RESCSV["results/experiments/ID/\nresults.csv"]
        METAJSON["results/experiments/ID/\nmeta.json"]
        RUSHCSV["results/rush_hour_comparison.csv"]
        DASH["results/dashboard.html"]
    end

    OSM -->|"netconvert"| NET
    NET --> CFG
    DEMANDMATH --> GENDEM
    GENDEM -->|"generira povpraševanje"| ROU
    GENDEM -->|"generira konično povpraševanje"| ROURUSHHOUR
    ROU --> CFG
    ROURUSHHOUR --> CFG

    CONFIG --> AGENTFIL
    CONFIG --> TLSPROG
    NET --> TLSPROG

    CFG --> EXPERIMENT
    AGENTFIL --> EXPERIMENT
    TLSPROG --> EXPERIMENT
    CUSTOMREW --> EXPERIMENT

    EXPERIMENT -->|"bazna linija"| RESCSV
    EXPERIMENT -->|"model"| MODEL
    EXPERIMENT -->|"dnevnik"| TRAINLOG
    EXPERIMENT -->|"metapodatki"| METAJSON

    MODEL --> EVALUATE
    ROURUSHHOUR --> EVALUATE
    EVALUATE -->|"KPI po scenarijih"| RUSHCSV
    EVALHELPER --> EXPERIMENT

    RESCSV --> DASHBOARD
    RUSHCSV --> DASHBOARD
    ANALYZE --> DASHBOARD
    DASHBOARD --> DASH

    MODEL --> SCHEDCTRL

    DEF --> SLURM
    SLURM -->|"zažene na Vegi"| EXPERIMENT
```

## Struktura projekta

```
zeleni-signalj/
├── data/
│   ├── osm/              # Surovi OSM izvlečki
│   ├── networks/         # SUMO .net.xml datoteke
│   ├── routes/           # Prometno povpraševanje .rou.xml
│   └── gtfs/             # LPP avtobusni podatki (neobvezno)
├── src/
│   ├── experiment.py         # Celoten eksperiment (bazna linija → učenje → evalvacija)
│   ├── evaluate.py           # Multi-scenarij evalvacija (morning/evening/offpeak)
│   ├── config.py             # ID-ji križišč, parametri simulacije, PPO hiperparametri
│   ├── agent_filter.py       # PettingZoo ovoj za filtriranje na 5 križišč
│   ├── tls_programs.py       # Obnovitev izvornih SUMO programov za neopazovana križišča
│   ├── custom_reward.py      # Prilagojene nagradne funkcije
│   ├── demand_math.py        # Dvokonična 24h prometna krivulja (get_vph)
│   ├── generate_demand.py    # Generiranje povpraševanja (uniform + scenariji koničnih ur)
│   ├── schedule_controller.py  # Načrtovalec: RL v konicah, fiksni čas drugače
│   ├── eval_helper.py        # Pomočnik za evalvacijo v podprocesih (curriculum)
│   ├── analyze_sim.py        # Analiza SUMO izhoda (teleporti, pretoki)
│   └── dashboard.py          # Generiranje HTML nadzorne plošče
├── hpc/
│   ├── common.sh             # Skupna HPC nastavitev (venv, SUMO_HOME)
│   ├── traffic_rl.def        # Apptainer definicija vsebnika
│   ├── sweep/                # Hiperparametrsko iskanje (30 SLURM skript)
│   │   ├── generate_jobs.py  # Generator SLURM skript
│   │   ├── submit_all.sh     # Oddaja vseh sweep opravil
│   │   └── *.slurm           # Posamezne sweep skripte
│   └── statistical-test/     # 24h statistično testiranje mega-politik
│       ├── generate_mega_jobs.py  # Generator 10 SLURM skript
│       ├── submit_all.sh          # Oddaja vseh mega-testov
│       └── mega_*.slurm           # 9 mega-politik + 1 bazna linija
├── models/               # Shranjeni modeli (kontrolne točke)
├── logs/                 # Dnevniki učenja
└── results/
    ├── experiments/      # Rezultati po eksperimentih (meta.json, results.csv, model)
    ├── statistical-test/ # Rezultati 24h statisticnih testov (50 ponovitev x 10 pogojev)
    ├── rush_hour_comparison.csv  # KPI primerjava po scenarijih koničnih ur
    ├── comparison_summary.csv    # KPI po posameznih križiščih
    └── dashboard.html    # Interaktivna nadzorna plošča
```

## Tehnologije

- **SUMO 1.26.0** — mikroskopski prometni simulator (DLR)
- **sumo-rl 1.4.5** — PettingZoo ovoj za SUMO (več-agentno)
- **stable-baselines3 2.8.0** — implementacija PPO algoritma
- **SuperSuit 3.9+** — vektorizacija PettingZoo okolja za SB3
- **HPC Vega** (IZUM) — superračunalnik za učenje z večjim številom epizod

## HPC eksperimenti

30 pripravljenih SLURM skript v `hpc/sweep/` za sistematično iskanje najboljše konfiguracije.

**Matrika eksperimentov:**
- 3 nagradne funkcije: `queue`, `pressure`, `diff-waiting-time`
- 2 hitrosti učenja: `1e-3`, `3e-4`
- 3 dolžine učenja: 100, 250, 500 epizod
- Entropy annealing variante (0.05 -> 0.01)
- Curriculum learning variante

```bash
# Oddaj vse eksperimente
bash hpc/sweep/submit_all.sh

# Oddaj samo pressure variante
bash hpc/sweep/submit_all.sh pressure

# Oddaj samo 100-epizodne eksperimente
bash hpc/sweep/submit_all.sh 100ep

# Oddaj samo entropy annealing variante
bash hpc/sweep/submit_all.sh entanneal

# Spremljaj status
squeue -u $USER
```

## Izbor najboljših politik in mega-politike

Po izvedbi 24 eksperimentov na HPC (3 nagradne funkcije x 2 hitrosti učenja x 2 scenarija + entropy annealing variante) smo na podlagi nadzorne plošče izbrali **3 najboljše politike za jutranjo konico** in **3 najboljše za vecerno konico** glede na izboljsanje % nad bazno linijo (fiksni casi).

### Najboljse politike — jutranja konica (06:00-10:00)

| Rang | Nagradna funkcija | Hitrost ucenja | Izboljsanje |
|------|-------------------|----------------|-------------|
| M1 | diff-waiting-time | 1e-3 | **+18.2%** |
| M2 | pressure | 1e-3 | **+17.2%** |
| M3 | queue (privzeta) | 3e-4 | **+17.1%** |

### Najboljse politike — vecerna konica (14:00-18:00)

| Rang | Nagradna funkcija | Hitrost ucenja | Entropy annealing | Izboljsanje |
|------|-------------------|----------------|-------------------|-------------|
| E1 | pressure | 1e-3 | da | **+15.1%** |
| E2 | diff-waiting-time | 1e-3 | da | **+15.0%** |
| E3 | pressure | 3e-4 | da | **+12.9%** |

### Mega-politike (3 x 3 = 9 kombinacij)

Mega-politika kombinira eno jutranjo in eno vecerno politiko z nacrtovalcem (Schedule Controller):
- **V konicnih urah** (06:00-10:00, 14:00-18:00): RL agent (PPO model) krmili 5 ciljnih kriziscc
- **Izven konic** (10:00-14:00, 18:00-06:00): vsa kriziscca teccejo na izvornih SUMO fiksnih ccasih

To daje 9 mega-politik: M1E1, M1E2, M1E3, M2E1, M2E2, M2E3, M3E1, M3E2, M3E3.

## Statisticcno testiranje (24h simulacije)

Za statisticcno veljavno primerjavo mega-politik z bazno linijo izvajamo **50 neodvisnih ponovitev** vsake mega-politike na polnih 24-urnih simulacijah.

### Zasnova testa

- **10 pogojev:** 9 mega-politik + 1 bazna linija (sami fiksni ccasi)
- **50 ponovitev** na pogoj, vsaka z unikatnim SUMO semenom (1-50)
- **Iste prometne poti** za vse ponovitve (`routes_full_day.rou.xml`) — izoliramo uccinek strategije krmiljenja od variabilnosti povprasevanja
- **Polni 24h dvokonniccni profil** z usmerjeno asimetrijo (70% vhodni promet zjutraj, 70% izhodni zvecer)

### Merjene metrike (na ponovitev)

- Skupna kumulativna nagrada (vsa kriziscca, vsi ccasovni koraki)
- Nagrada, povpreccna ccakalna vrsta in ccakalni ccas **po krizisccu** (5 kriziscc)
- Razdelitev **po ccasovnem oknu** (nocc, jutranja konica, dnevni obok, veccerna konica, veccerni obok)
- Skupni teleporti, vozila odsla/prispela

### Statisticcna analiza

Iz 50 ponovitev izracunamo za vsako mega-politiko:
- **Povpreccje, mediana, standardni odklon**
- **95% interval zaupanja** (CI = 1.96 * std / sqrt(n))
- **Welchov t-test** proti bazni liniji (neenake variance)
- **Mann-Whitney U test** (neparametriccni, brez predpostavke o porazdelitvi)
- **Cohenov d** (velikost uccinka)

Mega-politika je statisticcno znaccilno boljsa od bazne linije, cce je p < 0.05 pri obeh testih.

### Zagon

```bash
# 1. Generiraj 24h prometno povprasevanje (enkrat)
python src/generate_demand.py --scenario full_day

# 2. Generiraj SLURM skripte (10 opravil)
python hpc/statistical-test/generate_mega_jobs.py

# 3. Oddaj vse na HPC
bash hpc/statistical-test/submit_all.sh

# 4. Rezultati: results/statistical-test/{M1E1,...,baseline}/summary.csv
```

### Ocenjen ccas na HPC

| Operacija | Ccas |
|-----------|------|
| Ena 24h simulacija | ~10-20 min |
| 50 ponovitev (50 vzporednih delavcev, 64 CPE) | ~15-30 min |
| Skupaj na opravilo (z varnostno rezervo) | 4h (zahtevano) |
| Vseh 10 opravil (vzporedno) | ~1h stenske ure |

## Vizualizacija in demo (TODO)

Načrt za pripravo končne predstavitve za sodnike hackathona.

### Rezultati in nadzorna plošča
- [ ] Zaženi 5-10+ eksperimentov na HPC z različnimi nagradnimi funkcijami
- [ ] Prenesi rezultate iz Vege (`results/experiments/`)
- [ ] Generiraj nadzorno ploščo: `python src/dashboard.py`
- [ ] Preveri zavihke: primerjava, križišča, učne krivulje, hiperparametri

### Vizualizacija prometa
- [ ] Zaženi najboljši model v SUMO GUI: `python src/evaluate.py --gui --model <pot_do_modela>.zip --scenario morning_rush`
- [ ] Posnimi 30-sekundni video zaslona SUMO GUI kot rezervo (ffmpeg ali OBS)
- [ ] Priprava toplotne karte prometnih čakalnih vrst (baseline vs. RL, po korakih)

### Predstavitev
- [ ] Kontekstna slika: `data/media/Observed_intersections.png` — 5 križišč na zemljevidu
- [ ] Arhitekturni diagram: Mermaid diagram iz README-ja
- [ ] Nadzorna plošča v živo: odpri `results/dashboard.html`, pokaži zavihke
- [ ] SUMO GUI demo v živo ALI predposneti video
- [ ] Zaključni diapozitiv: tabela scenarij x nagradna funkcija x izboljšanje %

### Poročilo
- [ ] Metodologija: IPPO z deljeno politiko, 5 križišč, SUMO simulator
- [ ] Rezultati: tabele in grafi iz nadzorne plošče
- [ ] Analiza: katera križišča se izboljšajo, katera ne, zakaj
- [ ] Zaključek: primerjava nagradnih funkcij, priporočila za produkcijo

## Ekipa

- Nik Jenic
- Tian Kljucanin
- Nace Omahen
- Masa Uhan
