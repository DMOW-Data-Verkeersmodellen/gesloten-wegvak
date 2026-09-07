# corvia

Toolkit voor het reconstrueren en valideren van verkeersvolumes op een
wegennetwerk op basis van beperkte sensormetingen. Gegeven een wegennetwerk
en een set van gedeeltelijke tellingen, balanceert en herleidt corvia
consistente verkeersstromen over het volledige netwerk en rapporteert hoe
goed de reconstructie overeenkomt met de onderliggende metingen.

**Track en upload nooit grote bestanden (zoals data) in git. Git houdt
verschillende versies van alle bestanden bij om aan version control te doen.
Eens de bestanden ooit zijn toegevoegd, kunnen ze dus nooit meer verwijderd
worden uit de geschiedenis van een git-project en zo kan het nodige geheugen
snel exploderen!**

## Concepten

Een korte begrippenlijst vóór de installatie, aangezien corvia zijn eigen
vocabularium heeft:

- **Network** — het wegennetwerk-model: nodes, links en de road sections die
  daaruit worden opgebouwd (via `NetworkBuilder`).
- **FlowStore** — houdt verkeersvolumes bij per road section, tijdstip en
  voertuigtype, met een onderscheid tussen directe sensormetingen
  (`SOURCE_OBS`) en waarden afgeleid via reconstructie.
- **Reconstructie** — het proces waarbij ontbrekende stromen worden aangevuld
  en verzoend over het netwerk, uitgaande van gedeeltelijke metingen.
  `FlowLinkBalancer` bewaakt het behoud van stroom tussen stroomopwaartse en
  stroomafwaartse secties; `FlowResolver` combineert deze tot één consistente
  schatting per sectie.
- **Validators** — controles op metingen en herleide stromen om outliers of
  inconsistenties te signaleren (bv. `ZScoreValidator`, `RelativeErrorValidator`,
  `TScoreValidator`).

## Installatie

Om dependency-conflicten te vermijden, werken we bij voorkeur in een
virtuele omgeving. Onderstaande instructies gebruiken (mini)conda.

### 1 — Open een terminal

Gebruik op Windows bij voorkeur **Git Bash** om met conda en git te werken.

### 2 — Maak een conda-omgeving aan en activeer deze

Controleer de vereiste Python-versie in `pyproject.toml` en pas indien nodig
aan. Zorg dat `pip` wordt meegenomen bij het aanmaken van de omgeving.

```bash
conda create -n corvia-env python>=3.11 pip
conda activate corvia-env
```

### 3 — Navigeer naar je projectmap

```bash
cd /path/to/software/directory
```

### 4 — Clone de repository

```bash
git clone https://github.com/DMOW-Data-Verkeersmodellen/gesloten-wegvak.git
```
of, met SSH geconfigureerd:
```bash
git clone git@github.com:DMOW-Data-Verkeersmodellen/gesloten-wegvak.git
```

Ga vervolgens de map van de repository binnen:
```bash
cd gesloten-wegvak
```

Vervolgens moet je nog de juiste versie activeren. Stabiele versies vind je op de `main` branch. Checkout de laatste versie met tag `latest-version`. Let op, **voeg zelf geen commit toe** aan deze branch. Wil je een nieuwe feature helpen ontwikkelen, maak dan een nieuwe branch vanaf `main`waar je alles aanpast en vraag later een `pull request` aan.
```bash
git checkout latest-version
```

### 5 — Installeer de package en de dependencies

```bash
pip install -e .
```

De `-e` flag installeert het project "editable", zodat lokale codewijzigingen
(bv. door van branch te wisselen of updates op te halen) meteen worden
meegenomen zonder herinstallatie.

Wil je bijdragen aan de code, installeer dan ook de optionele
development-dependencies en voer het dev setup-script uit:
```bash
pip install -e ".[dev]"
python setup_dev.py
```

### 6 — Verifieer de installatie (optioneel)

```bash
pip list
```
`corvia` moet in de lijst staan, met een verwijzing naar je lokale bronmap.

## Snelstart

Een minimaal end-to-end voorbeeld — een netwerk inladen, verkeersmetingen
inladen, en consistente stromen over het netwerk herleiden:

```python
from corvia.framework.builder import NetworkBuilder
from corvia.network_states.flow import FlowStore
from corvia.reconstruction.flow_link_balancer import FlowLinkBalancer
from corvia.engine.flow_resolver import FlowResolver
from corvia.loaders.network import VCNetworkLoader
from corvia.loaders.flow_data_loader import VCFlowDataLoader

# 1. Bouw het netwerk op uit een geopackage
loader = VCNetworkLoader("network.gpkg")
links_data, nodes_data, counts_data, crs = loader.load()

network = NetworkBuilder.from_data(
    name_network="voorbeeld-netwerk",
    links_data=links_data,
    nodes_data=nodes_data,
    counts_data=counts_data,
    crs=crs,
)

# 2. Laad de verkeersmetingen in
flow_loader = VCFlowDataLoader(
    csv_filenames=["counts.csv"],
    agg_freq="1D",
    start="2026-03-25",
    end="2026-03-26",
    vehicle_types=["TOTAL"],
)
store: FlowStore = flow_loader.load()

# 3. Evalueer het gesloten wegvak over het netwerk
reconstructors = [
    FlowLinkBalancer(direction="upstream", min_obs_degree=1, max_link_degree=3),
    FlowLinkBalancer(direction="downstream", min_obs_degree=1, max_link_degree=3),
]
resolver = FlowResolver(network=network, reconstructors=reconstructors)
resolved_store = resolver.resolve(store)

resolved_store.dataframe.head()
```

Voor een volledigere doorloop met validatie en visualisatie, zie `examples/`.

## Data en bestandsbeheer

De loaders van corvia verwachten een specifiek inputformaat:

- **Netwerk**: een GeoPackage (`.gpkg`) met link-, node- en (optioneel)
  sensor-lagen.
- **Verkeersmetingen**: CSV-bestanden met telgegevens per sensorlocatie en
  tijdstip.

Momenteel wordt enkel data in het formaat van het Vlaamse Verkeerscentrum ondersteund.
Zie `examples/data/` voor voorbeelddata en de bijhorende structuur.

Data wordt niet meegeleverd met deze repository
**Upload nooit ruwe of verwerkte databestanden naar git.**
Zet je data in een aparte map buiten de repository, in een map 'data/' in de repository.
Wil je je eigen mappenstructuur binnen de repository, vergeet dan niet om deze mappen/files
toe te voegen aan de git-ignorerenlijst (`.gitignore`) zodat ze niet in git terecht komen.


## Voorbeelden

De map `examples/` bevat notebooks die de kernwerking van de package
demonstreren aan de hand van synthetische of voorbeelddata in
`examples/data/`.

Om ze te openen:
```bash
jupyter notebook
```

**Overschrijf deze voorbeeld-notebooks nooit met echte studiedata en upload
nooit verwerkte output van echte analyses naar git.**

## Ontwikkeling

Wil je bijdragen:
- Maak een nieuwe branch aan vanaf `main` voor elke wijziging; vraag daarna
  een pull request aan.
- Voer `python setup_dev.py` uit na het installeren van de
  development-dependencies — dit houdt notebook-metadata en andere
  diff-ruis buiten je commits.

<!-- TODO: licentie, contactpersoon/beheerder en citatie-info toevoegen
     indien relevant. -->