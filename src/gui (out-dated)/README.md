# Link en Linkketen Visualizer

Een grafische gebruikersinterface voor het visualiseren van links en linkketens op een interactieve Leaflet kaart.

## Installatie

1. Installeer de vereiste dependencies:
```bash
pip install -r requirements.txt
```

2. Start de applicatie:
```bash
python main.py
```

## Gebruik

1. **Laad Test Data**: Klik op deze knop om voorbeelddata te laden met links en hun intensiteiten
2. **Maak Linkketens**: Creëert linkketens uit de geladen links
3. **Visualiseer op Kaart**: Toont de links en linkketens op een interactieve kaart

## Functies

- **Interactieve kaart**: Gebaseerd op OpenStreetMap via Leaflet
- **Link visualisatie**: Individuele links worden getoond in grijs
- **Linkketen visualisatie**: Linkketens worden getoond in verschillende kleuren
- **Popup informatie**: Klik op een link of linkketen voor gedetailleerde informatie
- **Automatische zoom**: De kaart past zich automatisch aan om alle data te tonen

## Data structuur

De applicatie gebruikt de klassen uit `klasseDefinities.py` en functies uit `functies.py` om:
- Links te creëren met start- en eindnodes
- Linkketens te vormen uit verbonden links
- Intensiteitsgegevens te berekenen en weer te geven

## Uitbreidingsmogelijkheden

- Real-world coördinaten gebruiken in plaats van grid-posities
- Data laden uit bestanden (CSV, JSON, database)
- Meer visualisatie opties (heatmaps, verschillende symbolen)
- Export functionaliteit voor kaarten
- Filtering en zoekfuncties