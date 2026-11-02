# Bladeren in Spanderswoud

De stelling: *de bladeren kwamen dit jaar later aan de bomen, en daardoor hangen er nu nog relatief veel aan.*
Die stelling is getoetst voor het loofbos in Spanderswoud (Hilversum) met Sentinel-2-opnamen van 2017–2026.
De resultaten staan op de pagina in [`docs/`](docs/index.html).

**Uitkomst (stand 28 oktober 2026):** de bladeren kwamen niet later aan de bomen, eerder vroeg. Nu is het loofbos rond of
iets boven normaal groen. Dat valt vooral op naast 2025, het vroegst kale jaar van de reeks. Een duidelijke
weersverklaring is er niet.

1. **Uitloop niet laat.** Op 21 april was het loofbos al over de helft (normaal ±20 april). De schattingen lopen van
   13 april (per pixel) tot 19 april (fit). Late jaren waren 2021 en 2023.
2. **Nu rond of iets boven normaal.** Op 28 oktober was er 58–65% van het zomergroen over, afhankelijk van het gekozen
   zomerniveau (normaal 59%, 2025: 37%). Het model schat het midden van de verkleuring rond 2 november, +2 dagen ten
   opzichte van normaal en binnen de onzekerheid.
3. **Niet door de uitloop.** Over de jaren gaat een late uitloop vaak samen met een late herfst (ρ 0,67, n=8), maar
   2026 liep niet laat uit.
4. **Geen duidelijke weersverklaring.** De septembertemperatuur hangt niet samen met de verkleuring (ρ 0,05, n=9).
   Er was geen grondvorst, maar dat gold ook voor 2017 en 2023–2025.

Sentinel-2 ziet groen blad, niet bruin blad dat nog aan de boom hangt. Het gaat om één bos, geen landelijk beeld.

## Werkwijze

1. `s2_extract.py`: haalt Sentinel-2 L2A (tegel 31UFT) op uit het openbare Earth Search-archief en schrijft per
   opname NDVI, blauw en SCL voor een venster rond het bos weg. `sentinel-2-c1-l2a` is de hoofdbron. Het gat in 2022
   en de jaren 2017–2018 komen uit `sentinel-2-l2a`. De BOA-offset wordt per opname uit de data bepaald, omdat de
   metadatavlag in de oude collectie niet betrouwbaar is.
2. `s2_phenology.py`: maakt loof- en naaldbosmaskers binnen de OSM-grens, berekent de reeks
   ΔNDVI = NDVI(loof) − NDVI(naald) en fit per jaar een logistische afname in de herfst met vaste steilheid.
   Uitkomsten zijn t50 en t15 (≈ HR-VPP *end of season*). Ruwe NDVI loopt mee als gevoeligheidscheck. Het script
   schrijft ook de kaart van 28 oktober 2026.
3. `s2_spring.py`: dezelfde aanpak voor de bladuitloop (logistische toename, sos50) en het verband met het
   herfstmoment.
4. `brackets.py`: bakent beide momenten af met de waarnemingen zelf: de laatste opname vóór en de eerste opname ná
   het halverwegepunt.
5. `s2_pixel_doy.py`: dag van uitloop en verkleuring per loofbospixel per jaar, volgens de *day-of-year*-aanpak:
   per pixel genormaliseerd, gecorrigeerd met het naaldbos, een overschrijding pas tellen als de volgende opname hem
   bevestigt, en interpoleren tussen de opname ervoor en erna.
6. `knmi_summary.py`: september- en oktobertemperatuur, luchtvorst (TN < 0) en grondvorst (T10N < 0) in
   september–oktober, plus de eerste grondvorst, in De Bilt (KNMI 260).
7. `export_web.py`: schrijft `docs/data.json` voor de Observable Plot-pagina.

```bash
uv run --no-project --with rasterio,pyproj,numpy,requests python s2_extract.py
uv run --no-project --with rasterio,pyproj,numpy,scipy,matplotlib,pandas python s2_phenology.py
uv run --no-project --with numpy,scipy,pandas python s2_spring.py
uv run --no-project --with rasterio,pyproj,numpy,scipy,matplotlib,pandas python s2_pixel_doy.py
uv run --no-project --with pandas python knmi_summary.py   # na download KNMI (TG, TN, T10N)
uv run --no-project --with numpy,pandas,pyproj python export_web.py
```

Bij een complete run haal je de oude collectie met `S2_COLLECTION=sentinel-2-l2a`, `S2_CUBE=...` en `S2_PERIOD=...`
op (zie `CUBES` in `s2_phenology.py`). De KNMI-dagwaarden komen van
`https://www.daggegevens.knmi.nl/klimatologie/daggegevens` (station 260, TG en TN).

## Kanttekeningen

- Sentinel-2 meet **groen** blad. Bruin blad dat nog aan de boom hangt telt als "weg".
- In het voorjaar van 2026 zit er tussen 22 maart en 21 april geen heldere opname. In de herfst is 28 oktober de
  laatste heldere opname en was het halverwegepunt nog niet bereikt.
- Het gaat om één bos met ongeveer 50 ha loofbos. Dit is geen landelijk beeld. Met 9 à 10 jaren is een verband tussen
  uitloop en verkleuring niet hard te maken.

## Bronnen en licenties

Copernicus Sentinel-2-data (ESA/EU) via Earth Search (Element 84, AWS Open Data) · begrenzing © OpenStreetMap-bijdragers
(ODbL) · KNMI daggegevens.
