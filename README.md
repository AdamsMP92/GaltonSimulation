# GaltonSimulation

Eine nachvollziehbare, ereignisgetriebene Referenzimplementierung des in
[`galton_raytracing_spec.md`](galton_raytracing_spec.md) beschriebenen
deterministischen Galtonbretts. Zwischen Ereignissen werden die Flugbahnen
analytisch ausgewertet; es gibt keine Zeitintegration mit festen Schritten.

## Installation und erster Lauf

```shell
pixi install
pixi run simulate
```

Der Lauf speichert Konfiguration, Messgrößen, Rohdaten und eine
zweigeteilte Ergebnisgrafik unter `results/latest/`. Links werden
standardmäßig sämtliche simulierten Trajektorien ohne Subsampling gezeigt,
rechts das Dichtehistogramm aller kontinuierlichen Auftreffpositionen mit
einer deskriptiven KDE-Hülle. Titel und Detektormarkierungen nennen ebenfalls
explizit die vollständig dargestellte Population.

Die Parameter des Beispiels sind vollständig und direkt lesbar in
[`scripts/run_single.py`](scripts/run_single.py) definiert. Das Beispiel nutzt
echte reflektierende Seitenbanden. Der Detektor liegt bei `y=0`, sodass der
freie Abstand zur untersten Stiftreihe unmittelbar ablesbar ist.

Für den üblichen Einstieg benennt `simple_galton_config` alle physikalischen
Längen als explizite Verhältnisse zur Startbreite. Keines dieser Verhältnisse
besitzt einen versteckten Default:

```python
from galton import run_simulation, simple_galton_config

config = simple_galton_config(
    rows=8,
    peg_diameter_ratio=0.1,
    ball_diameter_ratio=0.05,
    peg_spacing_ratio=0.3,
    release_height_ratio=0.2,
    detector_gap_ratio=2.0,
    detector_width_ratio=3.1,
    boundary_behavior="reflecting",
    starts=2000,
    kinetic_energy_retention=0.99,
)
result = run_simulation(config)
```

| Parameter | Bedeutung |
| --- | --- |
| `rows` | Zahl der Stiftreihen |
| `peg_diameter_ratio` | Stiftdurchmesser geteilt durch Startbreite |
| `ball_diameter_ratio` | Kugeldurchmesser geteilt durch Startbreite |
| `peg_spacing_ratio` | Nachbarabstand der Stifte geteilt durch Startbreite |
| `release_height_ratio` | Abwurfhöhe über dem obersten Stift geteilt durch Startbreite |
| `detector_gap_ratio` | Freier Detektorabstand geteilt durch Startbreite |
| `detector_width_ratio` | Für Kugelmittelpunkte verfügbare Breite zwischen den unteren Banden |
| `boundary_behavior` | `reflecting` für echte Banden, `absorbing` für ein Austrittskriterium |
| `starts` | Deterministische Startpunkte innerhalb der Öffnung |
| `kinetic_energy_retention` | Einheitlicher Stoßfaktor `w_e`; `1.0` ist elastisch, `0 < w_e < 1` dissipativ |

Der Wrapper setzt lediglich `start_width=1`, `v0=(0,0)`, `g=1` und
`kinetic_energy_retention=1` als offene Modellstandards. Er leitet den
vertikalen Abstand für ein
gleichseitiges Dreiecksraster, die äußere Referenzgrenze und die absolute
Höhenlage automatisch ab. Der effektive Kollisionsradius folgt transparent
aus `(peg_diameter_ratio + ball_diameter_ratio)/2`. Nur für Sonderfälle ist
die vollständige `SimulationConfig` nötig.

Bei jedem Stift- und Bandenkontakt wird zunächst die Geschwindigkeit
spiegelnd reflektiert und danach vollständig mit `sqrt(w_e)` skaliert. Damit
gilt unmittelbar am Stoß exakt

```text
|v_after|² = w_e |v_before|²
K_after   = w_e K_before
```

Das ist ein bewusst einfaches, phänomenologisches Verlustmodell. Anders als
ein gewöhnlicher normaler Rückprallkoeffizient dämpft es auch die tangentiale
Geschwindigkeitskomponente. Da keine Masse modelliert wird, sind die
gespeicherten Energien spezifische Energien `K/m = |v|²/2`.
In der erweiterten `SimulationConfig` lassen sich zusätzlich getrennte
normale Rückprallkoeffizienten für Stifte und Banden setzen; diese wirken vor
`w_e`. Der einfache Wrapper fixiert beide auf `1`, damit die obige Gleichung
ohne weitere versteckte Dämpfung gilt.

Die Bandengeometrie ist stetig und besteht aus drei Teilen:

1. einem vertikalen Einlass von der Startlinie bis zur obersten Reihe,
2. schrägen Seitenwänden mit derselben Neigung wie das Dreiecksraster,
3. vertikalen Banden von der untersten Reihe bis zum Detektor.

`detector_width_ratio` bezeichnet bewusst die nutzbare Breite für die
Kugelmittelpunkte, passend zu den gespeicherten Koordinaten `x_detector`. Damit
alle Startpunkte und Stifte innerhalb der Banden liegen, prüft der Wrapper die
geometrische Mindestbreite und meldet zu kleine Werte als Fehler.

Ein vollständiger Lauf verwendet dieselbe explizite Konfiguration:

```python
from galton import run_simulation
from scripts.run_single import EXAMPLE_CONFIG

result = run_simulation(EXAMPLE_CONFIG)
print(result.summary.to_dict())
```

Die KDE dient ausschließlich der Darstellung. Sie ist keine Anpassung einer
Gaußverteilung und beeinflusst die Dynamik oder Auswertung nicht.

Parameter werden über eine unveränderliche Konfiguration gesetzt:

```python
from galton import SimulationConfig, run_simulation

config = SimulationConfig(
    n_rows=11,
    n_starts=512,
    boundary_mode="collision_envelope",
)
result = run_simulation(config)
```

Alle Längen verwenden dieselbe frei wählbare Einheit. `start_width` ist die
Längenskala, `sqrt(start_width / gravity)` die natürliche Zeitskala. Die
Standardwerte sind dimensionslos und nur ein reproduzierbares Beispiel, kein
experimentell validierter Benchmark.

## Parameterscan

```python
import numpy as np
from galton import scan_rho_delta

scan = scan_rho_delta(
    rho_values=np.linspace(0.45, 0.75, 7),
    delta_values=np.linspace(2.2, 3.2, 9),
)
```

Standardmäßig bleibt dabei `h/d` konstant. Mit `keep_h_over_d=False` bleibt
stattdessen `h/W` konstant; diese beiden Schnitte beantworten unterschiedliche
physikalische Fragen.

## Deterministischer Startgitter-Scan

Die normalen Ensembleläufe verwenden bereits keine Zufallszahlen, sondern
äquidistante Mittelpunkte innerhalb der Startbreite. Um zu prüfen, ob bei
feiner werdendem Startgitter weiterhin neue Detektorpositionen auftreten,
führt ein eigener Runner mehrere Auflösungen nacheinander aus:

```shell
pixi run scan-start-grid
```

Die expliziten Einstellungen stehen in
`scripts/run_start_grid_scan.py`:

```python
GRID_SIZES = (100, 200, 400, 800, 1600)
ENDPOINT_TOLERANCE_RATIO = 0.01
```

Der Scan speichert unter `results/start_grid_scan/`:

- die direkte, histogrammfreie Abbildung `x0 -> x_detector`,
- die Zahl exakt verschiedener Detektorpositionen,
- die Zahl bei einer vorgegebenen räumlichen Toleranz noch trennbarer
  Endpunktgruppen,
- empirische Verteilungsfunktionen für alle Gitterauflösungen,
- Kolmogorov- und Wasserstein-Abstände zur feinsten Auflösung,
- alle ausgerichteten Startpositionen, Endpositionen und Statuswerte in NPZ.

Die toleranzbasierte Gruppenzahl besitzt im Gegensatz zu einem Histogramm
keinen frei wählbaren Bin-Ursprung. Sie beantwortet daher direkter, ob die
erreichbaren Detektorpositionen bei wachsender Auflösung weiter zunehmen oder
auf wenige Attraktoren sättigen.

## Modellentscheidungen

- spiegelnde Stöße an unbeweglichen Kreisstiften und Banden, optional mit
  demselben kinetischen Energieerhaltungsfaktor `w_e` für jeden Kontakt;
- keine Kugel-Kugel-Kollisionen oder Rotation;
- optional reflektierende oder absorbierende Seitenbanden;
- kontinuierliche Detektorebene;
- `collision_envelope`, überwacht ab `y_top` und geradlinig bis zum Detektor
  extrapoliert;
- nahezu tangentiale Kontakte ohne einlaufende Normalgeschwindigkeit werden
  nicht reflektiert;
- simultane Mehrfachkontakte werden als `numerical_failure` ausgewiesen und
  nicht durch eine willkürliche Stoßreihenfolge aufgelöst;
- Grenz- und Abbruchzustände zählen nie als Überleben.

`run.json` enthält `w_e` und den mittleren kumulierten Energieverlust.
`run.npz` speichert zusätzlich für jeden Stift- und Bandenkontakt die
kinetische Energie davor, danach und die dabei entfernte Energie. Dadurch
lässt sich die Dämpfung ohne erneute Simulation auswerten.
Die reine Simulationszeit ohne Dateiausgabe und Plot wird als
`result.simulation_seconds` zurückgegeben, unter `timing.simulation_seconds`
in `run.json` gespeichert, vom Beispielskript ausgegeben und im Kopf der
Ergebnisgrafik angezeigt.

Unabhängige Starttrajektorien können mit `run_simulation(config, workers=2)`
auf zwei Prozesse verteilt werden. Im Beispielskript steht dafür die gut
sichtbare Konstante `WORKERS`. Zulässig sind bewusst nur `1` und `2`, damit
der Rechner gemäß der lokalen Ressourcenregel reaktionsfähig bleibt.

Die Kollisionssuche verwirft Poller außerhalb der räumlichen Hülle des jeweils
nächsten freien Flugsegments, bevor sie die aufwendigen Quartikgleichungen
löst. Das verändert die Ereignisse nicht, vermeidet bei großen Gittern aber
den Großteil unnötiger Wurzelsuchen. Bei stark dissipativen Läufen kann es
trotzdem zu einem inelastischen Kollaps kommen: Eine Kugel führt dann immer
kleinere Stöße am selben Poller aus. Wenn Stoßort, Poller und die bereits
unaufgelöste winzige Flugzeit über mehrere Kontakte ein numerisches Plateau
bilden, wird die Bahn als `stalled` ausgewiesen. Sie geht damit wie andere
nicht detektierte Zustände weder in Histogramm noch KDE ein. Vorübergehend
kleine Kontakte werden ausdrücklich nicht gefiltert. Das ist ein Filter für
einen vom Hartstoßmodell nicht mehr auflösbaren Grenzzyklus, keine
Fortsetzung der Kontaktphysik.

### Konsequenz der vorgeschlagenen Standardhülle

Die Kombination aus einer einzelnen Spitze, vertikalem Start oberhalb des
obersten Stifts und Überwachung der `collision_envelope` bereits ab `y_top`
bildet geometrisch einen geschlossenen Engpass: Um am obersten effektiven
Kollisionskreis vorbeizukommen, muss der Kugelmittelpunkt bei `y_top` bis auf
einen tangentialen Sonderfall `|x| > R_peg + R_ball` erreichen und hat damit
die Hülle bereits verlassen. Der Standardlauf ergibt deshalb korrekt
`survival_fraction = 0`; das ist keine numerische Instabilität.

Für eine nichttriviale Sensitivitätsanalyse ist bereits
`boundary_mode="fixed_walls"` verfügbar. Diese Grenzen sind nur ein
absorbierendes Austrittskriterium und keine reflektierenden physischen Banden.
Eine andere Eintrittshülle oder ein späterer Beginn der Hüllenüberwachung
sollte als eigene, ausdrücklich benannte Modellentscheidung ergänzt werden und
nicht stillschweigend den Standard verändern.

Tests starten mit:

```shell
pixi run test
```
