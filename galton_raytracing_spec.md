# Deterministisches Galtonbrett: ballistisches Raytracing und geometrische Überlebensanalyse

## Zweck dieses Dokuments

Dieses Dokument ist eine **Implementierungs- und Forschungs-Spezifikation** für die lokale Weiterentwicklung mit Codex. Es beschreibt ein zweidimensionales Galtonbrett, bei dem Kugeln unter konstanter Gravitation zwischen Hinderniskontakten exakt ballistisch fliegen. Kollisionen mit unbeweglichen kreisförmigen Stiften werden als instantane, reibungsfreie und zunächst vollständig elastische Stöße behandelt. Eine zeitdiskrete ODE-Integration ist **nicht** erforderlich.

**Primäre Forschungsfrage:** Für welche (dimensionslosen) Stiftradien, Kugelradien und Rasterabstände erreichen die meisten gestarteten Kugeln die untere Detektorebene, **ohne vorher das seitlich offene Stiftraster zu verlassen**?

**Sekundäre Forschungsfrage (noch nicht optimieren):** Ist eine seitliche Bande notwendig, um eine hohe Überlebensquote bei gleichzeitig vielen Kontakten zu erhalten?

Zunächst **keine Bande** verwenden. Das Auswertungsziel besteht ausdrücklich nicht nur darin, einen hohen Überlebensanteil zu erreichen, sondern auch triviale Fälle ohne Kollisionen zu erkennen.

---

## 1. Koordinaten, Geometrie und Anfangsbedingungen

- Koordinaten: horizontal `x`, vertikal `y` nach oben; Gravitation `g_vec = (0, -g)` mit `g > 0`.
- Kugelmittelpunkt: `r = (x, y)`; Kugelradius `R_ball`.
- Fester Stift `i`: Mittelpunkt `c_i`, Radius `R_peg` (zunächst für alle Stifte identisch).
- Kontakt des Kugelmittelpunkts mit einem Stift bei `||r - c_i|| = a`, wobei `a = R_peg + R_ball` der **effektive Kollisionsradius** ist.
- `N` Reihen, horizontaler Nachbarabstand `d`, vertikaler Reihenabstand `h`.
- Dreiecksanordnung mit Reihe `i = 0,...,N-1` und `i+1` Stiften:

  ```text
  x_(i,j) = (j - i/2) * d,  j = 0,...,i
  y_i     = y_top - i*h
  ```

  Diese Anordnung ist mittig um `x=0` zentriert; bei der obersten Reihe befindet sich ein Stift bei `x=0`. Diese Konvention muss in Grafiken und Tests einheitlich bleiben. Als spätere Variante kann ein Raster mit konstanter Anzahl von Stiften pro Reihe ergänzt werden, aber **nicht mit dieser Geometrie vermischen**.

- Startposition `r0 = (x0, y_release)`, zunächst `v0 = (0,0)`.
- Startinterval `x0 ∈ [-W/2, W/2]` mit Startöffnungsbreite `W`.
- Untere Detektorebene `y = y_det < min(y_i) - a`.
- **Kontinuierlicher Detektor:** jede Kugel liefert ihre reelle Auftreffposition `x_det`. Histogramme entstehen nur nachträglich, ohne Detektor-Bins in der Dynamik.
- Prüfen, dass die Startposition außerhalb aller Kollisionskreise liegt; ungültige Anfangszustände melden, nicht stillschweigend verschieben.

**Wichtig:** Die bisher diskutierten Beispielwerte (`W=0.19 m`, `d=0.54 m`, `h=0.49 m`, `R_peg=0.075 m`, `R_ball=0.05 m`, `N=11`) dienten nur der Illustration. Ihre Werte und die früher erwähnten numerischen Resultate sind **nicht als validierte Benchmarks** anzusehen. Neu implementierte Ergebnisse müssen reproduzierbar sein und durch Tests abgesichert werden.

---

## 2. Analytische Bewegung zwischen Stößen

Für einen Flugabschnitt ab Zustand `(r_k,v_k)`:

```text
r(t) = r_k + v_k*t + 0.5*g_vec*t^2
v(t) = v_k + g_vec*t
```

mit lokaler Zeit `t >= 0`.

### Nächste Hinderniskollision

Für jeden Kandidatenstift `c_i` löse

```text
F_i(t) = ||r(t) - c_i||^2 - a^2 = 0.
```

Setze `q = r_k - c_i`, `b = v_k`, `c = 0.5*g_vec`. Dann hat das Quartikpolynom die Koeffizienten

```text
(c·c)*t^4 + (2*b·c)*t^3 + (b·b + 2*q·c)*t^2
    + (2*q·b)*t + (q·q - a^2) = 0.
```

Mit `numpy.roots` oder einer numerisch robusteren Alternative die **positive, nahezu reelle** Nullstelle mit kleinstem `t > t_min` bestimmen, die einem **eintretenden Kontakt** entspricht:

```text
n = (r(t) - c_i)/a
n · v(t) < -contact_velocity_tol
```

Tangentiale Berührung mit `n·v ≈ 0` separat behandeln. Start direkt auf der Oberfläche sowie Rundungsfehler erfordern eine konsistente Toleranzstrategie; **nicht** willkürlich lange Zeitsprünge einführen. Beachte, dass große Real-/Imaginär-Toleranzen falsche Kollisionen erzeugen können.

### Elastische Kollision

Für den einlaufenden Zustand unmittelbar vor Kontakt:

```text
v_minus = v_k + g_vec * t_hit
n       = (r_hit - c_i) / a
v_plus  = v_minus - (1 + e)*(v_minus·n)*n
```

Zunächst `e = 1` (vollständig elastisch), keine Rotation, keine Reibung, unbewegliche Stifte, keine Kugel-Kugel-Kollisionen. Allgemeiner kann `0 <= e <= 1` als späterer Modellparameter vorgesehen werden.

Bei `e=1` muss die Geschwindigkeit unmittelbar am Stoß betragsgleich bleiben. Energieerhaltung gilt über den gesamten Prozess einschließlich der gravitativen potenziellen Energie (bis auf numerische Fehler):

```text
E = 0.5*||v||^2 + g*y
```

pro Einheitsmasse.

### Ereignissteuerung

Der nächste Event ist das Minimum von:

1. nächster *zulässiger* Stiftkollision,
2. erstmaliger abwärts gerichteter Detektorkreuzung,
3. erstmaligem **seitlichen Austritt**, falls dessen Grenzkurven dynamisch geprüft werden,
4. gegebenenfalls einer wohldefinierten Abbruchbedingung (z. B. maximale Eventzahl, maximale Gesamtflugzeit).

Bei `y(t) = y_det` analytisch die Zeitpunkte berechnen; nur positive Zeitpunkte mit `v_y(t) < 0` gelten als abwärts gerichtete Detektorkreuzungen. Abbruchzustände müssen als eigene Kategorie ausgegeben werden und dürfen nicht als „überlebt“ gelten.

**Keine Zeitintegration mit festen Zeitschritten.** Zwischen Events können Punkte für Grafiken gesampelt werden; das betrifft nur die Darstellung.

---

## 3. Präzise Definition von „im Spiel bleiben“

Die Formulierung „nicht seitlich aus dem Raster reflektiert“ braucht eine mathematisch prüfbare Definition. Die untere Detektorebene allein ist **nicht** ausreichend: Eine Kugel kann das Raster seitlich verlassen und trotzdem später die untere Ebene erreichen.

### Vorgeschlagene Standarddefinition: erstes Überschreiten einer Rasterhülle

Definiere links/rechts eine polygonale Hülle `x_left(y), x_right(y)`, die durch die äußeren Stiftmittelpunkte (oder optional deren effektive Radien) aufgebaut wird. Die Art der Hülle muss als `boundary_mode` explizit festgelegt sein:

- `center_envelope`: Hülle entlang der äußersten **Stiftmittelpunkte**; geometrisch streng und frühzeitiger Austritt.
- `collision_envelope`: Hülle entlang der äußersten **effektiven Kollisionskreise** (`±a` horizontal) an den Reihen; näher am tatsächlich zugänglichen Kontaktgebiet.
- `fixed_walls`: vertikale Referenzgrenzen, etwa `x=±L/2`; einfacher kontrollierter Vergleich, aber kein identischer Begriff wie Dreieckshülle.

Für die Dreieckshülle unterhalb der obersten Reihe die Eckpunkte/lineare Interpolation verwenden; für den Bereich **oberhalb der ersten Reihe** muss eine separate Eintrittsregion definiert werden, damit ein Start mit `W>0` nicht künstlich als sofort ausgetreten zählt. Ein möglicher Standard ist, die Hülle erst **ab dem ersten Erreichen der obersten Reihenhöhe** zu überwachen. Die Hülle muss bis zur Detektorebene wohldefiniert fortgesetzt werden.

**Erste Implementierungsentscheidung:** `boundary_mode="collision_envelope"`, Monitoring ab `y=y_top`, mit dokumentierter Extrapolation bis `y_det`. Dazu eine Sensitivitätsanalyse mit `fixed_walls` durchführen, um die Abhängigkeit vom Austrittskriterium zu erkennen.

Wichtig: Eine seitliche *Flucht* ist ein **erstmaliges Überschreiten** der Hülle. Eine spätere Rückkehr darf nicht wieder als Überleben zählen.

Eventdetektion der Hülle soll ebenso analytisch (Schnitt der Parabel mit stückweise linearen Grenzen) oder durch zuverlässige ereignisbasierte Intervallprüfung erfolgen; **nicht nur nach Stoßereignissen** prüfen, da eine Flugparabel vorher austreten und wieder eintreten könnte.

### Ergebnis je Trajektorie

```text
status ∈ {"detected_inside", "escaped_left", "escaped_right",
          "invalid_initial_state", "max_events", "max_time", "numerical_failure"}
x_detector: float | null
n_collisions: int
collision_history: optional list
trajectory_segments: optional list
```

Auch für seitlich entwichene Kugeln kann optional deren hypothetischer `x_detector` ohne weitere Hüllenbeschränkung separat erfasst werden; die Definition der Überlebensquote darf dadurch nicht verändert werden.

---

## 4. Dimensionslose Parameter und tatsächliche Freiheitsgrade

Skaliere alle Längen mit `W`:

```text
x_hat = x/W,    y_hat = y/W,
alpha = R_peg/W,
beta  = R_ball/W,
rho   = alpha + beta = (R_peg + R_ball)/W,
delta = d/W,
eta   = h/W.
```

Für ideale harte Kreise und reibungsfreie Stöße hängt der **Kontaktverlauf** ausschließlich von `rho`, nicht getrennt von `alpha` und `beta` ab, **wenn** die Geometrie der Grenzen und alle Anfangs-/Auswertungsbedingungen ebenfalls nur vom effektiven Radius abhängen. Ausnahmen sind reale Banden, Kugelmittelpunkt-Zugänglichkeit, Kugel-Kugel-Stöße, Rotation oder zusätzliche physikalische Effekte.

Wähle als natürliche Zeitskala `t0 = sqrt(W/g)` und als Geschwindigkeitsskala `sqrt(g W)`. Dann lautet die ballistische Gleichung in dimensionslosen Einheiten:

```text
r_hat(tau) = r_hat0 + u_hat0*tau + (0,-1)*tau^2/2.
```

Zusätzliche Steuerparameter sind unter anderem:

- `N` (Reihenzahl),
- `eta=h/W` oder bei fester Form das Verhältnis `h/d`,
- `(y_release-y_top)/W` und `(y_bottom-y_det)/W`,
- Anfangsgeschwindigkeit `v0/sqrt(gW)` sowie deren Verteilung,
- Hüllen-/Austrittsdefinition.

**Wichtige Unterscheidung beim Scan:** Wenn `h/d` festgehalten wird, verändert eine Variation von `delta=d/W` zugleich `eta=h/W` und damit die ganze Rastergeometrie relativ zur Startbreite. Das ist zulässig, muss aber explizit ausgewiesen werden. Alternativ kann `eta` konstant gehalten werden. Beide Schnitte durch den Parameterraum beantworten unterschiedliche Fragen.

Zur Orientierung: Innerhalb derselben Reihe überlappen die **effektiven** Kollisionskreise bei `2*rho >= delta`; das ist eine geometrische Schwelle, **kein** Beweis für Verbleib, Durchgang oder Blockade im gesamten Raster.

---

## 5. Zielgrößen und Forschungsdesign

Für `x0` gleichverteilt auf `[-W/2,W/2]` und festgehaltene übrige Anfangsbedingungen definiere

```text
chi(x0; parameters) = 1, falls detected_inside, sonst 0.
S(parameters) = (1/W) * integral_{-W/2}^{W/2} chi(x0;parameters) dx0.
```

Numerische Näherung mittels **deterministischer** Mittelpunkt-Abtastung:

```text
x0[k] = -W/2 + (k+0.5)*W/M,  k=0,...,M-1
S_M = count(detected_inside)/M
```

Auch alternative Quasi-Monte-Carlo-Abtastung ist möglich. Wegen empfindlicher Abhängigkeit der Kollisionsfolge vom Startwert sind **Gitter-Konvergenz und Verfeinerung** entscheidend; der Fehler muss nicht glatt mit `M` abfallen. Reproduzierbare Startwerte sind wichtiger als nominell viele Zufallsziehungen.

**Zielgrößen getrennt speichern:**

1. `survival_fraction = S_M` – Anteil ohne seitlichen Austritt.
2. `collision_count_mean_all` – mittlere Stoßzahl aller gültigen Startbedingungen.
3. `collision_count_mean_survivors` – mittlere Stoßzahl der überlebenden Kugeln (NaN bei keinen Überlebenden).
4. `fraction_with_ge_k_collisions` – Anteil mit wenigstens `k` Stößen (z. B. `k=3` oder `k=N/2` als frei wählbare Schwelle).
5. `detector_x` – kontinuierliche Auftreffkoordinaten der Überlebenden.
6. `escape_left/right` – richtungsaufgelöste Austrittsquoten.
7. `failure_fraction` – Max-Events, Max-Time, numerische Fehler etc., **separat** von physikalischen Austritten.

Ziel ist die Identifikation eines Parameterbereichs mit hohem `S_M` **und** hoher Stoßaktivität. Ein trivialer freier Fall mit `S_M≈1` und praktisch null Kollisionen ist **kein** physikalisch interessantes Optimum.

### Zunächst ein zweidimensionaler Scan

- `rho = (R_peg+R_ball)/W` und `delta = d/W` variieren.
- `N`, Start- und Detektorhöhen sowie `h/d` (oder alternativ `eta`) vorab festhalten.
- Ausschluss geometrisch ungültiger Fälle (z. B. initiale Überlappung; Hindernisüberschneidungen nur dann, wenn im Modell verboten).
- Grobes Gitter, anschließend gezielte Verfeinerung von Übergangsregionen und Bereichen mit guten Zielgrößen.
- Pro Gitterpunkt gleiches deterministisches Startgitter; zunächst z. B. `M=256`, danach Konvergenztests `M=512, 1024, 2048` für ausgewählte Punkte.
- Ein **zweites** Raster mit unabhängigen Parametervariationen (`eta` statt konstantem `h/d`) erst nach Etablierung des ersten Referenzscans.

Outputs: Heatmaps für `S_M`, `collision_count_mean_all`, `fraction_with_ge_k_collisions`, `failure_fraction`; aussagekräftige Beispieltrajektorien, und eine Konturdarstellung für geometrische Schwellen. Alle Achsen **dimensionslos und eindeutig beschriftet**.

---

## 6. Numerik und Robustheit

- Event-getriebene Schleife statt `solve_ivp`/Zeitschritt-Simulation.
- `numpy.roots` einer Quartik ist als **erste, überprüfbare Referenzimplementierung** akzeptabel, aber bei nahezu tangentialen Kontakten numerisch empfindlich. Später optimieren (Root-Bracketing, Vorfilterung von Stiften, robuste skalierte Polynomkoeffizienten).
- Bei jeder Kollision: einlaufende Normalgeschwindigkeit, Kontaktabstand und Energiebilanz validieren.
- Doppelkollisionen und praktisch zeitgleiche Kontakte explizit erkennen. Eine willkürliche Reihenfolge ist bei ideal starren Kugeln unter Umständen physikalisch nicht eindeutig; diese Situationen zunächst als Spezialfall zählen/kennzeichnen.
- Unmittelbare Wiederkollision mit demselben Stift bei `t≈0` unterdrücken, ohne gültige spätere Wiederkontakte zu verlieren.
- Gesonderte Statusmeldungen für event limits oder numerische Probleme statt stiller Zählung als escaped.
- Caching/Pruning von Kandidatenstiften nur nach korrekter Referenzlösung hinzufügen.
- Die Simulation soll alle wichtigen Parameter mit Einheiten/Skalen und einen reproduzierbaren Konfigurations-Dump ausgeben.

### Minimaltests

1. **Freier Fall:** Keine Stifte; analytische Detektorkreuzung stimmt mit dem Solver überein.
2. **Zentraler senkrechter Stoß:** `v_y` wird bei `e=1` umgekehrt; Geschwindigkeit/kinetische Energie bleibt am Stoß gleich.
3. **Schräger Stoß:** Normalanteil wird reflektiert, Tangentialanteil bleibt erhalten.
4. **Vorbeiflug:** Keine künstliche Kollision bei `min_t ||r(t)-c|| > a`.
5. **Kollision vs. Detektor:** Das zeitlich frühere Ereignis gewinnt.
6. **Kontinuierlicher Detektor:** `x_det` ist eine reelle Zahl, ohne dynamische Bin-Zuordnung.
7. **Seitlicher Austritt:** Hüllengrenze wird während des Flugs überschritten, auch wenn keine weitere Kollision eintritt.
8. **Spiegelsymmetrie:** Bei symmetrischer Geometrie und `v_x=0` ergeben sich für `x0` und `-x0` gespiegelte Ergebnisse, abgesehen von degenerierten Spezialfällen.
9. **Skaleninvarianz:** Vollständige Längenskalierung bei entsprechend dimensionslos gleichen Zuständen ändert die dimensionslosen Trajektorien nicht.
10. **Redundanz der Radien:** Paare `(R_peg,R_ball)` mit identischem `R_peg+R_ball` liefern bei unveränderten übrigen Annahmen identische Trajektorien.
11. **Auflösungsvergleich:** `S_M` für verfeinertes Anfangswertgitter vergleichen und Unsicherheiten aus der Diskretisierung transparent machen.
12. **Kein falscher Erfolg:** `max_events`, `max_time`, `numerical_failure` werden nicht als Überlebende gezählt.

---

## 7. Gewünschte Projektstruktur (Vorschlag)

```text
galton-raytracing/
├── README.md
├── pyproject.toml                 # oder pixi.toml; pixi-kompatibel bevorzugt
├── src/galton/
│   ├── __init__.py
│   ├── geometry.py                 # Stifte, Hülle, dimensionslose Konfiguration
│   ├── ballistics.py               # analytische Flugbahn und Schnittzeiten
│   ├── collisions.py               # Kontaktsuche und elastische Stoßabbildung
│   ├── tracer.py                   # Eventloop, Status, Kollisionshistorie
│   ├── observables.py              # S, Stoßzahlen, kontinuierliche Detektorwerte
│   ├── scans.py                    # deterministischer Parameterscan
│   └── plotting.py                 # Trajektorien und Heatmaps
├── tests/
│   ├── test_ballistics.py
│   ├── test_collisions.py
│   ├── test_boundaries.py
│   └── test_symmetry_scaling.py
└── scripts/
    ├── run_single.py
    └── scan_survival.py
```

Bevorzugt `numpy`, `scipy` nur bei tatsächlichem Bedarf, `matplotlib`, `pytest`. Keine GUI-Pflicht, keine Notebook-Pflicht. Speichere numerische Ergebnisse als `.npz` oder `.csv` plus Konfigurationsmetadaten. Erst Funktionalität und Tests, dann Performance-Optimierung.

---

## 8. Konkreter Codex-Arbeitsauftrag

> Implementiere auf Basis dieser Spezifikation eine minimale, gut testbare, ereignisgetriebene 2D-Galtonbrett-Simulation. Zwischen Kontakten ist die Bahn analytisch ballistisch; Stiftkontakte sind vorerst vollständig elastisch, reibungsfrei und ohne Rotation. Verwende zunächst eine kontinuierliche Detektorebene und keine seitlichen Banden. Priorisiere die korrekte Auswahl des zeitlich nächsten Ereignisses einschließlich der seitlichen Austrittsgrenze. Baue einen reproduzierbaren Scan in `(rho,delta)` bei festem `N` und festem `h/d` und gib Heatmaps der Überlebensquote und Kollisionsaktivität aus. Implementiere zuerst die oben aufgeführten Minimaltests; dokumentiere jede verbleibende offene Modellannahme. Ergebnisse bitte nicht als Gaußverteilung voraussetzen oder auf eine solche hin optimieren.

### Empfohlene Umsetzung in Etappen

- **Etappe 1:** Flugparabel, Quartik-Schnitt, Einzelstoß, Detektor, Unit-Tests.
- **Etappe 2:** Dreiecksraster, Eventloop, Trajektorienplots, physikalische Invarianten.
- **Etappe 3:** explizite Hülle und Austritts-Events; Tests für Grenzfälle.
- **Etappe 4:** `x0`-Ensemble und konsistent definierte Observablen.
- **Etappe 5:** 2D-Scan, Auflösungskonvergenz, Heatmaps und exemplarische Kollisionshistorien.
- **Etappe 6 (später):** Bande, Dämpfung (`e<1`), Kugel-Kugel-Kollisionen und gegebenenfalls Rotation/Reibung – jeweils getrennt, um Effekte eindeutig zuordnen zu können.

## 9. Offene Entscheidungen – nicht stillschweigend verstecken

1. **Seitliche Hülle:** Mittelpunkt-, Kollisionskreis- oder feste Referenzgrenze? Alle sind plausible, aber verschiedene Fragestellungen. Für den ersten Run ist `collision_envelope` vorgeschlagen.
2. **Eintrittsbereich:** Wie wird die Hülle oberhalb der ersten Stiftreihe behandelt, wenn `W` größer als die obere Rasterbreite ist?
3. **Variable Rastergröße:** Soll bei Änderung von `delta` `h/d` oder `h/W` konstant bleiben?
4. **Simultankontakte:** Wie sollen idealisierte Mehrfachkontakte behandelt werden?
5. **Physikalisch sinnvoller Mindestkontaktanteil:** Wie wird „nicht bloß durchfallen“ operationalisiert?
6. **Gleichverteilung der Startposition:** Zunächst ja; später andere Startprofile und horizontale Anfangsgeschwindigkeiten prüfen.

**Wichtigstes Prinzip:** Erst eine klar definierte, reproduzierbare Überlebenskarte ohne Banden bestimmen. Erst danach ist die Aussage sinnvoll, in welchem Sinn Banden erforderlich oder hilfreich sind.
