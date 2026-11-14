---
id: apollo-lm-geometry-research
title: Apollo LM geometry research
type: research
status: current
updated: 2026-11-14
summary: Apollo LM landing-gear, body, nozzle and RCS geometry from primary NASA and Grumman documents, with page references; input for the lander changes.
---

# Apollo LM geometry research (2026-11-14)

Gathered by a research agent from documents it opened; "secondary" marks the 2026 NASA paper
whose geometry comes from a CAD model. Source keys are in [[sources-bibliography]].

## Footpad and struts

| Quantity | Value | SI | Source |
| --- | --- | --- | --- |
| Footpad | dish, 37 in diameter, about 7 in deep | 0.940 m, 0.178 m | TN-D-6850 App. C p. 50 |
| Footpad joint | ball and socket; restraint straps fail at 3340 in-lb | 377 N·m | TN-D-6850 p. 50–51 |
| Footpad design bearing | 1.0 lb/in² | 6.9 kPa | TN-D-6850 p. 50 |
| Footpad mass | 11 kg (model value) | | AAS-26-167 Table 1, secondary |
| Primary strut length | about 10 ft | about 3.05 m | TN-D-6850 p. 46 |
| Primary strut crush | 4500 lb for 0–10 in, then 9500 lb for 10–32 in; energy 254,000 in-lb | 20.02 kN / 0.254 m, 42.26 kN / 0.559 m; 28.7 kJ | TN-D-6850 Fig. 7, p. 46 |
| Strut upper attach (CAD) | radius 3.061 m, 2.874 m above pad base; inclination about 23° (derived) | | AAS-26-167 Table 2, secondary |
| Secondary strut | compression 4500 lb over 12 in; tension 500 lb over 4 in then 5000 lb to 16 in | 20.02 kN / 0.305 m; 2.22 kN / 0.102 m, 22.24 kN / 0.305 m | TN-D-6850 Fig. 9 |
| Surface probes | 5.6 ft below the pads, on −Z, +Y, −Y legs (not the ladder leg) on Apollo 11 | 1.707 m | TN-D-6850 p. 8 |
| Tread radius | 167 in design; pad centres at 167.57 in | 4.256 m | TN-D-6850 |
| Span over opposite pads | 31.0 ft | 9.449 m | TN-D-6850 Fig. 5 |

## Body, nozzle, RCS, mass

| Quantity | Value | SI | Source |
| --- | --- | --- | --- |
| Descent stage | 13 ft 10 in wide, 5 ft 8.86 in high | 4.216 m, 1.749 m | AOH Fig. 1-3 |
| Pad bottom to stage underside | 4 ft 8.295 in (LM-3); 4 ft 10.30 in (later) | 1.430 m; 1.481 m | AOH Fig. 1-3 |
| Decks | lower X 131.14 in, upper X 196.0 in | 3.331 m, 4.978 m | LM Structures Study Guide p. 46 |
| Nozzle exit diameter | 59 in (LM-3 era); 63 in (LM-10 on) | 1.499 m; 1.600 m | AOH Table 2.3-1 |
| Nozzle exit above pads, unstroked | 19 in (Apollo 11 text) | 0.483 m | TN-D-6850 p. 20 |
| Nozzle clearance after Apollo 11 landing | 13.5 in | 0.343 m | TN-D-6850 p. 20 |
| Nozzle skirt | crushable, up to 28 in | 0.711 m | AOH sec. 2.3 |
| Ascent stage envelope | 14 ft 1 in × 13 ft 3 in × 9 ft 3.495 in | 4.293 × 4.039 × 2.832 m | AOH Fig. 1-3 |
| Top above pads | 19 ft 8.65 in (LM-3); 19 ft 10.65 in (later) | 6.011 m; 6.062 m | AOH Fig. 1-3 |
| RCS jet plane | X = 254 in; up jets 258.8 in, down jets 248.7 in; quads at Y = Z = ±66.1 in | 6.452 m; ±1.679 m | MIT R-567 sec. 6.4.1 (data for LM-6) |
| c.g. at lunar landing | X 213.5, Y 0.4, Z 1.6 in | 10.2 mm, 40.6 mm offsets | Mission Report mass table |
| Inertias at lunar landing | Ixx 12,582, Iyy 13,867 (?), Izz 16,204 slug ft²; products 182, 555, 74 | | Mission Report (SP-238 edition Table B-I) |
| Static tip-over angle | about 39.7° (derived: arm 3.010 m, c.g. 3.62 m high) | | derived |

## Caveats

- Footpad plane station for LM-5: 70.84 in comes from the LM-3 handbook; later drawings imply
  about 68.8 in. Not settled.
- Iyy 13,867 against the preset's 13,667: check the page ([[pending-user-decisions]]).
- Not found: footpad swivel limits, strut inclination and outrigger station in a primary source,
  nozzle exit station.
- Downloaded PDFs were in a temporary scratchpad of the 5060 machine and may be gone; the URLs
  are in [[sources-bibliography]].

- feeds: [[terrain-integration-plan]]
- refines: [[lander-preset-apollo-lm]]
