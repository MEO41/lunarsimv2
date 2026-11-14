---
id: lunar-surface-statistics-research
title: Lunar surface statistics research
type: research
status: current
updated: 2026-11-14
summary: Published laws for small craters, crater depth, rocks, slopes and regolith strength, with sources; input for the terrain detail.
---

# Lunar surface statistics research (2026-11-14)

Main source: NASA SLS-SPEC-159 Rev I, "Cross-Program Design Specification for Natural
Environments" (DSNE), 2021, NTRS 20210024522. Other keys in [[sources-bibliography]].

## Small craters (cumulative number per area)

| Law | Form | Range | Source |
| --- | --- | --- | --- |
| Design equilibrium (upper limit) | N(≥D) = 0.079433·D⁻² per km² with D in km, equal to 0.0794·D⁻² per m² with D in m | D ≤ 200 m; may be extrapolated below 10 m | DSNE §3.4.1.1 |
| Measured equilibrium, Apollo 15 mare | n(>r) = 0.0084·r⁻² per m² → N(>D) = 0.0336·D⁻² | 10 m < r < 50 m | Minton et al. 2019, Icarus 326, Eq. 2 |
| Alternative fit | n(>r) = 0.0051·r⁻¹·⁸ | | Minton et al. 2019 |
| Production | n(>r) = 4.3·r⁻³·² per m² | r < 500 m, 3.54 Gy | Minton et al. 2019 §1.2 |

The DSNE law is about 2.4 times the measured fit: a conservative envelope. No Taurus-Littrow value found.

## Crater shape

- Fresh depth/diameter (DSNE Table 3.4.1.2-1, after Stopar et al. 2017): 40–100 m: 0.12–0.14;
  100–200 m: 0.14–0.16; 200–400 m: 0.16–0.17.
- Degradation classes for D < 1 km (DSNE Table 3.4.1.2-2, after Basilevsky 1976):

| Share | d/D | Max wall slope |
| --- | --- | --- |
| 0.5 % | 0.12–0.2+ | 35°+ |
| 2.5 % | 0.12–0.2 | 25–35° |
| 17 % | 0.10–0.15 | 15–25° |
| 30 % | 0.07–0.10 | 10–15° |
| 50 % | < 0.07 | < 10°, no rim |

- No sourced rim-height ratio or analytic bowl-and-rim profile was found: it is a model choice.

## Rocks

- DSNE §3.4.1.4: N(>D) = C·D^γ per 100 m², D in m, valid below 2 m (Shoemaker and Morris 1970).
  Surveyor 1: C = 0.234, γ = −2.11 ("91st percentile environment", usable for design), i.e.
  0.00234·D⁻²·¹¹ per m². Others: Surveyor III 0.069/−2.56, V 0.014/−2.65, VI 0.058/−2.51,
  VII (Tycho rim) 2.739/−1.82.
- Rock height to diameter 0.60, σ 0.28 (Demidov and Basilevsky 2014, in DSNE).
- Diviner rock abundance: typically < 1 % of area, mean 0.3 % (DSNE); global 0.004 (Bandfield et al. 2011).

## Slopes and LM design surface

- Maria median slope 2.0° (quartiles 1.0–4.1°) at about 17 m baseline; highlands 7.5°
  (Rosenburg et al. 2011, in DSNE Table 3.4.1.3-1).
- LM design surface (TN D-6850): mean slope ≤ 6°, effective slope ≤ 12°, relief in the gear
  footprint ≤ 24 in (0.61 m).

## Regolith

- Cohesion 0.1–1.0 kPa, friction angle 30–50° (Apollo 17 Preliminary Science Report §8; DSNE
  Table 3.4.2.3-1). Shear τ = c + σ·tan φ.
- Footpad sliding friction 0.4 to 1.0 specified; 0.4 judged realistic after flight (TN D-6850).
- Apollo 11 footpad penetration 1–2 in; 18–22 in slide in the post-flight simulation.

- feeds: [[terrain-integration-plan]]
