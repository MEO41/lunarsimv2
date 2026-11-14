---
id: sources-bibliography
title: Sources bibliography
type: research
status: current
updated: 2026-11-14
summary: Every public source used for numbers in the preset and the plans, with its short key and link.
---

# Sources bibliography

Short keys match the lander preset file (`configs/landers/apollo_lunar_module.yaml`) where they exist there.

## Apollo LM

| Key | Source | Link |
| --- | --- | --- |
| A11-MR | Apollo 11 Mission Report, NASA MSC-00171 (1969), mass table "Lunar landing", sec. 9.13.1. The SP-238 edition has the mass table as Table B-I | https://www.nasa.gov/wp-content/uploads/static/apollo50th/pdf/A11_MissionReport.pdf ; https://ntrs.nasa.gov/api/citations/19710015566/downloads/19710015566.pdf |
| D-7143 | Hammock, Currie, Fisher, "Apollo Experience Report - Descent Propulsion System", NASA TN D-7143 (1973) | https://ntrs.nasa.gov/citations/19730011150 |
| D-6850 | Rogers, "Apollo Experience Report - Lunar Module Landing Gear Subsystem", NASA TN D-6850 (1972) | https://ntrs.nasa.gov/citations/19720018253 |
| LM-NEWS | Grumman, "Apollo Lunar Module News Reference" (1969) | ibiblio.org/apollo (nasa.gov links returned 404 on 2026-11-14) |
| COX | Cox, "Design Development of the Apollo Lunar Module Digital Autopilot", NASA (1978) | https://ntrs.nasa.gov/citations/19780015068 |
| KLUMPP | Klumpp, "Apollo Lunar-Descent Guidance", MIT R-695 (1971) | https://www.ibiblio.org/apollo/Documents/Apollo%20Lunar%20Descent%20Guidance.pdf |
| LM-STRUCT | Grumman, "LM Structures Study Guide", course 30915 | https://www.ibiblio.org/apollo/Documents/lm_structures_study_guide.pdf |
| AOH-LM3 | Apollo Operations Handbook LMA790-3-LM 3, Vol I (1968) | https://www.ibiblio.org/apollo/Documents/LM-3_Apollo_Operations_Handbook_Vol_I.pdf |
| AOH-LM10 | Apollo Operations Handbook LM-10 and subsequent, Vol 1 (1970/1971) | https://www.ibiblio.org/apollo/Documents/LMA790-3-LM10-ApolloOperationsHandbookLunarModuleLM10AndSubsequent-Volume1-SubsystemsData-SearchableText.pdf |
| R-567 | MIT R-567 GSOP Luminary, Section 6 "Control Data" (1968), sec. 6.4.1, data for LM-6 | https://www.ibiblio.org/apollo/NARA-SW/R-567-sec6.pdf |
| AAS-26-167 | Ali et al., "Apollo 11 Lunar Module Touchdown Dynamics Reconstruction Verification and Validation for Human Landing Systems", NTRS 20260001760. Secondary for geometry (from a CAD model) | https://ntrs.nasa.gov/citations/20260001760 |

## Lunar surface and terrain

| Key | Source | Link |
| --- | --- | --- |
| DSNE | NASA SLS-SPEC-159 Rev I, Cross-Program Design Specification for Natural Environments (2021) | https://ntrs.nasa.gov/citations/20210024522 |
| MINTON-2019 | Minton et al., Icarus 326, 63–87 (2019), small-crater equilibrium | https://arxiv.org/abs/1902.07746 |
| HIRABAYASHI-2017 | Hirabayashi et al., Icarus 289 (2017) | https://arxiv.org/abs/1701.00471 |
| BANDFIELD-2011 | Bandfield et al., JGR 116, E00H02 (2011), Diviner rock abundance | https://gemelli.spacescience.org/jbandfield/publications/bandfield_diviner_ra.pdf |
| A17-PSR | Mitchell et al., Apollo 17 Preliminary Science Report, NASA SP-330, sec. 8 Soil Mechanics | https://www.nasa.gov/wp-content/uploads/static/history/alsj/WOTM//a17psr_Section8_SoilMechanics.pdf |
| SOURCEBOOK | Carrier et al., Lunar Sourcebook, chapter 9 (1991) | https://www.lpi.usra.edu/publications/books/lunar_sourcebook/pdf/Chapter09.pdf |
| NAC-DTM | LROC NAC DTM products NAC_DTM_APOLLO17_2M and NAC_DTM_APOLLO17, Arizona State University | Download addresses to be confirmed ([[apollo17-height-maps]]) |

## Constants

Lunar GM, radius, surface gravity and g0 are in `src/lunarsim/core/physical_constants.py` with
their sources in comments: JPL DE440 (Park et al. 2021) for GM, NASA NSSDCA Moon Fact Sheet for
the radius, CGPM 1901 for g0.

Glide-slope figure attribution (Blackmore, Açıkmeşe, Scharf, JGCD 2010) was written from
memory and is not checked.

- cited by: [[lander-preset-apollo-lm]]
- cited by: [[apollo-lm-geometry-research]]
- cited by: [[lunar-surface-statistics-research]]
