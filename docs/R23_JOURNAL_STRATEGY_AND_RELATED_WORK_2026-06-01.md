# R23 journal strategy and related-work positioning

Date checked: 2026-06-01
Project: P4_SeismicFoundationModel

## 1. Target-journal decision

| Route | Recommendation | Rationale | Main risk |
|---|---:|---|---|
| Nature Machine Intelligence | Stretch only | The journal scope emphasizes high-quality research in machine learning, robotics and AI with cross-disciplinary impact. Current manuscript is engineering reliability-centered; the AI-method novelty is not yet broad enough. | Desk reject as an application case without a general AI advance. |
| Structural Safety | Primary | Official scope centers integrated risk assessment for constructed facilities, including response/performance prediction and techniques of decision analysis and risk management. This maps directly to event-level reliability, residual uncertainty, and false-safe decision risk. | Needs stronger structural reliability framing and careful PBEE boundary. |
| Engineering Structures | Safe-primary / engineering route | Official scope includes earthquake engineering, structural dynamics, structural reliability, performance-based design, and data-driven analysis methods. | Must present a clear high-level scientific/technical innovation rather than a case study. |
| EESD / Earthquake Engineering & Structural Dynamics | Safe / specialty route | Likely relevant to seismic performance and earthquake engineering, but the current round did not complete a live official policy extraction. | Need a separate official author-instructions pass before submission. |

## 2. Official-source checks used in this round

- Nature Machine Intelligence aims and scope: publishes original research and reviews across machine learning, robotics and AI, and discusses impacts on scientific disciplines, society and industry.
- Nature Portfolio AI policy: LLMs do not satisfy authorship criteria; substantive LLM use should be documented; Springer Nature states that generative AI images are generally not permitted for publication except limited labelled cases.
- Structural Safety journal page: identifies the journal as an international journal on integrated risk assessment for constructed facilities; listed topics include prediction of response/performance and techniques of decision analysis/risk management.
- Engineering Structures guide for authors: scope includes earthquake engineering, structural dynamics, structural reliability/stability, performance-based design, digital/data-driven analysis methods; it also states that case studies generally need clear high-level scientific or technical innovation.
- Elsevier generative AI policy in the Engineering Structures guide: authors must disclose generative AI use in manuscript preparation and remain responsible for reviewing/validating output; AI tools must not be listed as authors.

## 3. Related-work positioning axes

### Seismic surrogate modeling

Position this paper against work that builds surrogate or metamodel predictors for structural seismic response. The novelty should not be phrased as `we train a better surrogate`; it should be phrased as `we expose how surrogate reliability changes when finite labels, event-level separation, residual geometry and engineering decision loss are evaluated together`.

### Conformal prediction under shift

Use conformal-prediction literature as methodological context, not as a claim that this paper develops a new conformal algorithm. The paper's contribution is domain-specific evidence on how exchangeability stress and residual-scale mismatch appear in seismic surrogate calibration.

### PBEE and FEMA P-58

Use PBEE/FEMA P-58 as a boundary-setting framework. The current analysis is at the engineering demand parameter screening stage. It does not estimate damage measures, decision variables, consequences, repair cost, downtime, or casualties.

### Structural reliability and decision analysis

This is the strongest conceptual home. The false-safe cost ratio should be framed as a normalized decision-risk sensitivity device, not as a calibrated economic loss model.

## 4. Innovation hierarchy after R23

| Innovation type | Strength after R23 | Defensible wording |
|---|---:|---|
| Theory innovation | Moderate | Residual-scale and tail geometry provide a mechanism-aware explanation for metric-dependent reliability under event-level shift. |
| Mechanism innovation | Stronger than R22 | R23 shows mean error, coverage, q95 tail residual, and residual-scale mismatch occupy different residual-distribution regimes. |
| Method innovation | Moderate | The contribution is an integrated evaluation workflow: event-disjoint conformal stress, full residual ranking, decision-risk sensitivity and residual-mechanism diagnostics. |
| Data/evidence innovation | Moderate-strong | Finite-label event-level seismic surrogate evidence is directly tied to model selection and decision risk. |
| Journal-level broadness | Field-top strong, Nature-family weak-to-moderate | Best framed for structural reliability/risk rather than general machine intelligence. |

## 5. Claims allowed after R23

- Allowed: Metric-dependent model reliability is observed under the evaluated finite-label event-level protocols.
- Allowed: Residual-scale mismatch and tail amplification help explain why accuracy, coverage and decision-risk metrics can select different models.
- Allowed: False-safe cost sensitivity can change operational model preference.
- Not allowed: This is a universal seismic foundation model.
- Not allowed: This is a new conformal-prediction theorem.
- Not allowed: This performs a complete FEMA P-58 performance assessment.
- Not allowed: The manuscript has a guaranteed high probability of Nature-level acceptance.
