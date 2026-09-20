# Aariz 29-landmark schema

Proposed `aariz29-v1`. Zero-based output indices follow the pinned source configuration, not the paper table. Every annotation must be joined by its stable `landmark_id`. Full machine-readable IDs and aliases are in [landmarks-29.json](landmarks-29.json).

| Index | Model code | Source symbol | Landmark | Definition / protocol | DentalFlow v1 |
|---|---|---|---|---|---|
| 0 | A | A | A-point | Deepest anterior maxillary concavity. | A |
| 1 | ANS | ANS | Anterior Nasal Spine | Anterior tip of the anterior nasal spine. | ANS |
| 2 | B | B | B-point | Deepest anterior mandibular symphysis concavity. | B |
| 3 | Me | Me | Menton | Lowest point of the mandibular symphysis. | Me |
| 4 | N | N | Nasion | Anterior frontonasal suture point. | N |
| 5 | Or | Or | Orbitale | Lowest point of the inferior orbital rim. | Or |
| 6 | Pog | Pog | Pogonion | Most anterior bony chin point. | Pog |
| 7 | PNS | PNS | Posterior Nasal Spine | Tip of the posterior palatal spine. | PNS |
| 8 | Pn | Pn | Pronasale | Tip of the external nose. | — |
| 9 | R | R | Ramus | Most convex point on the external ramus border. | — |
| 10 | S | S | Sella | Center of the sella turcica cavity. | S |
| 11 | Ar | Ar | Articulare | Intersection of zygomatic shadow and posterior ramus border. | — |
| 12 | Co | Co | Condylion | Most posterosuperior mandibular condyle point. | — |
| 13 | Gn | Gn | Gnathion | Anteroinferior symphysis point between Pog and Me; Aariz construction. | Gn |
| 14 | Go | Go | Gonion | Midpoint of the mandibular angle contour; Aariz construction. | Go |
| 15 | Po | Po | Porion | Upper external auditory canal contour midpoint; anatomical porion. | Po |
| 16 | LPM | LPM | Lower 2nd PM Cusp Tip | Buccal cusp tip of lower second premolar. | — |
| 17 | LIT | LIT | Lower Incisor Tip | Incisal edge of the lower central incisor. | L1 |
| 18 | LMT | LMT | Lower Molar Cusp Tip | Mesiobuccal cusp tip of lower first molar. | — |
| 19 | UPM | UPM | Upper 2nd PM Cusp Tip | Buccal cusp tip of upper second premolar. | — |
| 20 | UIA | UIA | Upper Incisor Apex | Root apex of upper central incisor. | — |
| 21 | UIT | UIT | Upper Incisor Tip | Incisal edge of upper central incisor. | U1 |
| 22 | UMT | UMT | Upper Molar Cusp Tip | Mesiobuccal cusp tip of upper first molar. | — |
| 23 | LIA | LIA | Lower Incisor Apex | Root apex of lower central incisor. | — |
| 24 | Li | Li | Labrale inferius | Most prominent lower vermilion border point. | — |
| 25 | Ls | Ls | Labrale superius | Most prominent upper vermilion border point. | — |
| 26 | N_prime | N&#96; | Soft Tissue Nasion | Soft-tissue point overlying nasion. | — |
| 27 | Pog_prime | Pog&#96; | Soft Tissue Pogonion | Soft-tissue point overlying pogonion. | — |
| 28 | Sn | Sn | Subnasale | Columella–upper-lip junction in the midline. | — |

There are 15 skeletal, 8 dental and 6 soft-tissue landmarks. `Pn` is the nose tip, while `PNS` is the posterior nasal spine. `N_prime`/`Pog_prime` are separate soft-tissue points and must never alias hard-tissue `N`/`Pog`. `UIT→U1` and `LIT→L1` refer only to incisal tips; root apexes remain UIA/LIA.

Ground truth is the unrounded arithmetic mean of the two annotator groups after ID alignment. Preserve both groups and their original coordinates. The upstream loader rounds upward with `ceil`; keep this difference explicit in benchmark comparisons.

The 29-channel artifact retains all points. DentalFlow v1 initially consumes its existing 14-point projection and calculates only SNA/SNB/ANB. Future measurements using these 29 points still require reviewed formulas and suitable protocols; points absent from this schema cannot be gained without new labels/model work.

Sources: [pinned author mapping](https://github.com/manwaarkhd/aariz/blob/0634b8b4e6783b13fb6383fac694d61123904d3c/config.py), [dataset definitions](https://pmc.ncbi.nlm.nih.gov/articles/PMC12313948/). Source code license notice is retained in evidence/AARIZ-CODE-LICENSE.txt. Definitions are paraphrased, not a new clinical protocol.
