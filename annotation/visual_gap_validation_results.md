# Visual Gap Human Validation Results

This is an initial construct-validity check for the VLM-derived visual information gap score.

## Sample

- Annotated posters: 50
- Human scores: missing context, visual ambiguity, emotional tension, curiosity, each on a 1-5 scale.
- Note: the annotation page included VLM fields for auditability. The annotator reports scoring from poster judgment rather than using score bins, so this should be described as a single-rater pilot validation rather than a strict blind study.

## Descriptive Statistics

|       |   human_missing_context_1_5 |   human_visual_ambiguity_1_5 |   human_emotional_tension_1_5 |   human_curiosity_1_5 |   human_gap_mean |   human_gap_construct |   vlm_visual_information_gap_score |   vlm_cross_modal_gap_score |
|:------|----------------------------:|-----------------------------:|------------------------------:|----------------------:|-----------------:|----------------------:|-----------------------------------:|----------------------------:|
| count |                      50     |                        50    |                        50     |                50     |           50     |                50     |                             50     |                      50     |
| mean  |                       3.06  |                         2.68 |                         2.2   |                 2.46  |            2.6   |                 2.647 |                              0.81  |                       0.843 |
| std   |                       1.077 |                         1.22 |                         0.948 |                 1.147 |            0.92  |                 0.892 |                              0.146 |                       0.1   |
| min   |                       1     |                         1    |                         1     |                 1     |            1     |                 1     |                              0.6   |                       0.514 |
| 25%   |                       2     |                         2    |                         1.25  |                 2     |            2     |                 2     |                              0.7   |                       0.805 |
| 50%   |                       3     |                         3    |                         2     |                 2     |            2.5   |                 2.333 |                              0.8   |                       0.86  |
| 75%   |                       4     |                         3    |                         3     |                 3     |            3.438 |                 3.333 |                              1     |                       0.889 |
| max   |                       5     |                         5    |                         5     |                 5     |            4.5   |                 4.333 |                              1     |                       0.988 |

## Spearman Correlations

| predictor                        | target              |   spearman_rho |   p_value |
|:---------------------------------|:--------------------|---------------:|----------:|
| vlm_visual_information_gap_score | human_curiosity_1_5 |          0.85  |    0      |
| vlm_visual_information_gap_score | human_gap_mean      |          0.812 |    0      |
| vlm_visual_information_gap_score | human_gap_construct |          0.767 |    0      |
| vlm_cross_modal_gap_score        | human_curiosity_1_5 |          0.305 |    0.0314 |
| vlm_cross_modal_gap_score        | human_gap_mean      |          0.317 |    0.0251 |
| vlm_cross_modal_gap_score        | human_gap_construct |          0.299 |    0.0348 |

## Human-Item Reliability

- Cronbach alpha across all four human items: 0.855
- Cronbach alpha for the three construct items, excluding curiosity: 0.757

## Mean Scores by VLM Bin

| score_bin   |   n |   human_curiosity_mean |   human_curiosity_sd |   human_gap_mean |   human_gap_sd |   vlm_visual_gap_mean |
|:------------|----:|-----------------------:|---------------------:|-----------------:|---------------:|----------------------:|
| high        |  16 |                  3.812 |                0.655 |            3.656 |          0.523 |                 0.994 |
| low         |  17 |                  1.471 |                0.514 |            1.838 |          0.5   |                 0.647 |
| middle      |  17 |                  2.176 |                0.636 |            2.368 |          0.546 |                 0.8   |

## Likely VLM Overestimates

| annotation_id   | title                               | score_bin   |   vlm_visual_information_gap_score |   human_curiosity_1_5 |   human_gap_mean |   vlm_minus_human_rank |
|:----------------|:------------------------------------|:------------|-----------------------------------:|----------------------:|-----------------:|-----------------------:|
| A024            | Cold Comfort Farm (1995)            | middle      |                                0.8 |                     1 |             1.5  |                   0.4  |
| A027            | Beautiful Girls (1996)              | middle      |                                0.8 |                     1 |             2    |                   0.4  |
| A001            | Escape from L.A. (1996)             | high        |                                1   |                     3 |             3.5  |                   0.17 |
| A002            | Hot Shots! Part Deux (1993)         | high        |                                1   |                     3 |             2.75 |                   0.17 |
| A039            | Man Who Knew Too Little, The (1997) | high        |                                1   |                     3 |             3.5  |                   0.17 |
| A019            | Gods Must Be Crazy, The (1980)      | high        |                                1   |                     3 |             2.75 |                   0.17 |

## Likely VLM Underestimates

| annotation_id   | title                         | score_bin   |   vlm_visual_information_gap_score |   human_curiosity_1_5 |   human_gap_mean |   human_minus_vlm_rank |
|:----------------|:------------------------------|:------------|-----------------------------------:|----------------------:|-----------------:|-----------------------:|
| A042            | Bloodsport (1988)             | low         |                                0.6 |                     2 |             2.25 |                   0.31 |
| A011            | Lion King, The (1994)         | low         |                                0.6 |                     2 |             1.75 |                   0.31 |
| A014            | Apt Pupil (1998)              | high        |                                0.9 |                     4 |             3.75 |                   0.18 |
| A047            | Road to El Dorado, The (2000) | middle      |                                0.8 |                     3 |             2.5  |                   0.17 |
| A046            | Soul Food (1997)              | middle      |                                0.8 |                     3 |             3.25 |                   0.17 |
| A033            | Broken Arrow (1996)           | middle      |                                0.8 |                     3 |             2.5  |                   0.17 |

## Interpretation

- The VLM score is directionally aligned with human curiosity in this pilot sample.
- The human items are internally consistent enough to support a compact validation claim.
- The results are appropriate for an RA application as evidence of construct validation work, but should not be overstated as a multi-rater or fully blind validation.
- The next technical step is to recalibrate the VLM scoring function and analyze the overestimate/underestimate cases.
