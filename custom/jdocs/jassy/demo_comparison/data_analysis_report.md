# Data Quality Analysis Report: SO-ARM101 Pick-and-Place
**Generated:** 2025-12-21 15:10:01
**User Dataset:** Your Data (50 episodes)
**Baseline Dataset:** Baseline (so100_strawberry_grape) (5 episodes)

---
## Executive Summary
### Issues Identified:
- **Likely**: Moving Too Fast - Your velocity is 2.7x higher than baseline (2.05 vs 0.75 deg/frame)
- **Likely**: Gripper Opens Late - Your approach phase is 2.1s vs baseline 4.3s
- **Likely**: No Pause Before Grasp - Only 19% of your grasps have a pause vs 43% baseline
- **Confirmed**: Directional Bias - 74% of episodes move LEFT initially (vs 20% baseline)

---
## Hypothesis Validation
| Hypothesis | Verdict | Evidence | Recommendation |
|------------|---------|----------|----------------|
| Moving Too Fast | **LIKELY TRUE** | Your velocity is 2.7x higher than baseline (2.05 vs 0.75 deg/frame) | Slow down movements, especially during approach phase |
| Gripper Opens Late | **LIKELY TRUE** | Your approach phase is 2.1s vs baseline 4.3s | Open gripper earlier - at least 1 second before reaching target |
| Directional Bias | **TRUE** | 74% of episodes move LEFT initially (vs 20% baseline) | Augmentation may help but consider collecting more balanced data |
| No Pause Before Grasp | **LIKELY TRUE** | Only 19% of your grasps have a pause vs 43% baseline | Pause for 0.3-0.5 seconds before closing gripper |
| Moving During Grasp | **FALSE** | Pre-grasp velocity 1.12 is similar to baseline 0.76 | Pre-grasp motion is adequate |

---
## Velocity Analysis
| Metric | Your Data | Baseline | Ratio |
|--------|-----------|----------|-------|
| Mean Velocity | 2.050 °/frame | 0.749 °/frame | 2.74x |
| Max Velocity | 7.56 °/frame | 9.67 °/frame | - |
| Mean (deg/sec) | 61.5 °/s | 22.5 °/s | - |

**Reference:** Good velocity is typically 0.75-1.0 °/frame (~22-30 °/sec at 30 FPS)

---
## Gripper Timing Analysis
| Metric | Your Data | Baseline |
|--------|-----------|----------|
| Approach Duration | 2.14 sec | 4.30 sec |
| Pre-Grasp Velocity | 1.124 °/frame | 0.764 °/frame |
| Grasps with Pause | 19% | 43% |

**Key Insight:** The approach phase should be long enough for the robot to align with the target. Arm should be nearly stationary when gripper closes.

---
## Directional Bias Analysis
| Direction | Your Data | Baseline |
|-----------|-----------|----------|
| LEFT | 74% | 20% |
| RIGHT | 14% | 0% |

**Warning:** Strong LEFT bias detected. This can cause the model to always swing left at inference start.

---
## Grasp Position Distribution
### Shoulder Pan Range at Grasp:
- Your data: [-50.4°, 57.7°] (range: 108.1°)
- Baseline: [-24.4°, 7.2°] (range: 31.6°)

**Note:** Limited grasp position range means the model won't know how to grasp objects outside that range.

---
## Data Collection Recommendations
### Immediate Actions:
- Slow down movements, especially during approach phase
- Open gripper earlier - at least 1 second before reaching target
- Augmentation may help but consider collecting more balanced data
- Pause for 0.3-0.5 seconds before closing gripper

### General Best Practices:
1. **Open gripper early:** At least 1 second before reaching the target
2. **Slow down approach:** Final 30 frames before grasp should be slow and deliberate
3. **Pause before grasp:** Include 0.3-0.5 seconds of stationary time before closing gripper
4. **Top-down approach:** Come from above, not from the side
5. **Balanced coverage:** Ensure blocks are placed at various positions (left, center, right)
6. **Consistent speed:** Aim for 0.75-1.0 °/frame (~22-30 °/sec)
7. **Include failures carefully:** If you miss, show a clean recovery, don't flail

---
## Visualizations
The following plots have been generated:
- `velocity_comparison.png` - Velocity distribution comparison
- `gripper_timing_analysis.png` - Gripper timing patterns
- `grasp_position_heatmap.png` - Where grasps occur in joint space
- `direction_bias_comparison.png` - Directional bias in first 100 steps
- `trajectory_comparison.png` - Sample episode trajectories

---
## Next Steps
1. Review the visualizations to understand the patterns
2. Watch sample videos from baseline dataset to see good demonstration style
3. Collect new data following the recommendations above
4. Re-run this analysis to verify improvements
5. Train with the improved dataset
