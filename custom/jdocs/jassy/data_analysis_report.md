# Data Quality Analysis Report: SO-ARM101 Pick-and-Place
**Generated:** 2025-12-21 15:09:27
**User Dataset:** Your Data (50 episodes)
**Baseline Dataset:** Baseline (so100_strawberry_grape) (40 episodes)

---
## Executive Summary
### Issues Identified:
- **CRITICAL**: Moving During Grasp - Arm velocity at grasp: 1.12 deg/frame vs baseline 0.32
- **Likely**: No Pause Before Grasp - Only 19% of your grasps have a pause vs 63% baseline
- **Confirmed**: Directional Bias - 74% of episodes move LEFT initially (vs 50% baseline)

---
## Hypothesis Validation
| Hypothesis | Verdict | Evidence | Recommendation |
|------------|---------|----------|----------------|
| Moving Too Fast | **FALSE** | Your velocity (2.05) is similar to or slower than baseline (2.29) | Velocity is not the issue |
| Gripper Opens Late | **NEEDS VERIFICATION** | Approach duration: your 2.1s vs baseline 2.3s | Consider opening gripper earlier for better timing |
| Directional Bias | **TRUE** | 74% of episodes move LEFT initially (vs 50% baseline) | Augmentation may help but consider collecting more balanced data |
| No Pause Before Grasp | **LIKELY TRUE** | Only 19% of your grasps have a pause vs 63% baseline | Pause for 0.3-0.5 seconds before closing gripper |
| Moving During Grasp | **TRUE - CRITICAL** | Arm velocity at grasp: 1.12 deg/frame vs baseline 0.32 | Stop arm movement before closing gripper - this is likely causing blocks to slide |

---
## Velocity Analysis
| Metric | Your Data | Baseline | Ratio |
|--------|-----------|----------|-------|
| Mean Velocity | 2.050 °/frame | 2.289 °/frame | 0.90x |
| Max Velocity | 7.56 °/frame | 43.35 °/frame | - |
| Mean (deg/sec) | 61.5 °/s | 68.7 °/s | - |

**Reference:** Good velocity is typically 0.75-1.0 °/frame (~22-30 °/sec at 30 FPS)

---
## Gripper Timing Analysis
| Metric | Your Data | Baseline |
|--------|-----------|----------|
| Approach Duration | 2.14 sec | 2.33 sec |
| Pre-Grasp Velocity | 1.124 °/frame | 0.324 °/frame |
| Grasps with Pause | 19% | 63% |

**Key Insight:** The approach phase should be long enough for the robot to align with the target. Arm should be nearly stationary when gripper closes.

---
## Directional Bias Analysis
| Direction | Your Data | Baseline |
|-----------|-----------|----------|
| LEFT | 74% | 50% |
| RIGHT | 14% | 22% |

**Warning:** Strong LEFT bias detected. This can cause the model to always swing left at inference start.

---
## Grasp Position Distribution
### Shoulder Pan Range at Grasp:
- Your data: [-50.4°, 57.7°] (range: 108.1°)
- Baseline: [39.6°, 151.4°] (range: 111.8°)

**Note:** Limited grasp position range means the model won't know how to grasp objects outside that range.

---
## Data Collection Recommendations
### Immediate Actions:
- Augmentation may help but consider collecting more balanced data
- Pause for 0.3-0.5 seconds before closing gripper
- Stop arm movement before closing gripper - this is likely causing blocks to slide

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
