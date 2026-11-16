# Proposed MMASD extension: RL-guided adaptive temporal sampling

## Purpose and scope

The base paper learns pose-and-motion representations with a denoising temporal autoencoder, groups the resulting embeddings into behavioural primitives, and studies their temporal transitions. The proposed extension keeps this representation-learning and analysis pipeline, then adds a reinforcement-learning policy that chooses which temporal windows to encode under a fixed observation budget.

The policy acts as an **adaptive sampler**, not as a therapist or clinical decision-maker. MMASD contains recorded sequences, not logged therapist actions with outcome-based rewards. The RL environment will therefore be an offline replay simulator built from training sessions. It can train a sampler to spend a limited feature-processing budget on windows that best preserve the observed kinematic sequence; it cannot establish that a therapy action improves a child's learning or behaviour.

## Novelty position

RL-based adaptive video frame selection has prior work, including AdaFrame, and RL has also been studied for autism intervention decisions such as selecting motivators. Therefore, this project should not claim that RL for frame selection or RL in autism is new. The potential contribution is narrower: applying policy-optimized adaptive window selection to privacy-preserving MMASD temporal representations, rewarding reconstruction gain and evaluating the resulting primitive and transition stability with held-out subject groups. A focused, systematic literature review is still needed before making any “first” or “unexplored” claim.

## Proposed pipeline

1. **Prepare each session.** Use the available pose-derived/kinematic sequence. If the paper's 2D pose and optical-flow inputs can be recovered, align them and use them as separate modalities. The current local processed data has 64 kinematic channels per time step; it does not contain the paper's matching optical-flow stream.
2. **Pretrain the representation model.** Train a denoising temporal autoencoder on the outer-fold training subjects. Use the paper's eight-frame input window, stride-four uniform windows as the initial reference, and a 128-dimensional temporal representation. The encoder maps a candidate window to a latent state. The decoder is trained to reconstruct masked or unobserved portions of the sequence.
3. **Attach an actor and critic to the encoder.** The encoder supplies state features; a separate actor head represents the policy and a critic head estimates state value. The decoder is not the policy: it reconstructs the sequence and supplies a measurable information-gain reward.
4. **Train a window-selection policy in replay.** Each episode is one training session. The agent starts at the beginning and chooses the next window stride from a small discrete action set, for example {1, 2, 4, 8} candidate windows, or stops when its observation budget is used. The simulator reveals only selected windows to the policy. It can use the complete training sequence to calculate the reconstruction reward.
5. **Optimize with PPO.** Use a categorical policy, a value head, generalized advantage estimation, and the clipped PPO objective. Train and tune the policy using training subjects and an inner validation split only. Do not use held-out subjects to choose rewards, budgets, or policy settings.
6. **Analyze selected embeddings.** Run the base paper's primitive discovery and transition-graph analysis on the policy-selected windows. Compare the resulting representation with fixed and random sampling at the same observation budget.

## Markov decision process

| Element | Proposed definition |
|---|---|
| Environment | Offline replay of one MMASD training session. The recorded sequence is fixed; the agent controls which candidate temporal window is revealed next. |
| State | Current encoder embedding $z_t$, previous selected embedding $z_{t-1}$, their temporal difference, decoder uncertainty/reconstruction residual, normalized time position, and remaining sampling budget. |
| Action | Choose the stride to the next candidate window, such as 1, 2, 4, or 8 window positions, or stop. Invalid actions near the end of a sequence are masked. |
| Transition | Advance within the recorded session and reveal the selected window. This transition is simulated from existing data; it is not an intervention with a child or therapist. |
| Reward | Improvement in full-sequence reconstruction after adding the selected window, minus a sampling-cost penalty and a redundancy penalty. |
| Policy | Actor head $\pi_\theta(a_t\mid s_t)$ attached to the pretrained encoder. The critic estimates $V_\phi(s_t)$. |
| Terminal condition | The fixed window budget is reached or the session has no valid later window. |

One candidate reward is

```text
r_t = L_rec(O_t) - L_rec(O_t union {w_t})
      - lambda_cost * cost(w_t)
      - lambda_red * max_similarity(z_t, previously_selected_embeddings)
```

Here, $O_t$ is the set of windows selected so far and $L_{rec}$ is the decoder's reconstruction error across the training session. Reward coefficients must be selected using training/validation data only. The held-out test sequence can be used for final offline evaluation, but its unobserved windows must not be provided to the policy as state inputs.

## Evaluation protocol

Use five outer folds grouped by the stored MMASD subject identifier, after checking that these identifiers correspond to distinct participants. Within every outer fold, fit normalization, the autoencoder, decoder, policy, and reward coefficients using only the outer training subjects. Report fold-level results and mean with variation, not the best fold.

Compare at the same window budget:

- uniform eight-frame windows with stride four;
- random window selection;
- a deterministic uncertainty/reconstruction-error sampler;
- the PPO adaptive sampler.

Report held-out full-sequence reconstruction error and reconstruction error versus number of sampled windows. Also report cluster stability across seeds, silhouette and Davies-Bouldin indices, transition statistics against a within-session shuffled null, and the number of windows used. Association with the 11 activity labels (ARI/NMI or downstream macro-F1) can be included as a secondary measure, clearly separated from the paper's unsupervised learning-dynamics outcomes. Re-estimate KL-divergence/joint-attention association only if verified session-aligned joint-attention measurements are available.

## Hypothesis and interpretation

At the same number of sampled windows, the policy may reduce held-out reconstruction error and retain more stable latent primitives than uniform or random sampling. This is a hypothesis, not a result. The reward measures information preservation in recorded kinematics; it is not a clinical reward and must not be described as therapy progress or improved autism outcomes.

The project should claim a contribution only if the RL sampler improves prespecified held-out metrics over the equal-budget baselines without reducing seed or subject stability. Until that experiment is run, describe this section as a proposed method rather than completed work.

## Related primary sources

- Akinwumi, Qian, and Ojo (2026), the MMASD learning-dynamics base paper: https://doi.org/10.3389/fnins.2026.1737098
- Wu et al. (2019), AdaFrame, which uses policy-gradient learning for adaptive video-frame selection: https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_AdaFrame_Adaptive_Frame_Selection_for_Fast_Video_Recognition_CVPR_2019_paper.pdf
- Sun, Qian, and Miao (2024), CUDC, which uses adaptive temporal distances in unsupervised data collection for offline RL: https://doi.org/10.1609/aaai.v38i13.29437
- Kohls et al. (2022), an autism-intervention application of Q-learning for motivator selection; this is an intervention decision task, distinct from adaptive video sampling: https://doi.org/10.1007/s10209-022-00914-7

## Implementation status (12 November 2026)

Initial experimental scripts have been added:

- `experiments/mmasd_adaptive_sampling_rl.py` implements PPO-based adaptive sampling over MMASD kinematic windows and compares it with equal-budget uniform, random, and deterministic novelty sampling. It uses five folds based on the stored group identifiers and writes fold-level CSV and summary JSON results.
- `experiments/avasd_adaptive_sampling_rl.py` applies the same sampler to chronological AV-ASD audio segments and evaluates one majority-label sample per video. The available local AV-ASD bundle contains audio descriptors but no aligned visual or pose feature matrix, so this is explicitly an audio-only experiment. Its video-level folds do not establish participant-independent generalization because participant identities are not verified.

Both scripts have now been executed with the project's bundled PyTorch and NumPy libraries. Under the current preliminary setup, MMASD PPO obtained 66.62% mean accuracy and 0.5701 mean macro-F1 across five stored-group folds; uniform, random, and heuristic sampling scored 67.29% / 0.5762, 67.80% / 0.5939, and 67.65% / 0.5843, respectively. PPO therefore did not improve the tested MMASD classifier. On AV-ASD, the audio-only video-level PPO run obtained 60.98% / 0.4941; uniform sampling scored 62.22% / 0.5067. These preliminary runs do not establish improvement over published models and must not be presented as such.

The fold outputs are saved in `results/mmasd_adaptive_sampling_rl_folds_2026_11_12.csv` and `results/avasd_adaptive_sampling_rl_folds_2026_11_12.csv`; aggregate JSON files are saved beside them. They use a small PyTorch classifier and a descriptor-novelty reward, so the results assess the first sampler prototype, not a stronger backbone. The grouping caveats and AV-ASD audio-only limitation still apply. Further policy/reward development must be selected on training data, then evaluated once with a frozen outer-fold protocol; do not tune against these reported test-fold scores.
