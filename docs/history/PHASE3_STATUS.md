> 과거 PC별 검증 기록입니다. 현재 상태는 [../STATUS.md](../STATUS.md)를 참고하세요. 과거 대용량 영상·로그는 이번 배포에 포함하지 않습니다.

# Phase 3 — RBQ10 Teacher 학습 시작 기록

기록 시각: 2026-10-22T17:00:19.179443+09:00. **학습 진행 중이며 Phase 3 완료 보고서가 아니다.**

## 실제 시작한 작업

- 시작: 2026-10-22T16:51:26.683319+09:00.
- 알고리즘: 원본 `PPOWithExtractor` / `ActorCriticRMA`, 지형 정답 height scan·privileged latent·state estimator·history encoder 사용.
- 규모: 원본 기본 **6,144 환경, 24 steps/환경, 50,000 iteration, seed 1**. 1회 PPO rollout은 147,456 transitions.
- 제어: Phase 2의 RBQ10 implicit PD, physics 0.005 s × decimation 4 = 50 Hz. 제조사 보행 정책으로 시작하지 않고 원본 신경망 초기화부터 실제 PPO로 학습한다.
- 원본 actor/critic 512/256/128, PPO 5 epochs·4 minibatches, 초기 learning rate 2e-4·adaptive KL, gamma 0.99·lambda 0.95, 원본 reward/가중치·5종 지형 비율·10단계 curriculum·randomization 유지.
- GUI/동영상 없이 headless. Phase 3 종료 후 결과 평가 시에만 GUI를 연다. Vision Student는 아직 시작하지 않았다.

Run: `2026-10-22_16-51-32_phase3_teacher_s1_20261022_165126_683274`

현재 console: `logs/phase3/teacher_20261022_165126_683274.log`

학습 폴더: `logs/rsl_rl/rbq10_extreme_parkour/2026-10-22_16-51-32_phase3_teacher_s1_20261022_165126_683274`

Supervisor PID `129098`, 학습 PID `129099`. 이 PID는 이 기록 시점의 값이다. 이후 상태는 `teacher-job status`로 확인한다.

이 기록 시점 업데이트 141회, 누적 20,791,296 transitions, 최근 mean reward 6.286105, mean episode length 561.23 control steps. 학습 중 관측한 최근 episode 통계는 아래와 같다. 이것은 별도 평가 환경의 점프·Climb 성공률이나 수렴 결과가 아니다. 현재 1회 업데이트 시간으로 단순 계산하면 남은 학습은 약 52.8시간이며, 실제 소요 시간은 바뀐다.

| 학습 지형 | 최근 완전한 episode 표본 | 코스 완주 비율 | 낙상 비율 | 평균 terrain level |
| --- | --- | --- | --- | --- |
| parkour_flat | 1000 | 0.931 | 0.054 | 1.082 |
| parkour_hurdle | 1000 | 0.333 | 0.280 | 0.627 |
| parkour_step | 1000 | 0.015 | 0.813 | 0.001 |
| parkour_gap | 1000 | 0.485 | 0.215 | 0.381 |
| parkour | 1000 | 0.000 | 0.872 | 0.002 |

비율과 terrain level은 이 기록 시점의 학습 episode에서 계산한다. 낮은 난이도에서의 완주가 높은 장애물의 점프/Climb/Leap 성공을 보장하지 않는다. 최종 판정은 별도 평가와 마지막 GUI 확인이 필요하다.

## 확인·중단·재개

```bash
cd ~/Desktop/coop_ws/rbq10_vision_parkour
./run.sh teacher-job status
```

`logs/phase3/active_job.json`은 현재 job 파일을 가리킨다. Job 파일의 `log_dir`에서 다음 파일을 확인한다.

- `latest.json`: 최근 완료한 update, reward·episode length·loss·지형별 통계.
- `progress.jsonl`: 매 update의 실제 측정 기록.
- `events.out.tfevents.*`: 원본 TensorBoard 곡선과 추가 측정값.
- `model_0.pt`, `model_100.pt`, …: 원본 100 iteration 간격 및 최종/정상 중단 checkpoint. 번호는 0부터 시작하므로 `model_0.pt`도 PPO 업데이트 1회 후 저장한 초기 모델이다.
- `params/`: 실제 실행한 env/agent YAML·pickle.
- `source/`: 실행 시 RBQ10 Python 소스·asset manifest·launcher·upstream patch 사본.

TensorBoard는 프로젝트 `.venv/bin/python -m tensorboard.main --logdir logs/rsl_rl/rbq10_extreme_parkour --host 127.0.0.1 --port 6006`으로 필요할 때 실행할 수 있다. 이 작업에서 웹/GUI 화면을 자동으로 열지는 않았다.

```bash
# 필요할 때만 실행: 현재 PPO update를 마치고 저장 후 종료
./run.sh teacher-job stop

# 종료 확인 후에만 실행: checkpoint에서 추가 iteration 학습
./run.sh teacher-job start --resume <이_프로젝트의_model_N.pt> --iterations <추가_iteration_수>
```

중복 관리 작업은 거부하고 `/proc` 시작 시각으로 PID가 같은 프로세스인지 확인한다. Supervisor는 프로세스 종료와 heartbeat·최신 checkpoint를 기록하며 디스크 여유가 2 GiB 미만이면 정상 중단을 요청한다. 자동으로 실패 작업을 재시작하지 않는다. 컴퓨터 재부팅·종료 후 자동 재개는 설정하지 않았다.

## 추가한 측정과 저장 기능

`rbq10_parkour/metrics.py`는 원본 auto-reset **전**에 terminal state를 읽는다. 관측값·action·reward·curriculum·reset 조건을 바꾸지 않는다. 최초 episode는 원본 runner가 episode length를 무작위 초기화하므로 각 환경의 첫 부분 episode를 제외한다. 이후 지형별 최근 최대 1,000개의 완전한 episode를 기록한다.

- `Parkour/<terrain>/course_success_rate`: 원본 최종 goal 도달 및 원본 낙상 조건에 해당하지 않는 episode 비율. goal index 8 이상을 완주로 본다.
- `fall_rate`, `timeout_rate`, `completed_episodes`, `window_episodes`: 원본 물리적 낙상 기준과 시간 제한으로 분리한다. 표본 0일 때 출력된 비율 0은 측정된 성공률이 아니다.
- `goal_index`, `terrain_level`, `episode_steps`, `episode_reward`: 진행도와 curriculum을 함께 확인한다. 명령 속도와 20초 episode 제한 때문에 완주율만으로 모든 보행 품질을 판단하면 안 된다.
- `landing_events`, `peak_foot_force_n`: 네 발 모두 비접촉이 2 제어 스텝 이상 지속된 후의 접촉 복귀 횟수와 제어 시점 최대 발 접촉력. 초기 낙하·실패 착지도 포함할 수 있어 성공적인 점프 횟수로 해석하지 않는다. Physics substep의 최고 충격력을 완전히 측정한 값은 아니다.
- `peak_estimated_effort_ratio`, `peak_velocity_ratio`, `joint_limit_violation_steps`: 공식 모터 제한과 관절 상태 확인. Implicit PD torque는 Isaac Lab 근사값이다. 공식 effort/velocity limit과 원본 collision/stumble penalty를 유지했으며 실물 근거 없이 새로운 착지력 hard limit을 정하지 않았다.

원본 종료 설정은 하나의 term에 낙상·timeout·goal 완료를 합치고 `time_out=True`로 표시한다. 이번에는 그 학습 의미를 변경하지 않고, 추가 측정에서 실제 낙상 조건(roll/pitch 절댓값>1.5 rad 또는 world z<−0.25 m)을 직접 읽어 원인을 구분했다. 원본 종료/bootstrap 의미의 적절성은 학습 결과 해석 시 검토할 항목이다.

`rbq10_parkour/runner.py`는 원본 runner를 상속한다. 원본 rollout/손실/신경망/update 순서는 재작성하지 않았다. 원본 checkpoint에서 빠진 estimator optimizer·history optimizer·PPO counter·adaptive learning rate를 추가로 보존한다. 재개 시 완료한 iteration 다음부터 시작하고 checkpoint를 임시 파일에 쓴 뒤 atomic rename한다. Simulator 물리 episode와 terrain curriculum 상태는 다시 초기화되므로 trajectory를 완전히 이어가는 기능은 아니다.

SIGINT/SIGTERM은 즉시 연산을 끊지 않고 다음 완료 update에서 저장 후 종료한다. Loss가 비유한 값이면 실패로 종료하며 성공 결과를 만들지 않는다. `latest.json`과 JSONL에도 완료한 update만 기록한다.

## 실제 검증과 오류 수정

| 시험 | 결과 |
| --- | --- |
| 1,024환경 원본 PPO 10회 | 종료 0, 실제 checkpoint의 모든 policy weight 유한값 |
| 4,096환경 + 측정·저장 확장 PPO 10회 | 종료 0, 지형별 완전한 episode 측정과 JSON/TensorBoard 확인 |
| 6,144환경 PPO 10회 | 종료 0, 실제 policy weight 유한값; GPU 약 9,788 MiB 사용 관측 |
| 실제 checkpoint 재개 | `model_9.pt`에서 다음 iteration 10으로 시작, auxiliary optimizer/counter 복원 확인 |
| 실제 안전 중단 | iteration 24 완료 후 `model_24.pt` 저장, 종료 0; 다음 시작 iteration은 25 |
| TensorBoard/구문/patch | reward·episode length·지형별 완주율/낙상률/접촉력 tag 존재, Python AST·bash·git diff 검사 통과 |

첫 4,096환경 계측 시험은 원본 지형 이름이 `(N,1)` 배열인 것을 1차원 문자열 배열로 처리해 `TypeError: unhashable type: numpy.ndarray`가 발생했다. 측정 코드에서 `.reshape(-1)`을 적용했고 이후 같은 규모 검증과 재개·본 학습이 통과했다. 실패 로그 `logs/phase3/capacity_4096.log`를 보존했다. PPO나 reward 계산식을 변경해 오류를 우회하지 않았다.

검증 근거: 검증 기록 (원PC 기록: `logs/phase3/verification.json`), 중단·재개 기록 (원PC 기록: `logs/phase3/resume_stop_validation.json`). 용량 확인용 checkpoint와 본 학습 run은 별도 폴더다.

## 변경 범위와 남은 작업

- 새 파일: `rbq10_parkour/metrics.py`, `runner.py`, `tools/teacher_job.py`.
- RBQ10만 수정: `config.py`의 Teacher 측정 활성화, `env.py`의 reset 전/step 후 측정 연결, `run.sh`의 job 관리 명령.
- Upstream `train.py`: RBQ10 Teacher만 상속 runner 선택, 학습 종료/오류 시 정리. Go2는 기존 runner를 사용한다. Phase 3 추가 patch (원PC 기록: `patches/phase3_training.patch`), 현재 upstream 전체 patch (원PC 기록: `patches/current_upstream.patch`).
- 기존 IsaacLab 소스·다른 RBQ10 프로젝트·기존 가상환경 패키지는 수정하지 않았다. Phase 1/2 보고서와 patch도 보존했다.

이후 실제 학습 곡선·지형별 진행·checkpoint를 평가해 점프/Climb/Leap와 착지 실패를 판단해야 한다. 높은 장애물 성공, 일반화, 최종 GUI, Vision Student 학습은 아직 검증하지 않았다. 본 학습을 실행 중인 상태로 남겨 둔다.


## 2026-10-22 KST — GIF 미리보기 후 재개

동시 RTX 녹화의 GPU 메모리 부족으로 미리보기 프로세스만 종료했다. Teacher를 update 경계에서 `model_881.pt`로 저장·중단한 뒤, 이 checkpoint의 실제 6초 재생을 GIF로 만들고 같은 checkpoint에서 목표 50,000 iteration을 유지해 학습을 재개했다. Simulator episode/curriculum은 재초기화된다.

재개 폴더: `/home/user/Desktop/coop_ws/rbq10_vision_parkour/logs/rsl_rl/rbq10_extreme_parkour/2026-10-22_17-45-54_phase3_teacher_s1_20261022_174548_807247`. TensorBoard에는 `before_preview`와 `resumed`가 표시된다. GIF (원PC 기록: `artifacts/rbq10_teacher_model_881.gif`), 녹화·재개 기록 (원PC 기록: `logs/phase3/gif_preview_latest.json`).
