> 과거 PC별 검증 기록입니다. 현재 상태는 [../STATUS.md](../STATUS.md)를 참고하세요. 과거 대용량 영상·로그는 이번 배포에 포함하지 않습니다.

# Phase 2 — RBQ10 로봇 이식과 기본 물리 검증

검증일: 2026-10-22 KST. 작업 경로: `/home/user/Desktop/coop_ws/rbq10_vision_parkour`.

사용자의 Phase 2 진행 승인에 따라 공식 RBQ10 로봇을 별도 환경으로 이식했다. 실제 Isaac Sim에서 정지 자세, 평지 보행, 낮은 턱과의 물리 상호작용을 확인했다. **RBQ10 파쿠르 Teacher/Student 학습은 수행하지 않았다.** 보행 검증에는 제조사가 공개한 보행 정책을 사용했고, 정지 검증은 공식 초기 자세를 유지하는 PD 제어다. Phase 1 결과는 기존 보고서 (원PC 기록: `PHASE1_REPORT.md`)에 보존했다.

## 환경과 원본 재사용

Phase 1에서 확인한 Isaac Lab v2.3.2 / Isaac Sim 5.1.0 / Python 3.11.17 / PyTorch 2.7.0+cu128 / 프로젝트 전용 RSL-RL 2.3.3 환경을 그대로 사용했다. Phase 2에서 기존 가상환경의 패키지를 교체하거나 다른 RBQ10 프로젝트를 수정하지 않았다. 기존 IsaacLab checkout의 `demo/`, `workspaces/`도 유지했다.

Isaaclab_Parkour 기준 commit: `d6766d8883cffe427f2d4c087717d4eca1f23d8c`.

`rbq10_parkour/config.py`와 `agents.py`가 원본 Go2 설정과 runner 설정을 상속한다. 원본 `ParkourManagerBasedRLEnv`, 지형 생성기, reward 계산식과 가중치, curriculum, PPO, RMA/history encoder, estimator, Student depth encoder와 distillation 손실을 재사용한다. 원본 Go2 설정 파일은 수정하지 않았다. 공통 함수에 기본값이 `base`인 body 이름 매개변수를 추가해 RBQ10의 `trunk`를 선택하게 했다.

Teacher/Student 각각 Train·Play·Eval의 RBQ10 ID 6개를 추가했다. 기존 Go2 6개를 포함한 12개 설정을 실제 로딩했다. 실제 Teacher/Student 학습 설정으로 각각 4개 환경을 초기화하고 policy observation `[4,753]`, action 12개, Student depth `[4,58,87]`의 유한값을 확인했다. 초기화 시 학습 설정의 질량 randomization은 활성화돼 있으므로 명목 질량과 차이가 나는 것이 정상이다.

## 공식 모델 출처와 비교

공식 [RainbowRobotics/RBQ](https://github.com/RainbowRobotics/RBQ/tree/c7b80ba49f9a79f983cb977e2726cb2246701fcf), commit `c7b80ba49f9a79f983cb977e2726cb2246701fcf`에서 필요한 파일만 가져왔다.

- 모델: `resources/urdf/rbq10.urdf`, 해당 URDF가 참조하는 STL 5개. 원본 URDF의 질량·관성·관절 제한·충돌 형상을 변경하지 않았다.
- Isaac Lab 참고 설정: `rbq_simulator/rbq_lab/rbq_lab/envs/rbq10/rbq10.py`.
- 실제 보행 baseline: `rbq_simulator/rbq_gym/policy/rbq10/{policy.onnx,info.json}`.
- 원본 모델 안내문과 파일별 다운로드 URL·SHA-256: asset manifest (원PC 기록: `assets/rbq10/manifest.json`), 원본 안내문 (원PC 기록: `assets/rbq10/official/MODEL_README.md`).

| 항목 | 원본 Go2 파쿠르 | RBQ10 이식 |
| --- | --- | --- |
| 모델 | Unitree Go2 USD | 공식 팔 없는 RBQ10 URDF → 프로젝트 내부 USD |
| Base | `base` | `trunk` |
| 질량·관성 | Go2 asset | URDF 합계 40.683936 kg; PhysX 40.683929 kg; 모든 링크 관성 양의 정부호 확인 |
| 링크·관절 | Go2 명칭 | 17 링크, 회전 관절 12개, fixed foot 4개 |
| 초기 자세 | Go2 기본 자세 | R=0; 후방 P=46.5°, K=−80°; 전방 P=45.5°, K=−82°; spawn z=0.55 m |
| 액추에이터 | 원본 `ParkourDCMotor`, 속도별 토크 포화 | 공식 RBQ Lab의 implicit PD, R/P Kp=123.39138, K Kp=127.77215, Kd=2.5 |
| 토크·속도 제한 | R/P/K effort 35/40/40 Nm, velocity 52.4/30.1/30.1 rad/s | R/P/K effort 100/100/150 Nm, velocity 50 rad/s; `_sim` 필드로 PhysX에 명시 |
| Armature | Go2 설정 | R/P=0.014058, K=0.0214816 |
| Action·관측 | action 12, policy observation 753 | 같은 차원 유지; RBQ10 native 관절 순서 사용 |
| 파쿠르 Action scale | 0.25 rad | 0.25 rad 유지; 공식 보행 검증만 원본 보행 scale 0.1을 명시적으로 변환 |
| 학습 시간 간격 | physics 0.005 s × decimation 4 | 그대로 유지: 제어 50 Hz |
| Foot 접촉 | `FL/FR/RL/RR_foot` | `HR/HL/FR/FL_foot`, 이름으로 robot/sensor 인덱스 각각 탐색 |
| Depth 장착 | base 기준 `(0.33,0,0.08)` m | trunk 기준 `(0.42,0,0.10)` m; 실물 미보정 시뮬레이션 초기값 |

관절 제한은 URDF와 실제 PhysX 값을 관절마다 대조했다. R축은 좌우에 따라 `[-0.733,0.5411]` 또는 `[-0.5411,0.733]` rad, P축은 ±6.2832 rad, K축은 `[-2.8274,-0.4189]` rad이다. Soft limit factor 0.9를 유지했다.

공식 보행 모델의 순서는 다음과 같다.

```text
joint0_HRR, joint1_HRP, joint2_HRK,
joint3_HLR, joint4_HLP, joint5_HLK,
joint6_FRR, joint7_FRP, joint8_FRK,
joint9_FLR, joint10_FLP, joint11_FLK
```

실제 PhysX와 파쿠르 정책이 사용하는 순서는 다음과 같다.

```text
joint0_HRR, joint3_HLR, joint6_FRR, joint9_FLR,
joint1_HRP, joint4_HLP, joint7_FRP, joint10_FLP,
joint2_HRK, joint5_HLK, joint8_FRK, joint11_FLK
```

공식 모델 순서를 native tensor에서 읽는 인덱스는 `[0,4,8,1,5,9,2,6,10,3,7,11]`이다. 실제 이름 순서를 assert하고, checkpoint에 모델 hash·관절 순서·제어 설정·카메라 정보를 함께 저장하도록 했다. 차원이 같다는 이유로 Go2 가중치를 RBQ10에 넣지 않으며, 잘못된 체크포인트는 로드 전에 거부한다.

공식 Lab 설정과 달리 `merge_fixed_joints=False`를 사용한다. 네 발이 독립 접촉 body로 남아 원본 파쿠르의 foot reward와 contact observation을 재사용하기 위해서다. 별도 발 링크 질량을 포함한 총질량이 원본과 일치함을 확인했다. 모터 모델은 Go2용 토크 곡선을 가져오지 않고 RBQ10 공식 implicit PD를 사용했다. 이 차이는 향후 실제 모터 응답과 Sim2Sim에서 재검증할 대상이다.

Depth는 원본 RayCasterCamera와 광학 파라미터를 유지한다: 원본 60×106, 전처리 58×87, 최대 거리 2 m, 원본의 하향 장착 회전과 2-frame buffer. 실제 raw depth 표준편차 약 0.402 m, 전처리 표준편차 약 0.196으로 상수 영상이 아님을 확인했다. 카메라 위치는 로봇 전단부를 고려한 시뮬레이션 값이며 실물 장착 위치 측정은 아직 없다. RGB 렌더링 카메라로 대체하지 않았다.

## 실제 물리 검증 결과

모든 시험은 seed 1, 환경 4개다. 검증용 flat/step 설정만 push·질량/COM randomization·관절 reset noise·action delay를 끄고 마찰 0.9로 고정했다. 학습용 curriculum이나 randomization 설정을 이렇게 단순화하지 않았다. 중간 시험은 headless로, 마지막 step 시험은 GUI와 영상 저장으로 수행했다.

| 시험 | 제어 주기 / 시간 | 결과 | 주요 측정 |
| --- | --- | --- | --- |
| 공식 home 자세 PD | 100 Hz / 10 s | 4/4 통과 | 최대 tilt 0.0234 rad; 수평 이동 약 0.0567 m |
| 파쿠르 학습 주기의 home PD | 50 Hz / 10 s | 4/4 통과 | 최대 tilt 0.0229 rad; 수평 이동 약 0.0635 m |
| 공식 정책 평지 보행 | 100 Hz / 12 s | 4/4 통과 | 전진 3.597–3.601 m; 최대 tilt 0.0529 rad |
| 공식 정책 낮은 step, GUI | 100 Hz / 12 s | 4/4 통과 | 전진 3.537–3.550 m; 높이 변화 0.106–0.113 m; 최대 tilt 0.0659 rad |

Step은 원본 `parkour_step` 생성기를 사용해 단차 5 cm, 최고 15 cm인 낮은 계단을 만든 것이다. 12초 동안 올라가는 구간의 높은 발 접촉과 전진을 확인했다. 전체 파쿠르 코스 완주나 높은 턱 Climb 성능을 측정한 시험은 아니다.

보행 명령은 처음 2초 정지 후 1초 동안 0.4 m/s까지 증가하며 명령 적분 거리는 약 3.8 m다. 보행 각 환경의 발 착지 횟수 50–52, step은 51–54회를 기록했다. 네 환경 모두 넘어짐·episode reset·관절 위치/속도 제한 위반 표본이 없었다. 정지 상태의 네 발 접촉력 합은 약 399 N으로 명목 중량과 일치했다. `applied_torque` 최대값은 평지 약 47.63 Nm, step 약 55.09 Nm였다. **Implicit PD의 `applied_torque`는 Isaac Lab이 계산하는 근사값**이며 PhysX solver의 모터 토크 직접 측정은 아니다. 모터 effort limit은 PhysX 설정에도 명시했다.

성공 판정은 실제 상태에 기초한다: z<0.25 m 또는 tilt>0.8 rad의 낙상 표본, reset, 관절 제한 위반이 없어야 한다. 정지는 수평 이동<0.2 m 및 tilt<0.25 rad, 보행은 명령 적분 거리의 60% 이상 전진과 발 착지>4회, step은 추가로 z>0.06 m인 발의 접촉력>5 N을 요구한다. 이는 작은 모델 검증 시험이며 통계적 파쿠르 성공률을 뜻하지 않는다.

공식 보행 모델은 45→512→256→128→12의 ELU 신경망이다. ONNX의 실제 가중치를 동일한 PyTorch 연산으로 읽어 GPU에서 추론했고, ONNX reference와 32개 입력에서 최대 오차 `6.1035e-05`를 확인했다. 별도 학습이나 임의 가중치를 사용하지 않았다. 이 baseline은 depth를 입력하지 않는다. 시뮬레이션의 depth 파이프라인이 동작하는 것과 RBQ10 Vision Student가 학습된 것은 별개다.

학습 설정은 원본 50 Hz를 유지하지만 제조사 보행 정책은 공식 `info.json`의 100 Hz로 검증했다. 검증 JSON의 `physics_dt`/`control_dt`는 해당 시험의 실제 값이고, 중첩 `robot_contract`는 앞으로 학습할 파쿠르 모델의 50 Hz 계약이다. 정지 상태는 두 주기에서 확인했지만 50 Hz RBQ10 파쿠르 보행 정책은 Phase 3에서 학습해야 한다.

## 오류, 수정, 회귀 검사

- 원본 observation과 feet-edge reward의 `base` 선택을 기본값이 있는 매개변수로 바꿨다. RBQ10만 `trunk`를 지정하고 Go2는 종전 동작을 유지한다.
- 원본 Go2의 hip/foot 명칭과 기본 actuator를 RBQ10 공식 명칭·설정으로 교체했다. Robot body 인덱스와 contact sensor 인덱스가 다르므로 별도 이름 탐색을 사용한다.
- 첫 GUI 영상 시험은 simulation 7.4초 부근에서 Kit render/update 내부에 멈췄다. SIGUSR1 Python stack을 기록하고 해당 프로세스만 종료했다. 이미 렌더링된 프레임을 `render(recompute=True)`로 읽도록 바꾼 후 최종 1,200스텝이 종료 코드 0으로 완료됐다. 최초 정지의 외부 원인까지 확정한 것은 아니다. 실패 로그와 부분 영상은 `logs/phase2/step_gui*`에 보존했다.
- Go2 Teacher 체크포인트를 RBQ10 재생 명령에 넣은 시험은 simulator 시작 전에 명확한 contract mismatch로 종료 코드 1을 냈다. 실제 RBQ10 runner의 save/load에도 같은 확인을 적용했다.
- Go2 설정 생성 전후 fingerprint가 동일하며, 기존 Go2 Student 체크포인트는 headless 4개 환경·50스텝 실제 재생이 종료 코드 0으로 끝났다. 기존 모델을 다시 학습하거나 교체하지 않았다.
- 새 Python 파일 syntax, 런처 shell syntax, upstream `git diff --check`를 확인했다.

측정 파일: asset 검사 (원PC 기록: `logs/phase2/asset_audit.json`), Teacher 초기화 (원PC 기록: `logs/phase2/teacher_probe.json`), Student 초기화 (원PC 기록: `logs/phase2/student_probe.json`), 정지 (원PC 기록: `logs/phase2/stand.json`), 50 Hz 정지 (원PC 기록: `logs/phase2/stand_parkour_rate.json`), 평지 보행 (원PC 기록: `logs/phase2/walk.json`), 최종 step (원PC 기록: `logs/phase2/step_gui_v2.json`), Go2 회귀 로그 (원PC 기록: `logs/phase2/go2_regression.log`). 각 물리 시험의 `.jsonl`에는 실제 위치·발 높이·접촉력·자세·waypoint 기록이 있다.

최종 RBQ10 GUI 녹화 (원PC 기록: `artifacts/rbq10_step_gui.mp4`): 960×540, 25 fps, 300 frames, 12초. 초기 첫 프레임은 렌더링 초기화 때문에 검정이며 이후 실제 프레임을 읽어 확인했다.

## 변경 파일과 실행법

| 파일 | 역할 |
| --- | --- |
| `assets/rbq10/` | 고정 commit의 공식 URDF/STL/PD 참고 설정/실제 보행 ONNX, 출처와 hash |
| `rbq10_parkour/__init__.py` | 기존 Go2와 별도인 RBQ10 Gym 등록 |
| `rbq10_parkour/robot.py` | 공식 모델·초기 자세·PD·충돌/foot 설정 |
| `rbq10_parkour/config.py` | 원본 환경 상속, robot/selector/camera 교체, 작은 물리 검증 지형 |
| `rbq10_parkour/env.py` | 원본 환경 상속과 실제 관절/foot/action 계약 확인 |
| `rbq10_parkour/agents.py` | 원본 PPO/distillation 설정 상속, RBQ10 전용 로그·checkpoint 계약 |
| `rbq10_parkour/contract.py` | 모델 hash·순서·설정 및 잘못된 checkpoint 거부 |
| `rbq10_parkour/factory.py` | 실제 제조사 ONNX 가중치의 검증용 추론과 수치 비교 |
| `tools/task_entry.py` | RBQ10 등록 후 원본 train/play 진입점 실행 |
| `tools/validate_rbq10.py` | 실물리 초기화·정지·보행·step 시험, 측정·실패 판정·선택적 녹화 |
| `run.sh` | 기존 Go2 명령 유지, RBQ10 전용 명령 추가 |
| upstream `.../mdp/{observations,rewards}.py` | body 이름 매개변수 추가, 기본 Go2 유지 |
| upstream `.../on_policy_runner_with_extractor.py` | RBQ10 checkpoint identity 저장·검사 |

Phase 2만의 upstream 수정은 `patches/phase2_port.patch`, Phase 1+2 전체 수정은 `patches/phase1_phase2_combined.patch`에 있다. 프로젝트 바깥에 만든 새 패키지는 위 표의 소스 파일로 관리한다. Phase 1 patch도 보존했다.

```bash
cd ~/Desktop/coop_ws/rbq10_vision_parkour

# GUI 없이 모델/센서 초기화
./run.sh validate-rbq10 --mode probe --role Student --num_envs 4 \
  --headless --output logs/phase2/manual_probe.json

# 파쿠르 학습 주기에서 정지 확인
./run.sh validate-rbq10 --mode stand --parkour_timing --num_envs 4 \
  --seconds 10 --headless --output logs/phase2/manual_stand.json

# 공식 보행 정책으로 낮은 step 시험; 영상은 기존 artifacts 파일로 확인 가능
./run.sh validate-rbq10 --mode step --num_envs 4 --seconds 12 \
  --headless --output logs/phase2/manual_step.json
```

GUI가 필요하면 마지막 명령의 `--headless`를 빼고, 필요 시 `--video artifacts/manual_step.mp4 --kit_args '--/app/window/width=1280 --/app/window/height=720'`를 붙인다. 사용자 요청에 따라 앞으로 중간 검증은 headless로 수행하고 각 Phase 마지막에만 GUI를 보여준다.

등록된 RBQ10 학습 명령은 `./run.sh train-rbq10-teacher`, 이후 `./run.sh train-rbq10-student --load_run <완성된_RBQ10_Teacher_run> --checkpoint <model_name>`이다. **Phase 2에서는 실행하지 않았다.** 원본 Teacher 기본 6,144개 환경은 노트북에 그대로 적용하지 말고 Phase 3에서 메모리와 학습 설정을 확인해야 한다. RBQ10 학습 완료 checkpoint가 없으므로 RBQ10 parkour play 명령은 아직 학습 결과 재생에 사용할 수 없다.

## 다음 단계에 남은 것

Phase 3에서 위 이식 설정으로 실제 RBQ10 Teacher를 학습하고, 큰 로봇의 몸체·토크·착지 충격에 맞는 결과를 평가해야 한다. Reward/terrain/curriculum은 우선 원본을 유지했으며 높은 점프·Climb·Leap의 학습 가능성은 아직 검증하지 않았다. Phase 4에서는 원본 Student에 남은 privileged base velocity/terrain-type 정보의 실센서 대체, depth noise와 지연을 감사해야 한다. 카메라 실물 보정과 recurrent ONNX 배포 계약은 이후 단계다. 여기서 Phase 2 작업을 마친다.
