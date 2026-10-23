# Phase 4 — RBQ10 Vision Student 학습

2026-10-23 10:23 KST 실제 학습 시작. 완료 보고서가 아니다.

## 전환 판단

Teacher는 4,833회 업데이트까지 진행하고 `model_4832.pt`를 저장한 뒤 정상 종료했다. 최근 250회와 직전 250회에서 보상·일부 완주율 개선은 있었지만 큰 도약은 없었다. 시간 제약을 고려해 Student에 자원을 배정했다.

동일 seed42/nominal 조건/고정 level9, 20환경×3 episode 평가:

| Teacher | 전체 완주 | Step | 나머지 4개 지형 |
| --- | --- | --- | --- |
| 기존 model_3674 | 58/60 | 10/12 | 각각 12/12 |
| 최종 model_4832 | 55/60 | 7/12 | 각각 12/12 |

따라서 `model_3674.pt`를 증류 대상으로 선택했다. 단일 seed의 작은 표본이므로 3개 episode 차이를 일반적인 성능 열화로 단정하지 않는다. 추가 PPO가 더 좋은 정책을 만들었다는 근거가 부족하다는 판단이다. 최신 모델도 보존했다. 증거: `evaluations/teacher_to_student/decision.json`, `teacher_4832_level9.json`.

## 실제 Student 실행

- run: `2026-10-23_10-23-30_phase4_student_s1_20261023_102324_394207`
- 192환경, seed1, 실제 rollout 120 steps/환경/update, 1차 목표 **1,000 Student updates**.
- 원본 `DistillationWithExtractor`, action L2 norm + yaw L2 norm loss, depth CNN+GRU와 depth actor 공동 학습, 원본 learning rate 1e-3.
- Teacher actor는 고정. 저장된 첫 checkpoint에서 Teacher weights가 원본과 동일하고 Student actor weights는 실제로 바뀐 것을 확인했다.
- Student iteration은 Teacher의 3674에서 이어 세지 않고 0부터 별도로 센다.
- `model_0.pt` 실제 저장, depth encoder/actor 및 실제 depth optimizer state 포함 확인. 최초 snapshot의 export와 0.2초 MuJoCo 연결도 확인했으며 주행 성능 검증은 아니다.
- 50회 업데이트 시 약 115만 transitions, update당 약 4.2초. 남은 950회 약 67분은 현재 속도 기반 추정이며 실측 속도와 부하에 따라 바뀐다.

## 사용자 입력 계약 적용

- 선속도는 Student 학습에서 simulator measured body velocity를 사용한다. 실행에서는 MuJoCo 측정 또는 외부 measured xyz를 받는다. reference vx와 별도 슬롯이다.
- 지형 label은 각 학습 코스의 known flat/non-flat이다. 실험에서는 운영자가 같은 의미로 지정한다.
- 방향은 `student_direction_source=joystick` 모드로, 학습에서 주어진 waypoint 방향을 command로 actor에 유지한다. 원본 Depth 방향 예측은 auxiliary yaw loss로 학습하지만 command를 예측값으로 덮지 않는다. MuJoCo는 joystick yaw rate를 horizon 기반 방향 command로 변환한다. 실제 회전 응답은 향후 검증 대상이다.
- 발 접촉은 시뮬레이터 측정이다.
- 공식 전방 OAK-D Depth mount를 사용한다. 원본 58×87 depth 입력, 주기/지연/GRU 구조를 유지한다.
- 원본 Student의 단순화 지형, camera pitch ±5°, 마찰/질량/COM randomization을 유지한다. 추가 실측 Depth noise 모델·프레임 유실 randomization까지 완성했다는 의미는 아니다.

## 원본 변경 이유

`RBQ10StudentRunner`는 원본 distillation collection/loss/update를 재사용하면서 JSON 진행률, 안전한 update 경계 stop, 실제 depth optimizer 저장/재개, Teacher→Student counter 분리를 추가한다. 원본이 120-step rollout을 24-step으로 집계하던 transition counter만 보정했다. RBQ10 joystick 모드만 방향 command를 유지하며 Go2의 기존 동작은 유지한다.

## 높이 스캐너 분석 정정

실제 Isaac 평가의 29번 관측에서 높이값은 −1..1로 변했고 평균 std≈0.275, upper clip 비율≈5.13%였다. 원본 관측을 zero-scan으로 바꾸면 평균 raw action 절대 차이≈0.396이 발생했다. 이 값은 관절 각도 자체가 아니다. 전체 +1 포화 우려는 확인되지 않았다.

이전 설명은 `cfg.offset`(ray 시작점 이동)과 `_offset`(tracked prim 변환)을 혼동한 오류였다. 현재 MuJoCo Teacher 진단과 문서를 수정했다. Student 학습을 막는 높이 입력 포화 문제로 판단하지 않는다.

## 확인/중단/재개

```bash
./run.sh student-job status
./run.sh student-job stop
# 저장된 Student checkpoint에서 depth optimizer까지 이어가기:
./run.sh student-job start --num_envs 192 --iterations 1000 --resume /absolute/project/logs/rsl_rl/rbq10_extreme_parkour/RUN/model_N.pt
```

`logs/phase4/active_job.json`이 현재 job을 가리킨다. 해당 run의 `latest.json`, `progress.jsonl`, TensorBoard `student/.`에서 loss/완주율을 본다. Student 보상은 원본 설정상 0이므로 PPO reward 증가 대신 distillation loss와 실제 episode 완주율로 판단한다. 초반 on-policy 분포가 변해 loss는 단조 감소하지 않는다. 현재 Step/Gap/Platform 능력과 최종 Student/MuJoCo 성능은 아직 확정하지 않았다.

후속은 학습 checkpoint의 fixed-level Student 평가, 필요 시 추가 증류, 학습된 Student의 MuJoCo 평가다. Teacher 수준 성능이나 실물 적용 성공을 아직 주장하지 않는다.
