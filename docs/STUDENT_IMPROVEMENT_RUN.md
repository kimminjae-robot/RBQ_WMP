# Student 추가 증류 및 MuJoCo 전이 검증 — 2026-10-23

**13:09 KST 업데이트:** 사용자 요청으로 전방 전용 학습을 model1549(누적1550 update)에
저장·중단했다. 전체 checkpoint를 `checkpoints/rbq10_student_front/`에 보관했다.
이후에는 [전방+하향4개 5뷰 Student](https://github.com/kimminjae-robot/RBQ_WMP/blob/experiment/depth5-student-20261023/docs/MULTIVIEW_STUDENT.md)만 별도로 학습한다.
아래 내용은 기존 전방 실험의 실행 당시 기록이다.

사용자 요청에 따라 12:30 KST에 model999에서 **추가2,000 update**를 시작했다.
이전1,000 + 추가2,000 = 누적3,000이 목표다. 원본 DistillationWithExtractor와
Teacher3674를 유지하고 실제 depth optimizer/counter를 복원했다.

## 학습

- 192환경, seed1, 120step/환경/update, 기존 learning rate1e-3.
- curriculum/지형 비율/네트워크/메모리 semantics는 우선 유지한다. 추가 증류량의 효과를
  센서·물리 수정 효과와 분리해서 보기 위한 기준 실험이다.
- 저장된 model1100에서 optimizer step이1000→1101로 이어졌고,
  Teacher 가중치는 비트 단위로 같으며 Student actor/depth encoder 가중치는 실제로 바뀌었다.
  [재개 검증](../evaluations/transfer_20261023_1230/resume_verified.json).
- 학습 run: `2026-10-23_12-30-37_phase4_student_s1_20261023_123031_630135`.
- [학습 설정 기록](../logs/phase4/improvement_20261023.json),
  [TensorBoard](http://localhost:6006/#scalars)의 `student_improve_2000`.

```bash
./run.sh student-job status
# 필요한 경우 현재 update를 마치고 저장/중단:
./run.sh student-job stop
```

## 학습과 병행하는 CPU 평가

학습이 GPU 약13GB를 사용하므로 새 Isaac Sim/GUI 프로세스를 추가하지 않았다.
고정 MuJoCo 개발 세트와 설치된 Isaac Lab 센서 소스/Warp CPU kernel 비교를 실행했다.
추가 GPU 메모리를 사용하는 live PhysX·카메라 시뮬레이션의 동등성 검증은 아직 아니다.

`tools/monitor_student_transfer.py`가 같은 조건으로 누적1,000/1,500/2,000/2,500/3,000 update
시점의 저장 snapshot을 평가한다. 주기와 실제 저장 시점 때문에 model1500처럼
1,501번째 완료 snapshot이 선택될 수 있으며 정확한 checkpoint 이름/hash를 기록한다.

6종×seed42/43/44, difficulty0.5, vx0.6, 최대30초, 알려진 맵 목표점 기반이다.
단일 episode/조건의 작은 개발 세트이며 보류 테스트 성공률은 아니다.
`source` 경계/`firm` 접촉 조건을 고정하며 실행 코드가 달라지면 비교를 중단한다.
학습 자체를 중단·재시작하거나 실패를 자동 리셋하지 않는다.

[모니터 상태](../evaluations/transfer_20261023_1230/status.json),
[모니터 로그](../evaluations/transfer_20261023_1230/monitor.log).

학습 전 model999 기준선:

| 지형 | 완주/시험 |
|---|---|
| 평지 | 3/3 |
| 허들 | 3/3 |
| 턱/계단 | 1/3 |
| Gap | 0/3 |
| 징검돌 | 2/3 |
| Mixed | 0/3 |
| 합계 | **9/18** |

[각 조건 원자료](../evaluations/transfer_20261023_1230/model_999/summary.json).
새 학습의 성능이 개선됐다는 판단은 이후 snapshot과 같은 조건에서 비교한 뒤 내린다.

## 카메라 검증 결과

`tools/check_transfer_depth.py`는 설치된 Isaac Lab의 실제 intrinsics/pinhole 함수,
좌표 변환, Warp raycast kernel과 원본 parkour crop/resize/normalize/buffer 코드를 실행한다.
MuJoCo도 실제 ray를 쏜다. 동일한 body pose와 관절 상태에서6장면×4pose/시간표를 비교했다.
실제 학습된 model999의 latent/action 민감도도 같은 proprioception으로 비교했다.

| 장면 | raw Depth 평균 MAE | 최대 픽셀 오차 |
|---|---|---|
| 평면/box 허들/단차 | 0.0002~0.0004mm | 약0.0105mm |
| 원본 징검돌 simplified mesh 대2cm hfield | 0.835mm | 0.752m |
| Gap | 0.841mm | 0.418m |
| Mixed | 1.670mm | 1.241m |

평균값과 최대값을 함께 본다. 적은 수의 모서리/가림 경계 픽셀에서는 hit하는 표면이 달라져
큰 거리 차이가 남았다. 이4pose 시험의 최대 raw action 차이는 약0.0111이었다.
이를 모든 trajectory의 카메라 오차 또는 낙상의 유일한 원인으로 일반화하지 않는다.
같은 raw 입력의 전처리 일치와 camera ray 방향 오차도 assertion으로 확인했다.

[Depth 결과·소스 hash](../evaluations/transfer_20261023_1230/depth/results.json).
CPU 검증 venv에 학습 venv와 같은 fast-simplification0.2.0만 추가했다.
기존 Isaac 학습 환경과 기반 venv는 수정하지 않았다.

## 관측/구동기 검증

- 기존 실제 Isaac 상태200개로 위치·관절 제한·속도 frame 검증을 다시 통과했다.
  [결과](../evaluations/transfer_20261023_1230/state_parity.json).
- `tools/check_transfer_actuators.py`가 실제 `rbq10_parkour/robot.py` 설정을 파싱해
  kp/kd/armature/force limit과 MuJoCo 값을 비교했다. 일치했다.
- 0/5/45/60rad/s 조건의 순간 PD 토크 법칙 오차는 수치 오차 수준이다.
  이는 **implicit solver의 동적 응답이 같다는 뜻이 아니다**.
- 실제50rad/s 속도 제한은 MuJoCo에서 여전히 강제되지 않는다. 기준선18조건 중6조건에서
  최대 관절 속도50rad/s를 넘었다. 낙상 뒤의 결과인지 원인인지 시간 순서로 더 분석해야 한다.
- PhysX와 비교할0.2초 무접촉 joint step 입력/CPU MuJoCo 응답을 저장했다.
  live PhysX 응답·착지/접촉 비교와 실제 카메라 intrinsics 보정은 남아 있다.

[구동기 설정/토크 검증](../evaluations/transfer_20261023_1230/actuators/results.json),
[현재 요약](../evaluations/transfer_20261023_1230/verification_summary.json).

## 재현

```bash
CUDA_VISIBLE_DEVICES='' .venv-sim2sim/bin/python tools/check_transfer_depth.py \
 --isaaclab-source /path/to/IsaacLab/source/isaaclab/isaaclab \
 --policy-dir evaluations/transfer_20261023_1230/model_999/export \
 --output evaluations/transfer_20261023_1230/depth
CUDA_VISIBLE_DEVICES='' .venv-sim2sim/bin/python tools/check_transfer_actuators.py \
 --manifest evaluations/transfer_20261023_1230/model_999/export/manifest.json \
 --output evaluations/transfer_20261023_1230/actuators
```

진행 중인 평가 모니터를 같은 폴더에서 중복 실행하지 않는다. 이 기록은 로컬 실행 기록이며
새 학습 완료나 보류 지형 성능 달성을 의미하지 않는다.
