# 5뷰 Student MuJoCo 전이 검증 — 2026-10-23

학습 도중 고정한 `model_400.pt`로 실제 MuJoCo 3.3.0에서 20개 조건을 실행했다.
중간 난이도 18회 중 15회, 최고 난이도 2회 중 1회 완주했다. **기본 전이는 가능하지만
Step의 반복 실패와 최고 난이도 Gap 낙하가 남아 있다.** 새 5뷰 학습은 별도로 계속 실행한다.
이는 작은 개발용 고정 조건 집합의 결과이며 일반적인 전이 성공률로 해석하지 않는다.

## 모델과 평가 조건

- 실제 체크포인트: `logs/rsl_rl/rbq10_extreme_parkour_multiview/2026-10-23_13-15-36_depth5_student_s1_20261023_131531_862725/model_400.pt`.
- SHA-256: `8f5cc545d026a0c8fb1955a45a9cb1ec069a994d43555a5ced73f546104ec2a6`.
- 새 5뷰 분기에서 401 update를 완료한 시점이다. 이전 전방 모델의 1,550 update에서
  warm start했다. 검사 중 생성된 model500 등으로 평가 대상을 바꾸지 않았다.
- 실제 Student CNN/fusion/GRU/actor를 TorchScript로 Export했다. Teacher action 대체는 없다.
  12회 recurrent sequence의 원본 대비 Export 최대 출력 오차는 **0.0**이었다.
- policy 50Hz, physics 200Hz, Depth 갱신 10Hz, 2-frame buffer의 이전 영상 사용,
  action 1 policy-step 지연, scale 0.25, 원본 raw-action 제한 ±4.8을 유지한다.
- 원본 지형 생성 함수를 재사용한다. 지형 seed 42/43/44, 중간 난이도 0.5,
  추가 최고 난이도 1.0의 seed42, 속도 명령 0.6m/s, 회당 최대 30초다.
- 목표점은 알려진 원본 코스의 현재/다음 waypoint다. 지형 종류도 알려진 flat/parkour label을
  준다. 측정 속도와 발 접촉은 MuJoCo에서 얻는다. 지형 height scan과 물성 정답은
  Student에 사용하지 않는다. 이 시험은 미지 환경의 waypoint 인식/계획 검증이 아니다.
- 완주는 순서대로 8개 목표점 도달, 실패는 roll/pitch 절댓값 >1.5rad 또는 base world-Z
  <0.15m, 미완주는 30초 제한으로 판정한다. 낙하 시 자동 reset하지 않는다.
- Isaac 영상과 동일한 모델/동일한 지형에 대한 PhysX 성공률 비교는 이번에 수행하지 않았다.
  두 엔진에서 같은 seed를 줘도 지형 생성 호출 순서가 달라 배치가 같다고 보장할 수 없다.

## 실제 결과

| 지형 | 난이도 | 완주 / 실행 | 실패 내용 |
|---|---:|---:|---|
| 평지 | 0.5 | 3 / 3 | 없음 |
| Hurdle | 0.5 | 3 / 3 | 없음 |
| Step | 0.5 | **0 / 3** | 2회 전도, 1회 목표 3/8에서 정체 |
| Gap | 0.5 | 3 / 3 | 없음 |
| Platforms / stepping stones | 0.5 | 3 / 3 | 없음 |
| Mixed | 0.5 | 3 / 3 | 없음 |
| Gap | 1.0 | **0 / 1** | 4.04초, 목표 2/8 뒤 낙하 |
| Step | 1.0 | 1 / 1 | 11.26초에 목표 8/8 |

중간 Step은 목표 3~4 부근에서 정체하거나 옆으로 밀린 후 전도했다.
seed42는 30초 뒤 base가 `(4.991, 0.884, 0.770)m`, 목표는 3/8이었다.
seed43은 6.72초, seed44는 7.54초에 전도했다. 세 경우 모두 토크 제한 150Nm에 도달했고
관절 속도 최고값은 각각 **61.68 / 61.33 / 56.49rad/s**였다. Isaac의 설정 50rad/s를
넘는 실제 구동계 차이가 확인되지만, 이것만으로 실패 원인을 단정할 수는 없다.

최고 Step 한 번의 성공으로 Step 전이가 안정적이라고 판정하지 않는다. 높이 외에도
발 디딤 위치와 운동 타이밍이 달라지므로 난이도 숫자에 따라 성공이 단조롭게 변하지 않는다.
최고 Gap은 몸체 높이가 떨어지며 두 번째 목표 이후 구간에 빠졌다.
정책 자체의 능력 부족과 엔진의 접촉/지형 차이의 기여도를 가르려면 같은 지형과 초기 상태의
Isaac/MuJoCo 짝 평가가 필요하다.

[최종 20조건 결과](../evaluations/multiview_sim2sim_model400/rollouts/summary.json),
[높은 Step 실제 MP4](../artifacts/student_depth5_model400_mujoco_step.mp4),
[높은 Step GIF](../artifacts/student_depth5_model400_mujoco_step.gif),
[최고 Gap 실패 MP4](../artifacts/student_depth5_model400_mujoco_gap.mp4),
[최고 Gap 실패 GIF](../artifacts/student_depth5_model400_mujoco_gap.gif).
두 영상의 수치도 최종 headless 평가 결과와 일치했다. GIF/MP4 끝의 정지 화면은
종료 상태 표시이며 추가 물리 시뮬레이션이 아니다.

## 카메라 반영과 입력 대조

기존 MuJoCo 경로는 전방 1개 전용이어서 5뷰 체크포인트를 실행할 수 없었다.
새 `rbq10_mujoco_student_depth5_v1` Export/입력 경로를 추가했다. 실제 MuJoCo 모델의
trunk에 5개 camera를 붙이고, `cam_xpos`, `cam_xmat`, 실제 모델 intrinsics에서 광선을
구성한다. ROS optical 규약에서 MuJoCo OpenGL camera 규약으로 회전을 변환했다.

| View | trunk 위치 x,y,z (m) | body 좌표의 optical 시선 x,y,z |
|---|---|---|
| front | 0.388031, −0.037500, 0.037764 | 1, 0, 0 |
| ground_0 | 0.364620, 0, −0.026630 | 0.342020, 0, −0.939693 |
| ground_1 | 0.260530, 0, −0.047590 | −0.241922, 0, −0.970296 |
| ground_2 | −0.195150, 0.006500, −0.048320 | 0, 0, −1 |
| ground_3 | −0.352993, 약 0, −0.020514 | −0.499820, −0.000057, −0.866129 |

- 순서: `front, ground_0, ground_1, ground_2, ground_3`. 위치 1mm 변경이나 순서가
  뒤바뀐 configuration은 거부한다. 모델·URDF·캘리브레이션·Export 파일 hash도 검사한다.
- 각 raw 영상 60×106, 원본 crop `[:-2,4:-4]`, bicubic antialias resize 58×87,
  `range / 2m - 0.5`, 최종 입력 `(1,5,58,87)`을 재현했다.
- Euclidean 거리 제한 2m, 하향 4개는 optical-Z 0.2m 미만을 invalid 처리한다.
  invalid는 2m로 채운다. 영상·행동의 지연 버퍼도 원본에 대조했다.
- 5개 영상을 하나씩 바꾸면 각 view가 실제 학습된 encoder 출력에 영향을 주었다.
  이는 카메라 입력 사용 여부 검사이며 성능 향상 측정은 아니다.
- 설치된 Isaac Lab의 실제 camera intrinsics/광선 변환, Warp raycast, 원본 front와 ground
  전처리 코드를 **CPU에서 직접 실행**해 6개 지형 × 4개 자세 × 5개 view를 비교했다.
  이는 GPU/PhysX의 닫힌 고리 동작 전체가 같다는 증거는 아니다.
- 같은 평면·박스 hurdle·박스 step에서 raw Depth 최대 차이는 **0.0123mm 이하**였다.
  같은 raw 영상의 전처리는 정확히 일치했다. 비스듬한 roll/pitch/yaw 자세도 검사했다.
- 원본 파쿠르 mesh와 MuJoCo의 2cm 경사 보정 hfield 사이에는 모서리 표현 차이가 남는다.
  지형·view별 평균 raw MAE는 약 **0.84~5.95mm**였지만, 불연속 경계 일부 픽셀은 최대
  **1.241m** 차이가 났다. 해당 자세들에서 raw action 최대 차이는 약 0.0158이었다.
  평균 오차가 작다고 모서리 영향이 없다고 결론 내리지 않는다.

[실제 5개 시야와 장착 위치 그림](../artifacts/student_depth5_model400_mujoco_cameras.png),
[카메라 대조 전체](../evaluations/multiview_sim2sim_model400/depth/results.json),
[Export 검사](../evaluations/multiview_sim2sim_model400/export/manifest.json).
시야 그림은 홈 관절 자세의 정적 진단 장면이며 주행 성공을 나타내는 영상이 아니다.
공식 장착 TF 근거와 기존 GPU sensor pose 검사는 [5뷰 학습 문서](MULTIVIEW_STUDENT.md)에 있다.

## 발견해 수정한 차이

1. **5뷰 미지원:** 전방 전용 exporter를 유지하고 별도 5뷰 exporter와 strict format을 추가했다.
   공유된 원본 CNN과 학습된 fusion/GRU/actor의 실제 가중치를 그대로 사용한다.
2. **배치 ray 조회의 표면 누락:** 설치된 MuJoCo 3.3.0의 `mj_multiRay`가 동일 박스 fixture에서
   scalar `mj_ray`와 최대 약 0.448m 차이를 보였다. 원인 미확정 상태에서 해당 조회를 사용한
   평가는 폐기 표시하고 scalar 조회로 바꿨다. 이후 동일 fixture와 전체 평가를 다시 실행했다.
   관련 기록은 `rollouts_unvalidated_batch_rays/`와 `batch_ray_rejection.json`에 보존했다.
3. **Action 제한 누락:** 원본 `DelayedJointPositionAction`은 지연된 raw action을 ±4.8로
   제한한 후 0.25를 곱한다. MuJoCo의 기존 경로에는 이 제한이 없었다. 목표 관절 위치에만
   제한을 적용하고, 관측 history에는 원본처럼 제한 전 action을 보존하도록 수정했다.
   제한 누락 상태의 scalar-ray 평가는 `rollouts_before_action_clip_fix/`로 분리했다.
   이 수정은 기존 전방 Student runner에도 동일하게 적용된다.
4. **기본 로봇/관측 재검사:** 기존 실제 Isaac 200-state trace를 MuJoCo에 대입해 joint 순서,
   관절 제한, 링크 pose와 body frame 속도를 확인했다. 위치 최대 오차 3.85µm,
   보정된 속도 좌표 변환 오차 6×10⁻¹⁶ 이하였다. 원본 R/P/K gain·armature·토크 제한과
   순간 PD 식도 대조했다. 이는 접촉 중 시간 응답의 동등성 검사가 아니다.

[상태/속도 검사](../evaluations/multiview_sim2sim_model400/state_parity.json),
[구동계 검사](../evaluations/multiview_sim2sim_model400/actuators/results.json).
원본 학습 알고리즘·Reward·Curriculum·실행 중 학습 소스/체크포인트는 수정하지 않았다.

## 실물 및 추가 전이 검증에서 남은 항목

1. **구동계:** MuJoCo에서 Isaac의 50rad/s 속도 제한이 아직 강제되지 않는다.
   실패한 중간 Step 3회 모두 이를 넘었다. qvel을 강제로 잘라 맞춘 결과를 만들지 않았다.
   같은 command의 관절 step response·접촉 impulse를 두 엔진에서 비교한 뒤
   실제 구동계에 근거해 제한/감쇠와 solver를 맞춰야 한다.
2. **모서리와 접촉:** MuJoCo hfield와 Isaac triangle mesh의 수직 모서리·마찰·충돌 여유·
   implicit drive 차이를 같은 맵에서 비교해야 한다. firm profile 한 종류만 평가했다.
3. **전방 FOV:** 장착 TF와 입력은 학습 설정에 맞지만, 전방 FOV는 기존 학습값 약
   H87°/V58°다. 실제 OAK-D Pro Wide 명목 Depth FOV H127°/V79.5°와 다르다.
   지금 모델의 입력을 임의로 넓히지 않았다. 실제 CameraInfo를 기준으로 crop/reprojection을
   정하거나 새로운 카메라 contract에서 별도 학습해야 한다.
4. **하향 센서:** 4개 D430의 위치/방향과 명목 H87°/V58°는 반영했다. 실물별 intrinsics,
   distortion, 동기화, 센서 노이즈/지연, stereo hole 및 자기 몸/다리 가림은 아직 측정하지 않았다.
   현재 두 시뮬레이터 모두 terrain-only raycast다. 실물 optical-Z를 Euclidean range로
   변환하는 처리도 실물 CameraInfo와 함께 검증해야 한다.
5. **평가 범위:** 동일 맵/초기 상태의 Isaac 짝 평가, 더 많은 seed 및 held-out 코스,
   완성된 2,000 update 모델 재평가가 필요하다. 알려진 지도 목표점을 쓰는 시연과
   조이스틱 지시만으로 주행하는 시험도 별도로 구분한다.

## 재현

MuJoCo 기본 표시 색상은 바닥 회청색, 높은 면 황토·주황·적색·보라색,
틈/구덩이 짙은 남색, 단차 경계 노란색으로 변경했다. 로봇은 흰 몸체·파란 다리·빨간 발,
목표점은 다음 목표 청록색·이후 목표 분홍색·통과한 목표 녹색이다.
열려 있는 viewer를 닫고 같은 명령으로 재실행하면 적용된다.
[변경 전후 실제 렌더](../artifacts/mujoco_colors_before_after.png)와
[색상 변경 전후 물리/5뷰 Depth 동등성 검사](../evaluations/mujoco_colors_20261023/validation.json)를 저장했다.
이 검사는 5개 장면의 물리 배열과 100 physics-step 상태 및 5뷰 Depth가 정확히 같음을 확인했다.
색상은 표면 texture와 RGBA로 표시하며 충돌 형상이나 Depth용 geometry를 추가하지 않는다.

```bash
cd ~/Desktop/coop_ws/rbq10_vision_parkour
# 실행 중 학습과 무관한 별도 MuJoCo Python 환경 사용
./run.sh export-multiview \
  --checkpoint logs/rsl_rl/rbq10_extreme_parkour_multiview/2026-10-23_13-15-36_depth5_student_s1_20261023_131531_862725/model_400.pt \
  --output evaluations/multiview_sim2sim_model400/export

# GUI 목표점 시연: 학습을 중단하지 않는다.
./run.sh sim2sim --policy-dir evaluations/multiview_sim2sim_model400/export \
  --course isaac_step --difficulty 1 --terrain-seed 42 --autonomous --viewer

# 중간 난이도 혼합 코스
./run.sh sim2sim --policy-dir evaluations/multiview_sim2sim_model400/export \
  --course isaac_mixed --difficulty .5 --terrain-seed 42 --autonomous --viewer

# 재평가 결과는 기존 기록을 덮어쓰지 않도록 새 디렉터리에 저장
CUDA_VISIBLE_DEVICES='' .venv-sim2sim/bin/python tools/evaluate_multiview_mujoco.py \
  --policy-dir evaluations/multiview_sim2sim_model400/export \
  --output evaluations/multiview_sim2sim_model400/rollouts_repeat --workers 2

.venv-sim2sim/bin/python tools/check_multiview_transfer.py \
  --policy-dir evaluations/multiview_sim2sim_model400/export \
  --output evaluations/multiview_sim2sim_model400/depth_repeat
```

이번 GitHub 배포에는 model400의 **검증한 TorchScript Export**를 위 `export/` 경로에 포함했다.
새 PC는 원본 학습 checkpoint를 준비하지 않아도 해당 Export로 재생할 수 있다.
model400을 다시 Export하려면 해당 원본 checkpoint를 별도로 복사한다. 학습 재개에는
동봉한 model700 전체 상태를 복원한다. [다른 PC 빠른 시작](DEPTH5_QUICKSTART.md).
기존 전방 model1549 복원본을 5뷰 모델인 것처럼 실행하지 않는다.
