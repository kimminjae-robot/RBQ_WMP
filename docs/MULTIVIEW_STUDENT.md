# RBQ10 전방 + 하향 4개 Depth Student — 2026-10-23

기존 전방 전용 학습은 사용자 요청으로 **13:09 KST, 누적 1,550 update**에서
`model_1549.pt`를 저장하고 정상 종료했다. Student actor/encoder뿐 아니라 Adam 상태,
다음 iteration 1550, frozen Teacher와 estimator도 포함한 **전체 재개 체크포인트**다.
원본 로그·과거 checkpoint는 삭제하지 않았다.

새로운 **5뷰 Student만 13:15 KST부터 GPU에서 실제 학습 중**이다.
192환경, seed 1, 새 분기 2,000 update가 목표다. 위 model1549를 초기화에 사용하며
분기 카운터는 0부터 시작한다. 초기 예약했던 model999 작업은 취소했다.
실행 중인 작업은 한 개이며 `multiview-student-job status`로 현재 진행을 확인한다.
이 문서와 evaluations의 숫자는 실행 시작 시점의 기록이며 학습 완료/성능 향상 판정이 아니다.

## 기존 구현 재사용과 변경 이유

- 원본 `DistillationWithExtractor`, 120 step/update, action 차이 + yaw 차이의 norm loss,
  Teacher3674, GRU512, Student actor, 지형·보상·curriculum·방향/속도 입력을 유지한다.
- 원본 58×87 Depth CNN을 5개 영상에 **공유**하고, 각 32차원 특징을 연결한 160차원을
  Linear로 32차원에 합쳐 기존 GRU에 넣는다. 새로운 카메라 정보를 수용하는 부분만 추가했다.
- 초기 fusion은 전방 identity, 하향 4개 zero다. 실제 사전학습 가중치에서 같은 입력 시퀀스의
  latent와 action이 기존 전방 모델과 정확히 같음을 검증했다. 이후 네 하향 view의 fusion이
  모두 실제 증류 update로 바뀌는 것을 확인했다.
- 구조가 늘어난 새 실험은 Adam을 새로 초기화한다. 기존 전방 checkpoint의 Adam은
  아카이브에 그대로 보존한다. 5뷰 실험 자체의 재개는 새 Adam·카운터를 복원한다.
  기존 run과 비교할 때 optimizer 재초기화와 추가 학습량의 영향도 구분해야 한다.
- CNN activation checkpointing으로 backward 때 중간값을 재계산한다. loss/gradient 목적식은
  유지하며 GPU 메모리를 줄인다. 최초 실행에서 약 7GB, update당 약 8초를 관측했다.
  따라서 2,000회는 대략 4~5시간 규모이며 실제 속도/완료 시점은 로그로 확인한다.
- 원본 crop·resize·정규화·2-frame buffer와 5 policy step마다 갱신하는 지연 입력을
  카메라별로 재사용한다. D430에는 명목 최소 optical-Z 거리 0.2m 미만을 invalid로 처리하는
  부분만 추가하고, invalid를 최대 거리 2m로 채운 후 원본 전처리를 적용한다.

## 공식 로봇과 추가 카메라 대조

추가 센서는 임의로 네 방향을 향하게 만든 카메라가 아니다. 공식 문서의 **하향 D430
Camera 0~3**을 그대로 사용했다. URDF에는 camera link가 없어 `trunk`에 센서를 추가했다.
공식 body와 URDF trunk는 네 hip offset과 질량/관성 원점이 일치한다. CoM offset을 camera
mount에 더하지 않는다. 회전은 ROS optical 규약(+Z 전방, +Y 아래)을 사용한다.

| View / 실물 ID | trunk 기준 위치 (m) | optical +Z 시선 (body x,y,z) | 명목 FOV H/V |
|---|---|---|---|
| ground_0 / sensor_0 | (0.364620, 0, −0.026630) | (0.342020, 0, −0.939693) | 87° / 58° |
| ground_1 / sensor_1 | (0.260530, 0, −0.047590) | (−0.241922, 0, −0.970296) | 87° / 58° |
| ground_2 / sensor_2 | (−0.195150, 0.006500, −0.048320) | (0, 0, −1) | 87° / 58° |
| ground_3 / sensor_3 | (−0.352993, 약 0, −0.020514) | (−0.499820, −0.000057, −0.866129) | 87° / 58° |

이는 각각 앞쪽으로 기울인 하향, 뒤쪽으로 기울인 하향, 수직 하향, 후방으로 기울인
하향 시야다. published TF 전체를 저장했고, 반올림된 회전행렬은 SVD로 가장 가까운
SO(3)에 투영했다. 회전행렬 성분 최대 수정은 약 3.9×10⁻⁷이다.

실제 GPU 환경의 sensor world pose를 다시 trunk 좌표로 역변환해 검사했다.
네 추가 카메라의 위치 오차 최대 약 **1.54μm**, 회전행렬 성분 오차 최대 **1.79×10⁻⁷**,
실제 intrinsics로 계산한 FOV **87.000001° / 57.999999°**였다. 5개 영상이 서로 다른
유한 Depth를 출력하며 실제 입력 크기는 `(192,5,58,87)`이다.

GPU 최초 검증에서는 Isaac의 camera 규약 변환에 TF32가 적용되어 회전 성분 약 6×10⁻⁵의
차이를 발견했다. 센서 구성/검증 동안 float32 matmul을 사용하도록 고쳤으며,
신경망 학습은 원본 TF32 설정을 유지한다. 실패한 초기화 로그도 보존했다.

근거: [공식 장착 TF](https://rainbowrobotics.github.io/RBQ/manual/hardware/coordinate-frames.html#ground-view-cameras-intel-realsense-d430),
[D430 사양](https://www.intel.com/content/www/us/en/products/sku/98320/intel-realsense-depth-module-d430/specifications.html),
[body frame 근거](https://rainbowrobotics.github.io/RBQ/manual/hardware/physical-parameter.html),
[실제 GPU camera 검사](../evaluations/multiview_start_20261023/hardware_camera_audit.json),
[캘리브레이션 전체](../assets/rbq10/multiview_depth_cameras.json).

## 실물과 남아 있는 차이

공개 장착 위치/방향과 추가 4개의 명목 FOV는 맞췄다. 다음 항목은 실물에서 아직 확인하지 않았다.

1. D430 개별 기기의 `CameraInfo`와 distortion, 실제 노출/FPS·지연·동기화.
   공식 최신 토픽은 `/rbq/vision/sensor_{0..3}/depth/{compressed,camera_info,tf}`이며
   front는 sensor_4다. publisher가 subscriber 연결 시에만 동작하므로 subscribe 후 측정한다.
2. Camera 3의 공식 TF y 값은 `−1.1e−050`로 인쇄되어 있다. 임의의 다른 값으로 고치지 않고
   거의 0인 원문 값을 사용했다. 실물 stream TF와 대조해야 한다.
3. 지형 전용 raycast는 자기 몸/다리의 가림, stereo hole·간섭·센서 노이즈를 재현하지 않는다.
   향후 실물 Depth 특성을 측정해 randomization 또는 mask를 적용해야 한다.
4. 원본 모델은 Euclidean `distance_to_camera`를 읽는다. 실물 optical-Z Depth는
   `CameraInfo`로 ray length로 환산한 후 같은 거리 제한·전처리에 넣어야 한다.
5. **전방 카메라** 위치와 방향은 공식 TF지만, FOV는 이번 비교에서 기존 학습값 약87°/58°를
   유지했다. 실제 OAK-D Pro Wide의 명목 Depth FOV 127°/79.5°와 차이가 있다.
   추가 4개의 FOV와 구분하며 실제 front intrinsics 보정은 별도 실험으로 남긴다.

[공식 카메라 토픽](https://rainbowrobotics.github.io/RBQ/software/sdk/ros/topics.html#vision-camera-sensors),
[OAK-D Wide 명목 FOV](https://rainbowrobotics.github.io/RBQ/manual/related-products/cameras.html).

## 기존 학습 보관과 새 학습 실행

저장소는 GitHub 단일 파일 크기 제한을 피하도록 **전체 checkpoint를 gzip 25개 조각**으로
보관한다. 각 조각·압축본·복원본 SHA-256을 검증하며 원래 checkpoint로 무손실 복원한다.
복원 시험을 실제로 통과했다. 이미 다른 checkpoint가 있으면 덮어쓰지 않는다.

```bash
python tools/restore_student_checkpoint.py
# model_1549.pt: SHA256 708a3591052e5696c92c82cfd50c387ff1be86416d5520fd43b33356c3b565bd

# 새 PC는 docs/PORTABILITY.md에 따라 Isaac Lab과 프로젝트 venv를 준비한 뒤:
./run.sh multiview-student-job start \
  --checkpoint checkpoints/rbq10_student_front/model_1549.pt \
  --iterations 2000 --num_envs 192 --seed 1
./run.sh multiview-student-job status
# 새 분기만 update 경계에서 저장/중단:
./run.sh multiview-student-job stop
```

- 태스크: `Isaac-Extreme-Parkour-Student-MultiDepth-Rainbow-RBQ10-v0`.
- 로그: `logs/rsl_rl/rbq10_extreme_parkour_multiview/`.
- 관리 상태: `logs/phase4_multiview/active_job.json`.
- TensorBoard: `http://localhost:6006/#scalars`, `student_multiview`.
- 저장 contract: `rbq10_extreme_parkour_student_depth5_v1`.

전방 전용 태스크·checkpoint·manager는 별도로 보존했다. 기존 MuJoCo front-only runner와
exporter는 5뷰 checkpoint를 거부한다. 새 5뷰 모델의 MuJoCo 카메라/입력 추가와 동일 조건
성능 평가는 이후 작업이다. 카메라가 늘었다는 이유만으로 성능 향상을 주장하지 않는다.

검증: CPU 실제 2회 증류, 현 contract의 GPU checkpoint를 CPU에서 실제 재개해
counter/Adam step 1→2 복원, 현 contract의 GPU 실제 update와
저장된 `model_0.pt`, frozen Teacher 동일성, Student 가중치/네 ground fusion 변경,
네트워크 초기 출력 동등성·TorchScript 검사.
[실제 GPU 학습 검증](../evaluations/multiview_start_20261023/gpu_training_validation.json).
[GPU→CPU 재개 검증](../evaluations/multiview_start_20261023/resume_validation.json).
