> **지형 추가:** 원본 Isaac 생성 함수를 재사용한 6종×3난이도와 전체 조감도를 준비했다. [맵 사진·실행 방법·물리 표현 차이](MUJOCO_TERRAINS.md).

# MuJoCo Student Sim2Sim

2026-10-23 사용자 지정 반영. 기본 `sim2sim`과 `export-sim2sim`은 이제 **RBQ10 Vision Student** 기준이다. 기존 Teacher 도구는 별도 진단용으로만 남겨 둔다. 진행 중인 Isaac Teacher 학습은 변경하지 않았다.

## 현재 입력 계약

| 입력 | MuJoCo | 실제 로봇 연결 |
| --- | --- | --- |
| reference vx | CLI/조이스틱/JSON, m/s, 상태 index 10 | 조이스틱 전진속도 명령 |
| reference yaw rate | 조이스틱/JSON, rad/s | 조이스틱 회전속도 명령 |
| measured linear velocity | 기본 MuJoCo body-frame xyz 또는 외부 JSON | 사용자가 공급하는 측정 선속도, body-frame xyz m/s |
| 지형 종류 | `--terrain-label flat` 또는 `parkour`, 코스와 독립된 수동 값 | 학습과 맞춰 구성한 실험 코스의 사전 지정 값 |
| 발 접촉 | MuJoCo contact force, HR/HL/FR/FL 순서 | 추후 contact estimator 연결 |
| depth | MuJoCo terrain ray distance → 원본 전처리 | 실제 Depth 카메라 입력으로 대체 |
| 관절/IMU/이전 action | MuJoCo 상태·정책 이력 | 실제 encoder/IMU/action 이력 |

원본 actor에는 선속도 명령과 측정 선속도가 별도 위치에 있다. **reference vx를 측정값 대신 넣지 않는다.** 측정값 xyz는 2배 scaling 후 185:188에 넣고 padding 6개는 0으로 유지한다. 별도 velocity estimator는 필수가 아니다.

원본 actor는 yaw rate를 직접 받는 대신 현재/다음 방향 오차 2개를 받는다. 기본 joystick 모드는 `yaw_rate × yaw_rate_to_error_seconds`를 두 방향 오차에 넣고 ±1.5rad로 제한한다. 기본 horizon은 0.5초이고 설정 파일에서 바꿀 수 있다. waypoint 정답은 사용하지 않는다. **이 변환은 인터페이스 정의이며 해당 회전속도를 정확히 추종한다고 검증된 것은 아니다.** 기존 Student의 waypoint 기반 학습과 함께 회전 응답 검증이 필요하다. `--direction-source vision`은 원본 Depth 방향 예측 경로 비교용이다.

원본 명령에는 lateral vy 입력이 없으므로 JSON의 nonzero vy는 거부한다. 조이스틱 역방향은 입력할 수 있으나 원본 학습의 forward vx 0.3–0.8m/s 범위 밖이므로 별도 성능 검증이 필요하다.

## 카메라

공식 [Coordinate Frames 문서](https://rainbowrobotics.github.io/RBQ/manual/hardware/coordinate-frames.html#front-and-rear-cameras-oak-d-pro-wide)의 **전방 OAK-D Pro Wide Depth TF**를 사용한다. RGB TF나 하향 D430 TF가 아니다.

- trunk/body 기준 translation: `(0.388031, -0.037500, 0.037764)m`
- ROS optical quaternion `(w,x,y,z)`: `(0.5,-0.5,0.5,-0.5)`
- 광학 축 매핑: camera x→body −Y, camera y→body −Z, camera z→body +X. 광축은 정면이며 기존 임시 하향 pitch를 제거했다.
- 제조사 base/body frame은 관절 원점/질량 중심이 일치하는 URDF `trunk`에 대응시킨다.

원본 TF와 출처는 `assets/rbq10/front_depth_camera.json`에 저장한다. Isaac Student는 이 파일에서 position/rotation을 읽어 `offset.convention="ros"`로 적용하고, MuJoCo의 `student_config.json`도 동일한 값을 사용한다. 기본 카메라는 URDF 링크 유무와 무관하게 제조사 문서로 지정한다.

새 Student 계약은 `rbq10_extreme_parkour_student_front_depth_v2`이며 camera quaternion도 기록한다. export/실행에서 계약과 카메라 설정을 비교한다. 카메라를 사용하지 않는 기존 Teacher checkpoint 계약은 정확히 유지해 현재 학습과 resume에 영향을 주지 않는다. 원본 distillation의 Teacher→Student 초기화만 명시적으로 허용하고 Student playback/export에 Teacher를 쓰는 것은 거부한다.

이번 변경은 **장착 TF**를 맞췄다. 입력 해상도, intrinsics/FOV, 최대 거리, 전처리는 원본 학습 값을 유지하며 실제 OAK-D의 개별 intrinsic calibration을 적용했다는 의미는 아니다.

Depth는 raw 60×106, ray distance 최대 2m, crop `[:-2,4:-4]`, bicubic antialias resize 58×87, `distance/2−0.5`를 유지한다. 원본과 마찬가지로 terrain만 보며 robot/self 및 동적 물체는 제외한다. TorchVision 원본 resize와 MuJoCo 전처리의 fixture 오차는 0이었다. GUI 렌더러 없이 CPU ray로 생성한다.

시간·메모리는 이번 범위에서 변경하지 않았다. 기존 50Hz 정책, 10Hz depth, 2-frame depth buffer의 한 프레임 지연, action 1-step delay, 내부 GRU state를 유지한다. 매 rollout은 depth 모듈을 새로 로드한다.

## 실행

**RBQ10 Student 학습을 2026-10-23 10:23 KST에 시작했다.** 초기 실제 Student checkpoint의 export와 짧은 MuJoCo 입력 연결도 확인했지만 완성 정책의 주행 성능 검증은 아니다. 아래 export/rollout은 검증할 실제 Student checkpoint를 지정한다. checkpoint가 없거나 Teacher/Go2 checkpoint이면 명확한 오류를 반환한다. Teacher를 Student로 대신 실행하지 않는다.

```bash
cd ~/Desktop/coop_ws/rbq10_vision_parkour
./run.sh export-sim2sim --checkpoint /path/to/real_rbq10_student.pt

# 평지 실험: 지형 라벨은 자동 판정이 아니라 운영자가 지정
./run.sh sim2sim --course flat --terrain-label flat --seconds 10

# 학습과 맞춘 parkour 코스 라벨
./run.sh sim2sim --course step --terrain-label parkour --seconds 20

# Linux joystick: axis1 전진(최대 ±0.8m/s), axis0 회전(최대 ±1rad/s)
./run.sh sim2sim --joystick /dev/input/js0 --terrain-label parkour --viewer --seconds 60

# 사용자가 전달하는 command + 측정 선속도
./run.sh sim2sim --input-json sim2sim/input_example.json --velocity-source external --terrain-label parkour

# CPU 인터페이스/전처리/네트워크 직렬화 검증
.venv-sim2sim/bin/python sim2sim/check_student_bridge.py
```

`input_example.json` 형태:

```json
{
  "command": {"vx": 0.6, "vy": 0.0, "yaw_rate": 0.0},
  "measured_velocity_body_m_s": [0.0, 0.0, 0.0]
}
```

JSON은 매 policy step 다시 읽는다. 외부 프로그램은 임시 파일 작성 후 atomic rename으로 갱신한다. 측정값은 예시의 0을 실제 측정값으로 교체한다. `--velocity-source external`에서는 측정값 누락/NaN/잘못된 차원을 거부한다. 기본 `mujoco`에서는 시뮬레이터 측정값을 쓴다. 실제 joystick 축 매핑은 장치에 맞춰 확인해야 하며 현재 연결된 joystick으로 시험한 것은 아니다.

## 파일과 검증 범위

- `tools/export_sim2sim.py`: 원본 Student `depth_actor_state_dict`, `depth_encoder_state_dict`를 strict load한다. 실제 로봇 계약과 URDF hash도 검증한다. `student.pt`(obs, depth latent)와 `depth.pt`(depth, proprioception; 원본 내부 GRU state)를 생성한다.
- `sim2sim/run_student.py`: 기본 실행기. Teacher terrain scan은 생성하지 않고 actor에도 공급하지 않는다. measured velocity, 수동 label, joystick 방향을 넣으며 contact는 MuJoCo에서 생성한다.
- `sim2sim/student_inputs.py`: live JSON/Linux joystick/측정 선속도/카메라 전처리.
- `sim2sim/student_config.json`: 카메라와 입력 선택, 방향 변환 horizon.
- `sim2sim/check_student_bridge.py`: 실제 정지 모델의 네 발 contact, 명령/측정값 분리, 수동 label, 카메라 ground ray와 resize, checkpoint 거부를 검사한다.

공개 Go2 Student 체크포인트가 있으면 이를 **오프라인 네트워크 직렬화 검증 fixture로만** 사용한다. 8-step recurrent export 비교 오차는 0이었고 저장/재로드 초기 state도 확인했다. Go2 가중치로 RBQ10을 제어하거나 RBQ10 Student 성공 결과를 만들지 않았다. 실제 RBQ10 Student 성능 검증은 아직 수행하지 않았다. 학습 시작 후 초기 checkpoint의 export/0.2초 연결 확인은 성능 평가에 포함하지 않는다. 증거: `evaluations/sim2sim_student_interface/checks.json`.

## 별도 실행 환경

현재 PC의 `.venv-sim2sim`은 독립 3.11 venv이며 기존 Isaac Python site-packages를 읽는 `.pth`가 있다. MuJoCo 3.3.0만 별도로 설치했고 학습 venv는 변경하지 않았다. 다른 PC는 기존 설치 문서로 원본 코드를 준비한 뒤:

```bash
python3.11 -m venv .venv-sim2sim
.venv-sim2sim/bin/pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
.venv-sim2sim/bin/pip install -r sim2sim/requirements.txt
# export도 할 때만 필요. 이미 받은 export 파일을 실행할 때는 불필요.
.venv-sim2sim/bin/pip install rsl-rl-lib==2.3.3
```

검증 도구는 기존 Isaac 환경의 torchvision도 사용한다. 독립 환경에서 검증하려면 torch2.7 대응 torchvision0.22를 설치한다. rollout 자체는 torchvision 없이 동일한 torch resize를 사용한다. `RBQ10_MUJOCO_PYTHON`으로 실행 Python을 지정할 수 있다.

남은 작업은 실제 RBQ10 Student 학습/전이 검증, joystick 회전 응답 확인, 모터 속도 제한과 PD/contact 물리 차이 보정이다. [실물 입력 결정 사항](DEPLOYMENT_GAPS.md).

---

# 이전 Teacher 진단 기록

아래는 Student 전환 전 수행한 기록이며 현재 기본 경로가 아니다. 필요하면 `./run.sh export-sim2sim --role teacher`와 `.venv-sim2sim/bin/python sim2sim/run_teacher.py`로 명시적으로 실행한다.

# MuJoCo Sim2Sim 준비 상태

2026-10-23. 현재 구현은 **RBQ10 Teacher의 실제 체크포인트를 MuJoCo에서 실행하는 초기 검증 도구**다. 안정적인 정책 전이, Vision Student 전이, 실물 배포 완료를 뜻하지 않는다. Isaac 학습 코드·환경은 변경하지 않는다.

## 구현

- `tools/export_sim2sim.py`: 원본 `Actor`, `StateHistoryEncoder`, `DefaultEstimator` 클래스를 직접 불러온다. checkpoint의 로봇 계약·관절 순서·URDF 해시를 검증하고 원래 Teacher play 경로처럼 estimator와 history inference를 묶어 CPU TorchScript로 내보낸다. PPO·reward·신경망을 새로 만들지 않는다.
- `sim2sim/build_scene.py`: 공식 URDF를 MuJoCo의 URDF importer로 읽어 질량·관성·관절 축·제한·충돌 형상을 가져오고 floating base와 PD actuator를 추가한다. 원본 자산은 수정하지 않는다. 평지, 10cm hurdle, 10cm step을 제공한다. 이 세 지형은 인터페이스 점검용이며 원본 curriculum 지형을 재현하지 않는다.
- `sim2sim/run_teacher.py`: 이름으로 MuJoCo 관절을 Isaac native 순서로 재배열한다. 753차원 관측값, 10프레임 history, 132개 terrain ray, 접촉 필터, 이전 action, body-frame angular velocity, roll/pitch를 생성한다. 방향 목표는 단순한 직선 waypoint다. 정책 50Hz, 물리 200Hz, 원본 Teacher처럼 action delay 기본 0이다. Student의 원본 action delay는 1 policy step이다.
- `sim2sim/check_bridge.py`: 실제 원본 관측값 함수에 tensor fixture를 넣어 배열·history 호환성을 검증하고, 10초 정지 PD와 관절/액추에이터 매핑을 점검한다.

MuJoCo는 [공식 URDF compiler](https://mujoco.readthedocs.io/en/3.3.0/XMLreference.html)와 [Python API](https://mujoco.readthedocs.io/en/3.3.0/python.html)를 사용한다. visual mesh도 유지하지만 GUI는 `--viewer`를 명시했을 때만 연다.

## 실행: 현재 PC

프로젝트 경로에서:

```bash
# 기존 Teacher checkpoint의 원본 actor + estimator export. Isaac Sim을 시작하지 않는다.
./run.sh export-sim2sim --checkpoint checkpoints/rbq10_teacher/model_3674.pt

# CPU headless. 진행 중인 Isaac 학습과 독립 실행.
./run.sh sim2sim --course flat --seconds 10
./run.sh sim2sim --course hurdle --height 0.10 --seconds 10
./run.sh sim2sim --course step --height 0.10 --seconds 10

# GUI가 필요할 때만:
./run.sh sim2sim --course flat --seconds 30 --viewer

.venv-sim2sim/bin/python sim2sim/check_bridge.py
```

`outputs/sim2sim/teacher/{teacher.pt,manifest.json}`가 생성된다. 외부 PC에는 두 파일과 프로젝트의 assets/sim2sim 코드를 전달하면 rollout에 Isaac Sim이 필요 없다. 학습 중인 checkpoint 대신 완전히 저장된 checkpoint를 지정한다.

## 다른 PC의 별도 실행 환경

기존 프로젝트 설치 문서로 원본 코드를 준비한 뒤, 별도 Python 3.11 환경에서:

```bash
python3.11 -m venv .venv-sim2sim
.venv-sim2sim/bin/pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
.venv-sim2sim/bin/pip install -r sim2sim/requirements.txt
# 이미 export한 teacher.pt와 manifest.json을 받았다면 아래 추가 설치/export는 불필요.
.venv-sim2sim/bin/pip install rsl-rl-lib==2.3.3
.venv-sim2sim/bin/python tools/export_sim2sim.py
./run.sh sim2sim --course flat --seconds 10
```

현재 PC의 `.venv-sim2sim`은 별도 3.11 venv이며 기존 Isaac Python site-packages를 읽는 `.pth`가 있다. MuJoCo 3.3.0만 별도로 설치했고 학습 venv는 수정하지 않았다. 이 `.pth` 절대 경로를 다른 PC에 복사하지 말고 위 독립 설치 방법을 사용한다. `RBQ10_MUJOCO_PYTHON`으로 실행 Python을 바꿀 수 있다.

## 실제 확인 결과

`model_3674.pt`, CPU, MuJoCo 3.3.0, nominal friction 0.9:

| 확인 | 결과 |
| --- | --- |
| 원본 actor+estimator 대 TorchScript: 16개 무작위 입력 | 최대 절대 오차 0 |
| 원본 관측값 함수 대 bridge: 7 step fixture | 최대 절대 오차 약 4.7e-8 |
| 물리 모델·관절/actuator 이름 매핑 | 통과, 총 질량 40.68392kg |
| home pose PD 10초 | 유지, 최종 몸체 높이 약 0.478m |
| 실제 Teacher 평지 정책 | 약 1.96초에 fall 판정 |
| 실제 Teacher hurdle 정책 | 약 2.72초에 fall 판정 |
| 실제 Teacher step 정책 | 약 3.36초에 fall 판정 |

위 정책 실행은 **전이 실패를 기록한 초기 단일 시행**이다. 성공률 통계가 아니다. 원본 Isaac에서 측정한 level9 58/60과 비교 가능한 동일 지형/초기 상태/종료 조건 평가도 아니다. 증거는 `outputs/sim2sim/{bridge_checks,smoke_flat,smoke_hurdle,smoke_step}.json`에 저장된다. 넘어졌다는 결과를 숨기거나 성공 예제로 대체하지 않는다.

## 남은 작업

1. **높이 스캐너 정정 완료**: 실제 Isaac 평가에서 29회 raw scan을 측정해 변화를 확인했다. cfg.offset은 ray 시작점에만 더해지며 sensor pos_w는 tracked trunk 위치다. 이전 “20m sensor origin으로 인한 전체 +1 포화” 분석은 tracking offset과 ray offset을 혼동한 오류였다. MuJoCo Teacher bridge의 ray 시작점과 높이 기준을 수정했다. 위 Teacher 초기 낙상 기록은 수정 전 잘못된 scan bridge로 얻은 과거 결과이므로 수정 후 전이 성능을 나타내지 않는다.
2. 동일한 저장 상태에서 Isaac/MuJoCo 관절 위치·속도, quaternion/frame, contact, scan, action을 순서대로 비교한다. 현재 fixture는 source 배열 호환성 검증이며 simulator trajectory parity는 아니다.
3. PhysX implicit PD와 MuJoCo position servo는 같은 숫자의 gain만으로 동일해지지 않는다. 현재 100/100/150Nm torque limit은 적용하지만 PhysX 50rad/s 속도 제한은 MuJoCo에서 재현하지 않고 기록한다. 초기 정책 실행에서 50rad/s 초과가 관측됐으며 모터 모델·속도 제약·접촉 solver 차이를 보정해야 한다. qvel을 강제로 잘라 성공을 만들지 않는다.
4. 원본 지형 mesh·waypoint·reset·종료 조건을 공유한 본 평가를 추가한다. 현재 직선 test course 결과로 Jump/Climb/Leap 성능을 인증하지 않는다.
5. 실센서 Student 계약 확정 후 Depth range/camera axis/intrinsics/preprocessing을 맞추고, GRU hidden state를 명시적 입력/출력으로 export한다. Student 없이 Teacher를 Vision Student로 가장하지 않는다.

[실물 적용 전 해결 사항](DEPLOYMENT_GAPS.md)

키보드 GUI 실행: `./run.sh sim2sim-latest --course isaac_flat --difficulty 0 --terrain-label flat`. [조작법과 실제 최신 snapshot 테스트](KEYBOARD_PLAYBACK.md).
