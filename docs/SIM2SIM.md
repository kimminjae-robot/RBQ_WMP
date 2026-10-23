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

1. **높이 스캐너 의미 확인**: 원본은 mount z=20m, 현재 Isaac RayCaster의 sensor pos_w는 mount를 포함하고 `_get_heights`는 여기에서 hit height와 0.3만 뺀다. 평지 fixture는 132개 모두 +1로 clip된다. 현재 checkpoint 실행은 이 기존 의미를 `--scan-mode source`로 유지한다. `--scan-mode base-height`는 비교 실험용이며 기존 정책의 입력 분포를 바꾼다. 학습 중 실제 tensor의 min/max/std/clip 비율을 지형별로 수집하고, scan 제거/변경 ablation으로 정책의 지형 의존성을 확인해야 한다. fixture만으로 전체 학습 기간의 관측값이 항상 동일했다고 결론 내리지 않는다.
2. 동일한 저장 상태에서 Isaac/MuJoCo 관절 위치·속도, quaternion/frame, contact, scan, action을 순서대로 비교한다. 현재 fixture는 source 배열 호환성 검증이며 simulator trajectory parity는 아니다.
3. PhysX implicit PD와 MuJoCo position servo는 같은 숫자의 gain만으로 동일해지지 않는다. 현재 100/100/150Nm torque limit은 적용하지만 PhysX 50rad/s 속도 제한은 MuJoCo에서 재현하지 않고 기록한다. 초기 정책 실행에서 50rad/s 초과가 관측됐으며 모터 모델·속도 제약·접촉 solver 차이를 보정해야 한다. qvel을 강제로 잘라 성공을 만들지 않는다.
4. 원본 지형 mesh·waypoint·reset·종료 조건을 공유한 본 평가를 추가한다. 현재 직선 test course 결과로 Jump/Climb/Leap 성능을 인증하지 않는다.
5. 실센서 Student 계약 확정 후 Depth range/camera axis/intrinsics/preprocessing을 맞추고, GRU hidden state를 명시적 입력/출력으로 export한다. Student 없이 Teacher를 Vision Student로 가장하지 않는다.

[실물 적용 전 해결 사항](DEPLOYMENT_GAPS.md)
