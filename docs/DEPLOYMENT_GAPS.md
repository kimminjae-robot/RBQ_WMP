# 사용자 지정 입력 계약 반영 (2026-10-23)

현재 MuJoCo 기본 경로는 **Student**다. 아래 기존 검토 내용 중 다음 항목은 사용자 제공 방식으로 결정했다.

- 선속도: reference와 measured를 모두 받을 수 있으므로 별도 추정기는 필수가 아니다. reference는 명령 슬롯, measured body xyz는 속도 슬롯으로 분리했다. MuJoCo에서는 simulator measured velocity, 실제 로봇에서는 사용자가 공급하는 측정값을 받는다.
- 지형 종류: 실제 실험 환경을 학습 환경에 맞춰 구성하고 운영자가 flat/parkour label을 수동 지정한다. 추론기를 새로 만들지 않는다. 서로 다른 지형을 한 번에 이동할 때 label을 고정할지 전환할지는 후속 실험에서 정한다.
- 방향: joystick의 vx/yaw_rate를 사용한다. 원본에 직접 yaw-rate 입력이 없어 설정 가능한 horizon으로 방향 오차 2개로 변환한다. 원본 waypoint 정답을 MuJoCo Student에 공급하지 않는다. 이 변환의 실제 회전 응답은 학습된 Student로 확인해야 한다.
- 발 접촉: MuJoCo contact 정보를 사용한다. 실제 로봇에서는 추후 contact 추정기를 연결한다.
- 카메라: 제조사 Coordinate Frames의 전방 OAK-D Pro Wide Depth TF를 Isaac Student와 MuJoCo에 적용했다. 위치 `(0.388031,-0.037500,0.037764)m`, ROS quaternion `(0.5,-0.5,0.5,-0.5)`이며 광축은 정면이다. 원본 intrinsic/FOV 설정은 유지한다.
- 시간/메모리: 이번 변경에서 보류. 원본 주기, delay, 내부 GRU 상태를 유지한다.

이 변경은 MuJoCo 입력/배포 코드에 적용했다. 실행 중인 Teacher 학습이나 아직 시작하지 않은 Student 학습 경로를 수정한 것은 아니다. Phase 4의 학습/검증도 측정 선속도, label 운용, joystick 방향 계약과 일관되게 구성해야 한다.

[현재 실행 방법](SIM2SIM.md)

---

## 이전 코드 검토 기록

다음은 수정 전 원본 경로의 분석이다. 위에서 결정한 측정 선속도/수동 label/접촉 공급 방식을 부정하거나 별도 추정기를 요구하는 의미가 아니다.

# 실제 센서로 실행하기 위해 해결할 차이

2026-10-23 코드 검토 기준. 현재 RBQ10 학습 모델은 Teacher이며 Student는 아직 학습하지 않았다. 아래 문제는 현재 진행 중인 학습의 설정을 즉시 바꾸지 않고 별도 실험·버전 계약으로 해결해야 한다. 학습 입력이 바뀌면 기존 checkpoint가 같은 성능을 유지한다고 가정할 수 없다.

## 1. Student가 아직 정답 선속도를 사용한다

원본 753차원 observation의 `[185:194]`는 몸체 좌표 선속도 3개 ×2.0과 zero padding 6개다. Teacher PPO의 action 경로와 Teacher play는 estimator(53→128→64→9)로 이 영역을 대체한다. 그러나 Student `learn_vision`은 이 대체를 수행하지 않는다. Student play에서도 대체 줄이 주석 처리돼 있다. 따라서 Student action과 distillation Teacher target 모두 이 경로에서 시뮬레이션 정답 선속도를 볼 수 있다. 네트워크에 estimator 객체가 존재한다는 사실만으로 실센서 실행이 보장되지 않는다.

실물 IMU는 몸체의 정확한 선속도를 직접 주지 않는다. 원본 estimator를 Student 학습·평가·export에 일관되게 연결하거나, 실제 로봇 state estimator를 동일 좌표계·단위·지연으로 공급해야 한다. 추정 오차와 지연에 대한 학습/검증이 필요하다. 이번 MuJoCo Teacher export는 기존 Teacher play와 동일하게 estimator를 포함한다.

## 2. 53차원 상태가 전부 직접 측정 가능한 값은 아니다

| index | 값 | 실물에서 필요한 것 |
| --- | --- | --- |
| 0:3 | body angular velocity ×0.25 | IMU 축 정렬·bias·필터·타임스탬프 |
| 3:5 | roll/pitch | quaternion convention과 world/body frame을 맞춘 자세 추정 |
| 5:8 | zero 1개 + 현재/다음 waypoint 방향 오차 2개 | Student 시각 방향 예측 또는 실제 navigation 명령 |
| 8:11 | zero 2개 + 전진속도 명령 1개 | 사용자/상위 navigation의 명령. 측정 속도와 구분 |
| 11:13 | non-flat/flat 지형 정답 플래그 | 센서 추론 또는 플래그를 제거/고정한 재학습 |
| 13:25 | joint position − default pose | native 관절 순서·영점·부호 |
| 25:37 | joint velocity ×0.05 | encoder 차분/필터의 실제 지연 |
| 37:49 | 이전 raw action | 적용 PD target과 구분하여 동일 의미 유지 |
| 49:53 | 발 접촉 여부 −0.5 | 실제 접촉 센서 또는 motor torque 기반 접촉 추정 |

특히 **지형 플래그 2개는 시뮬레이터가 알려주는 정답**이다. Student가 depth를 사용해도 이 정답을 계속 받으면 실물 입력 조건과 다르다. 단순히 실행 시 0으로 바꾸면 학습 분포가 바뀐다. 동일 차원을 유지하며 플래그를 없애는 학습/미세조정과 성능 비교가 필요하다.

발 접촉은 시뮬레이터 net force norm>2N을 현재/이전 sample에서 OR한 값이다. 실물의 모터 전류·토크·발 센서는 이 값과 같지 않다. 어떤 실제 센서를 쓸지 정하고 threshold, hysteresis, false contact, 충돌·미끄러짐 상황을 검증해야 한다.

## 3. Student 방향 입력에 정답이 남는 조건

Depth encoder는 32차원 지형 특징과 2개 방향값을 예측한다. 원본 학습은 encoder 입력의 방향 두 값을 0으로 만들지만, Student actor의 방향 입력은 `delta_yaw_ok`일 때만 예측값으로 바꾼다. 나머지 경우 원본 waypoint 정답이 남는다. 이 mask 자체도 정답 방향으로 계산하며 조건은 현재 `delta_yaw < 0.6`이다(abs 조건이 아님). 실행에서는 예측값으로 항상 교체하므로 학습/실행 조건도 다르다.

Teacher target과 loss에 정답을 쓰는 것은 정상적인 지도학습이다. 해결 대상은 **Student가 행동할 때 정답 입력이나 정답 선택 조건에 의존하는 경로**다. 방향값을 항상 예측 또는 실제 상위 명령으로 공급하도록 통일하고, 큰 방향 전환·음/양 방향 오차를 모두 평가해야 한다.

## 4. Depth는 실제 카메라와 동일하지 않다

현재 depth는 실제 D435 측정이 아닌 지형 mesh만 보는 RayCaster다. 몸체 mount `(0.42,0,0.10)m`도 초기 시뮬레이션 값이며 실측 보정값이 아니다. 원본 입력은 60×106 ray distance를 아래 2행/양옆 4열 crop하고 58×87로 resize한 뒤 `distance/2.0−0.5`로 정규화한다. ray 길이(distance_to_camera)와 일반 depth 카메라의 optical-axis Z depth는 다르므로 단위를 m로 맞추는 것만으로 부족하다.

실물에서 확인할 것은 intrinsics, lens distortion, camera→trunk transform, optical-axis convention, 최대 2m clipping, invalid/0/NaN 처리, resize interpolation이다. 현재 RayCaster는 지형 외 동적 물체나 로봇 자체를 보지 않아 실제 depth와 가림 조건도 다르다. 원본 pitch randomization ±5도만으로 이 차이를 모두 처리하지 못한다. 거리별 noise·hole·edge artifact·exposure 영향·mount 흔들림을 추가하고 실측 데이터와 비교해야 한다.

## 5. 시간 축과 recurrent 상태

정책은 50Hz(20ms), physics는 200Hz(5ms)다. Depth 전처리/encoder는 5 policy step마다 갱신해 10Hz다. 2-frame buffer의 최신보다 한 프레임 이전 영상을 반환하므로, 첫 buffer 초기화 이후 약 100ms의 영상 지연에 특징 hold 시간이 더해진다. 원본 Student action delay 1 step은 추가 20ms이고 Teacher는 0 step이다. 실물 카메라가 30Hz라면 시간 기준 sampling/hold를 명시해야 한다.

지금은 고정 지연이 주로 구현돼 있고 실제 네트워크 jitter·프레임 유실·센서간 timestamp 차이까지 검증된 것은 아니다. 측정한 latency 분포로 학습하고 오래된 프레임의 처리와 timeout 동작을 정의해야 한다.

GRU hidden state 512개는 실행 간 메모리다. 원본 depth encoder는 hidden state를 내부 변수로 보관하고 runner는 detach만 한다. reset된 환경별 hidden state 처리를 별도로 검토해야 한다. 배포 export는 `(depth, proprioception, h_in) → (features, yaw, h_out)`로 상태를 명시해 재시작/episode/시뮬레이터 전환에서 재현 가능하게 만드는 편이 적합하다. 현재 준비한 export는 **Teacher만** 지원한다.

## 6. 액추에이터와 모터 모델

현재 목표는 `q_target = q_home + 0.25 × raw_action`이고 12개 출력은 토크 명령이 아니다. Isaac의 implicit PD gain은 R/P 123.39138, K 127.77215; damping 2.5, torque limit R/P 100Nm·K 150Nm, velocity limit 50rad/s다. 제조사 평지 ONNX policy는 45차원/100Hz/action scale 0.1로 별도 계약이므로 혼용할 수 없다.

MuJoCo position servo의 implicit 적분·접촉 해법은 PhysX와 다르고 실물은 모터 제어 주기, 전류 제한, torque-speed curve, 감속기 마찰·백래시, 배터리 전압, 온도 영향까지 존재한다. PD gain 숫자만 복사하면 동일 동작이라는 보장이 없다. 현재 bridge는 torque limit은 적용하고 velocity cap은 기록만 한다. 정책 실행 중 50rad/s 초과가 나타나므로 해당 차이는 실제로 조정할 항목이다. 별도 정지·관절 step response·보행 테스트로 파라미터를 맞추고 모터 모델/물리 오차 randomization을 검증해야 한다.

관절 순서는 Isaac native(R 네 다리→P 네 다리→K 네 다리)이며 제조사 배열은 다리별 R/P/K다. 반드시 이름 기반으로 매핑하고 영점·부호·발 HR/HL/FR/FL 순서를 확인해야 한다.

## 7. 높이 입력 포화 우려: 실제 관측으로 정정

2026-10-23 `model_4832.pt`의 Isaac 고정 level9 평가에서 raw observation을 29번 측정했다. scan 값은 −1..1에서 변화했고, 평균 upper clip 비율은 약 5.13%였다. zero-scan으로 바꿨을 때 행동도 달라졌다. 따라서 이전 fixture에 근거한 “전체 +1 포화 가능성”은 현재 실행에서 확인되지 않았다.

앞선 분석은 `RayCaster._offset`을 `cfg.offset`과 혼동한 오류였다. 실제 구현은 `cfg.offset`을 `_initialize_rays_impl`에서 ray 시작점에만 더한다. `data.pos_w`는 별도 tracking transform으로 얻은 trunk 위치다. 평지에서 body z=0.55이면 원본 높이식은 0.25이며 20.25가 아니다. MuJoCo Teacher 진단 bridge도 이 의미로 수정했다. Student는 Teacher scan을 직접 입력받지 않는다.

이전 JSON fixture는 가상의 sensor z=20.55를 가정한 계산일 뿐 실제 학습 센서 위치를 검증한 것이 아니었다. 근거는 `evaluations/teacher_to_student/teacher_4832_level9.json`의 `height_scan_audit`와 `decision.json`이다. 행동 차이는 지형 입력에 대한 의존성 증거이며, zero-scan으로 실제 코스를 재평가한 성능 ablation은 아니다.

## 권장 진행 순서

1. Teacher는 checkpoint를 보존하고 Student로 전환했다. 높이 관측 실측은 완료했으며, 방향/속도 경로는 사용자 지정 계약에 맞춰 검증한다.
2. 같은 저장 상태에서 두 엔진의 관측값/행동 비교 → PD/contact/velocity limit 보정 → 평지/낮은 step의 실제 전이 확보.
3. 실센서 Student 계약을 확정하고 정답 입력 누출을 제거한 원본 distillation 학습.
4. 동일 전처리·타임스탬프·명시적 GRU 상태로 export, depth 포함 MuJoCo 평가.
5. 여러 seed/물리 파라미터/센서 지연에서 성공률·실패 원인 측정 후 실물 센서 재생과 제한된 로봇 실험으로 진행.

실제 코드 위치: `upstream/Isaaclab_Parkour/parkour_isaaclab/envs/mdp/observations.py`, `scripts/rsl_rl/modules/on_policy_runner_with_extractor.py`, `scripts/rsl_rl/play.py`, `scripts/rsl_rl/modules/feature_extractors/depth_backbone.py`, `rbq10_parkour/contract.py`, `rbq10_parkour/robot.py`.
