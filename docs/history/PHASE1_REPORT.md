> 과거 PC별 검증 기록입니다. 현재 상태는 [../STATUS.md](../STATUS.md)를 참고하세요. 과거 대용량 영상·로그는 이번 배포에 포함하지 않습니다.

# Phase 1 실행 보고 — 2026-10-22 (KST)

## 1. 발견한 환경

프로젝트 대상 폴더와 부모 `~/Desktop/coop_ws`는 처음에 없었고 새로 생성했다. 기존 IsaacLab 소스는 `/home/user/isaac/IsaacLab`, Git tag `v2.3.2`이며 기존 작업의 untracked `demo/`, `workspaces/`를 유지했다.

| 환경 | Python | Isaac Sim | PyTorch | RSL-RL | 처리 |
| --- | --- | --- | --- | --- | --- |
| `/home/user/isaac/env_isaaclab` | 3.11.17 | 5.1.0.0 | 2.7.0+cu128 | 3.1.2 | 런타임 패키지를 읽어 재사용 |
| `/home/user/miniforge3/envs/env_isaaclab` | 3.11.15 | 5.1.0.0 | 2.13.0 | 5.0.1 | 변경하지 않음 |
| 프로젝트 `.venv` | 3.11.17 | 위 런타임 재사용 | 2.7.0+cu128 | **2.3.3** | 별도 설치·실행 |

Isaac Sim 5.0 standalone도 Downloads에서 발견했지만 사용하지 않았다. 실제 검증 GPU는 RTX 5080 Laptop 16 GB, 드라이버 580.65.06, PyTorch CUDA 12.8, compute capability 12.0이다. Ubuntu 24.04.5, kernel 6.8.0-146-generic이며 GUI는 X11 `DISPLAY=:0`을 사용했다. 처음 샌드박스 내부 `nvidia-smi`는 GPU 장치를 볼 수 없었고, 호스트 실행에서는 정상 확인됐다.

원본 작성자가 확인한 조합은 [Isaac Sim 4.5.0](https://github.com/CAI23sbP/Isaaclab_Parkour/issues/19)과 [Isaac Lab tag 2.1.0](https://github.com/CAI23sbP/Isaaclab_Parkour/issues/22)이다. 이번에는 기존 설치를 다운그레이드하지 않고 **Lab 2.3.2 + Sim 5.1 + RSL-RL 2.3.3**으로 실제 실행을 검증했다. 원본과 동일한 물리 엔진 버전으로 성능을 수치 재현한 것은 아니다.

## 2. 원본의 실제 구성

기준: [CAI23sbP/Isaaclab_Parkour](https://github.com/CAI23sbP/Isaaclab_Parkour), commit `d6766d8883cffe427f2d4c087717d4eca1f23d8c`.

| 기능 | 원본 파일/클래스 |
| --- | --- |
| 학습 진입점 | `scripts/rsl_rl/train.py` |
| GUI 재생·export | `scripts/rsl_rl/play.py` |
| 평가·별도 데모 | `scripts/rsl_rl/evaluation.py`, `demo.py` |
| 공통 runner | `scripts/rsl_rl/modules/on_policy_runner_with_extractor.py` / `OnPolicyRunnerWithExtractor` |
| Teacher | `PPOWithExtractor`, `ActorCriticRMA`, state estimator, history encoder |
| Student | `DistillationWithExtractor`, depth CNN + GRU, depth actor; Teacher action과 yaw를 증류 |
| Go2 설정 | `parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/` |
| 지형 | `parkour_isaaclab/terrains/extreme_parkour/`, `ParkourTerrainGenerator` |
| Reward·종료·randomization | Go2의 `parkour_mdp_cfg.py`, `parkour_isaaclab/envs/mdp/` |

Teacher/Student 각각 Train·Play·Eval의 총 6개 Gym ID를 실제 등록하고 configuration 로딩을 확인했다. 접미사 없이 `...Teacher-Unitree-Go2-v0`가 학습 환경, `...-Play-v0`가 재생 환경이다. 물리 간격은 0.005초, 제어는 decimation 4로 0.02초/50 Hz이다. action은 12개 관절, policy observation은 753차원(53 proprioception + 132 scan + 9 explicit privileged + 29 privileged latent + 530 history)이다. Depth 원본 센서는 60×106이며 crop·resize 후 58×87, 2-frame buffer의 이전 영상과 5 제어 스텝마다 갱신하는 depth encoder를 사용한다. 렌더링 RGB 카메라 대신 원본의 RayCasterCamera Depth를 유지했다.

## 3. 설치와 실제 실행

전용 환경에 `rsl-rl-lib==2.3.3`, `fast-simplification==0.2.0`, `gdown` 및 다운로드용 의존성을 설치하고 원본 두 패키지를 editable 등록했다. 기존 IsaacLab 소스와 기존 환경의 패키지는 수정하지 않았다. Kit는 실행 과정에서 자체 캐시·로그·사용자 설정 파일을 사용할 수 있다.

원본 README의 Teacher·Student Drive 페이지, `uc`와 `drive.usercontent` 다운로드 경로가 HTTP 404를 반환했다. GitHub release에는 모델이 없었다. 공개 포크를 조사해 다음 파일을 다운로드했다.

| 용도 | 공개 포크 경로 | iteration | 크기 |
| --- | --- | --- | --- |
| Teacher | `muzi2018/Isaaclab_Parkour`: `logs/rsl_rl/unitree_go2_parkour_student_ppo/teacher_policy/model_34700.pt` | 34700 | 9,789,238 B |
| Student | 같은 포크: `logs/rsl_rl/unitree_go2_parkour_student_ppo/2025-09-03_12-07-56/model_99998.pt` | 99998 | 46,374,344 B |

포크 commit은 `388f3df5bc01d7d8f036b25c25490252205e7dcb`로 고정했으며 Git blob 해시와 다운로드 바이트를 검증했다. 모델/estimator/depth encoder/depth actor가 원본 runner에 정상 로드됐다. 출처와 SHA-256은 `checkpoints/manifest.json`에 기록했다. 접근 불가능한 원본 Drive 배포본과 동일한 파일이라고 단정하지 않는다. [원본 작성자는 Isaac Gym 체크포인트의 직접 호환을 부정](https://github.com/CAI23sbP/Isaaclab_Parkour/issues/20)했으며, 이번에는 Isaac Lab용 체크포인트를 사용했다.

Teacher 학습 검증은 `./run.sh train-teacher --num_envs 16 --max_iterations 1`로 실제 PPO 업데이트 1회(384 transitions)를 수행했고 종료 코드 0이었다. value loss 0.0040, surrogate loss -0.0490, estimator loss 0.1743이 기록됐으며 TensorBoard, configuration YAML/pickle, `model_0.pt`가 생성됐다. 이 모델은 파쿠르 학습 완료 모델이 아니다. 질문한 명령은 이와 같은 Teacher 학습 경로이며, 새로 학습할 경우 Teacher 완료 후 Student distillation이 이어진다.

## 4. 오류와 최소 수정

| 증상 | 수정과 이유 |
| --- | --- |
| 설치된 RSL-RL 3.x/5.x의 storage·관측 API가 원본과 다름 | 전용 환경에 2.3.3 사용. 원본 PPO·distillation 재작성 없음. [관련 원본 이슈](https://github.com/CAI23sbP/Isaaclab_Parkour/issues/10) |
| `isaaclab.utils.pretrained_checkpoint` 없음 | play/evaluation/demo의 import를 이동된 `isaaclab_rl.utils.pretrained_checkpoint`로 변경 |
| `delta_yaw_ok`가 `(N,)`를 반환해 ObservationManager `IndexError` | `(N,1)`로 반환하고 distillation Boolean indexing에서 마지막 축 squeeze. [원본 이슈 #4의 해결과 동일](https://github.com/CAI23sbP/Isaaclab_Parkour/issues/4) |
| `dump_pickle` 제거됨 | train.py에서 표준 pickle 저장으로 기존 artifact 형식 보존 |
| 중첩 `parkour_tasks` 및 namespace package 경로 문제 | 런처에서 원본 루트와 내부 task 루트를 프로세스별 PYTHONPATH로 지정 |
| GUI 종료 시 키보드 unsubscribe AttributeError | 기존 IsaacLab 예제와 같은 `unsubscribe_to_keyboard_events` API 사용 |
| 오류 후 GUI 종료 루프가 traceback을 숨김 | 재생 스크립트의 예외를 먼저 출력하고 즉시 종료. 진단 도구는 SimulationContext 정리 |
| 1회 X11 `BadRRCrtc` | 1280×720 창 크기 지정 후 정상 시작 |

진단 중 일부 중간 GUI 프로세스가 멈추거나 예외 종료했으며 해당 로그를 보존했다. 원래 오류를 감추거나 성공으로 처리하지 않았다. 추가 녹화를 학습 검증과 동시에 실행한 때에도 GUI 정지가 한 번 발생해 종료됐고, 최종 검증은 별도 Kit 프로세스 없이 단독으로 실행했다. 동시 실행이 정지의 원인인지는 확정하지 않았다.

Reward, terrain 생성식, curriculum, actor/critic 구조, Teacher PPO 및 Student 손실은 보존했다. `play.py`에는 선택적 종료·카메라 추적·실제 telemetry·녹화 이름 옵션만 추가했다. 전체 diff는 `patches/phase1_compat.patch`에 기록한다.

## 5. GUI 검증 결과와 한계

첫 Student GUI 실행은 16개 환경에서 1,500 제어 스텝(30초 simulation), Teacher GUI는 400스텝(8초 simulation)을 완료했다. `logs/student_gui.log`, `logs/student_gui.jsonl`, `logs/teacher_gui.log`, `logs/teacher_gui.jsonl`에 실제 측정값이 있다. Student의 hurdle·step·gap·경사 발판·혼합 코스 진행을 확인했다. Hurdle 환경 4/5는 waypoint index 7까지 진행했으며 환경 6은 index 3에서 진전이 없었다. Step 환경 7/8/9도 index 7까지 진행했고 base의 세계 좌표 z 최대값은 각각 약 1.634/1.441/1.648 m였다. 이 값은 장애물 자체의 높이가 아니다.

Telemetry는 10 제어 스텝마다 저장한다. 네 발 접촉력 norm이 모두 2 N 미만인 표본과 이후 착지 접촉을 확인했지만, 모든 비접촉 표본이 성공적인 장애물 점프인 것은 아니다. Reset과 실패도 실제 관측됐다. 이 짧은 재생으로 공식 성공률이나 논문 수준 성능을 주장하지 않는다. 영상 검증·최종 실행 결과는 아래 추가 기록을 따른다.

원본 `play.py`의 Student 분기는 depth latent와 history encoder를 사용하지만 explicit privileged observation의 base linear velocity를 estimator로 대체하는 줄이 주석 처리돼 있다. 따라서 이번 성공은 **원본 Go2 시뮬레이션 정책 재생**이며, 실제 로봇 센서만 사용하는 Student를 검증한 것은 아니다. 이식 뒤 Phase 4에서 이 센서 계약과 terrain-type 표식을 별도로 감사해야 한다.

원본 exporter도 실제 실행돼 `checkpoints/exported_deploy/{policy,depth_latest}.{pt,onnx}`가 생성됐고 ONNX checker는 통과했다. 그러나 원본 depth ONNX 입력 선언은 `[1,87,58]`로 시뮬레이션의 `[N,58,87]`와 축 순서가 다르고 GRU hidden state가 명시적 입출력으로 분리돼 있지 않다. 현재 export는 원본 산출물 보존이며 MuJoCo/실물 배포 검증 완료를 뜻하지 않는다. Phase 5에서 수정·수치 검증할 항목이다.

## 6. 승인 후 RBQ10 이식에서 다룰 파일

RBQ10 관련 변경은 아직 적용하지 않았다. 다음 단계에서는 Go2 설정을 유지한 채 별도 `config/rbq10/`와 asset 설정을 추가해야 한다.

- `parkour_tasks/parkour_tasks/default_cfg.py`: 현재 Go2 USD·actuator·camera 기본값을 참고해 RBQ10 전용 scene/asset 설정 분리.
- `.../config/go2/parkour_teacher_cfg.py`, `parkour_student_cfg.py`: 별도 RBQ10 Teacher/Student scene 구성의 기준. contact body 이름, base prim, scanner·camera 장착 위치 비교.
- `.../config/go2/parkour_mdp_cfg.py`: joint/body expression, initial pose, action scale, 실제 센서 관측을 RBQ10 별도 설정으로 연결.
- `.../config/go2/__init__.py`: Go2 등록을 유지하고 별도 RBQ10 Gym ID 등록 파일 추가.
- `.../config/go2/agents/parkour_rl_cfg.py`: 차원과 encoder 입력 계약 비교. 차원이 바뀌지 않으면 원본 구조 유지.
- `parkour_isaaclab/actuators/parkour_actuator_cfg.py`, `parkour_actuator_pd.py`: RBQ10 공식 actuator 토크·속도·PD 설정과 비교하며 기존 Go2 actuator를 변경하지 않음.
- `parkour_isaaclab/envs/mdp/observations.py`, `scripts/rsl_rl/exporter.py`: 추후 관측 순서·실센서 입력·export 계약 검증 대상으로만 검토.

원본 terrain·reward·curriculum·학습 runner는 공통으로 재사용하는 것이 기본이다. Phase 2 진행은 사용자 승인 후에만 시작한다.

## 최종 확인 기록

단독 Student GUI 실행이 **1,250스텝(25초 simulation), 종료 코드 0**으로 완료됐다. `logs/student_final.log`에 완료 메시지가 있으며 최종 실행에서는 traceback 및 키보드 해제 오류가 없었다. `logs/student_final.jsonl`에 원본 정책의 실제 위치·접촉·waypoint 측정을 125회 저장했다. 이 최종 영상의 최초 650스텝은 hurdle 환경 4, 이후는 step 환경 8을 추적한다.

- Student 실제 GUI 녹화 (원PC 기록: `artifacts/student_jump_climb.mp4`): 1280×720, 50 fps, 1,249 video frames. 앞부분에 hurdle 통과·정체 사례, 13초 이후 높은 step 위 이동과 내려오기·다시 올라가기 사례가 포함된다.
- Hurdle 환경 5는 simulation 4.4초에 네 발 접촉력 모두 0 N, base z 0.544 m였고 4.6초에 앞발 접촉력이 약 77/111 N으로 복귀했다. 실제 hurdle 점프/착지를 확인했으며, 영상에서도 장애물 통과 뒤 착지 장면을 확인했다.
- Step 환경 7/8/9는 waypoint index 7까지 진행했다. 환경 8의 base z 최대값은 1.423 m였고 높은 플랫폼 위에 올라간 모습을 영상으로 확인했다. 영상 프레임 (원PC 기록: `artifacts/student_climb_frame.png`).
- Hurdle 환경 4는 최종 실행에서 index 2까지, 환경 5는 index 5까지 진행했고 환경 6은 index 7까지 진행했다. 원본의 높은 난이도 `(0.7, 1.0)` 설정에서 실패·정체 사례도 남아 있다.
- Teacher GUI 녹화 (원PC 기록: `artifacts/teacher_step.mp4`), 최종 Student 측정 요약 (원PC 기록: `logs/student_final_summary.json`), 실제 환경 기록 (원PC 기록: `logs/environment.json`), ONNX 구조 검사 (원PC 기록: `logs/export_validation.json`).

**Phase 1의 원본 Go2 Teacher/Student 코드 실행·사전학습 정책 GUI 재생과 점프·Climb 동작 확인을 마쳤다. 공식 Drive 원본과의 파일 동일성, 동일 버전 시뮬레이터에서의 논문 성능 재현, 센서 전용 실물 Student 검증은 이번 결과에 포함하지 않는다. 여기서 중단하며 Phase 2는 사용자 승인 후 진행한다.**
