> 과거 PC별 기록입니다. 현재 절차는 저장소 README와 docs/STATUS.md를 기준으로 합니다.

# RBQ10 Vision Parkour — Phase 3 학습 진행 중

작업 경로: `/home/user/Desktop/coop_ws/rbq10_vision_parkour`

원본 Go2 재생(Phase 1)과 RBQ10 모델 이식·기본 물리 검증(Phase 2)을 완료했다. **Phase 3 RBQ10 Teacher 학습을 2026-10-22 KST에 시작했다.** 원본 PPO·reward·curriculum으로 seed 1, 6,144환경, 50,000 iteration을 실행 중이다. 학습 완료나 파쿠르 성공을 의미하지 않는다. Phase 3 진행 기록 (원PC 기록: `PHASE3_STATUS.md`), Phase 1 보고서 (원PC 기록: `PHASE1_REPORT.md`), Phase 2 보고서 (원PC 기록: `PHASE2_REPORT.md`).

중간 검증은 headless로 진행하고 각 Phase 마지막에만 GUI를 보여준다.

## 실행 중인 Teacher 확인

```bash
cd ~/Desktop/coop_ws/rbq10_vision_parkour
./run.sh teacher-job status
```

실제 진행은 `logs/phase3/active_job.json`이 가리키는 job 파일과 해당 `log_dir`에서 확인한다. `latest.json`/`progress.jsonl`에 PPO loss·reward·episode length·지형별 완주율/낙상률이 있으며, TensorBoard event와 `model_*.pt`도 같은 학습 폴더에 저장된다. 완주율은 각 지형의 최근 최대 1,000개 완전한 episode 기준이며 `completed_episodes`가 0이면 아직 측정값이 없다.

```bash
# 필요할 때만: 현재 PPO 업데이트를 마치고 checkpoint 저장 후 중단
./run.sh teacher-job stop

# 중단한 뒤에만: 저장된 Phase 3 checkpoint에서 추가 iteration 학습
./run.sh teacher-job start --resume <이_프로젝트의_model_N.pt_경로> --iterations <추가_iteration_수>
```

기본 새 학습은 `./run.sh teacher-job start`다. 현재 실행 중인 작업이 있으면 중복 실행을 거부한다. 재개 시 PPO뿐 아니라 estimator/history optimizer와 원본 RMA schedule counter도 복원한다. Simulator episode는 새로 시작하므로 trajectory를 정확히 이어가는 기능은 아니다. 시작 규모 검증과 중단·재개 시험은 모두 실제 PPO 업데이트와 checkpoint로 확인했다. 초기 속도 기준 수십 시간이 필요한 장기 학습이다.

## RBQ10 이식 검증

공식 RBQ10 모델로 home 자세, 공식 보행 정책의 평지 보행과 낮은 step 통과를 각각 4개 환경에서 확인했다. 최종 GUI 영상 (원PC 기록: `artifacts/rbq10_step_gui.mp4`), 실제 step 측정값 (원PC 기록: `logs/phase2/step_gui_v2.json`).

```bash
cd ~/Desktop/coop_ws/rbq10_vision_parkour
./run.sh validate-rbq10 --mode probe --role Student --num_envs 4 \
  --headless --output logs/phase2/manual_probe.json
./run.sh validate-rbq10 --mode step --num_envs 4 --seconds 12 \
  --headless --output logs/phase2/manual_step.json
```

GUI 확인은 마지막 명령의 `--headless`를 빼면 된다. 검증용 공식 보행 정책은 100 Hz이며 depth를 입력하지 않는다. 새 파쿠르 환경은 원본의 50 Hz·PPO·Reward·Curriculum·Teacher/Student 구조를 유지한다. 50 Hz 정지 자세도 검증했지만 학습을 완료한 RBQ10 파쿠르 정책은 아직 없다.

RBQ10 학습 진입점은 기존 Go2 정책 학습 명령과 분리돼 있다. 백그라운드 작업 관리는 위 `teacher-job` 명령을 사용한다. 아래는 직접 실행할 때의 진입점이다.

```bash
# Phase 3: 직접 학습. 실행 중인 background 작업과 동시에 시작하지 말 것.
./run.sh train-rbq10-teacher --num_envs 6144 --seed 1
# Phase 4: 학습을 마친 RBQ10 Teacher로 Student 증류
./run.sh train-rbq10-student --load_run <RBQ10_Teacher_run> --checkpoint <model_name>
```

Go2 checkpoint를 RBQ10에 사용하는 것은 모델 계약 검사에서 거부한다. RBQ10 Teacher checkpoint는 학습 진행에 따라 생성되며 초기 파일은 파쿠르 성능을 검증한 모델이 아니다. 보행 ONNX는 Phase 2 모델·기본 물리 검증용이다. Vision Student 학습은 아직 시작하지 않았다.

## Go2 GUI 재생

```bash
cd ~/Desktop/coop_ws/rbq10_vision_parkour
./run.sh play-student
```

Teacher 재생:

```bash
./run.sh play-teacher
```

현재 환경에서 창 크기 지정이 필요하면 다음 옵션을 붙인다.

```bash
./run.sh play-student --kit_args '--/app/window/width=1280 --/app/window/height=720'
```

점프·Climb 영상을 다시 기록하는 명령:

```bash
./run.sh play-student --max_steps 1250 \
  --viewer_env 4 --viewer_switch_step 650 --viewer_second_env 8 \
  --video --video_length 1250 --video_prefix student_jump_climb \
  --telemetry "$PWD/logs/student_replay.jsonl" \
  --kit_args '--/app/window/width=1280 --/app/window/height=720'
```

이 명령은 원본의 terrain·reward·curriculum·actor·depth encoder·추론 루프를 사용한다. 추가 옵션은 종료 시점, GUI 추적 대상, 녹화 이름, 실제 측정값 저장만 제어한다. `run.sh`는 원본 Student 설정의 별도 OpenCV Depth 표시 창을 끄며, Depth 센서와 정책 입력은 계속 사용한다.

## Teacher 학습 명령의 의미

질문한 아래 명령은 이미 학습한 모델의 재생이 아니라 원본 Go2 Teacher를 처음부터 PPO로 학습하는 명령이다.

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0 --seed 1 --headless
```

이 프로젝트에서는 전용 환경과 패키지 경로를 설정하는 런처로 실행한다.

```bash
# 실행 검증에 사용한 작은 학습: 16개 환경, PPO 업데이트 1회
./run.sh train-teacher --num_envs 16 --max_iterations 1

# 원본 Teacher 학습 시작 — 장시간 작업이며 자동 실행하지 않음
./run.sh train-teacher
```

원본 Teacher 기본값은 6,144개 환경·최대 50,000 iteration이다. 노트북에서 이 기본 규모의 학습은 검증하지 않았다. 실제 장기 학습은 가용 GPU 메모리에 맞춘 `--num_envs` 조정이 필요하다. 1회 업데이트 검증으로 생성한 `logs/rsl_rl/.../model_0.pt`는 파쿠르 성능을 학습한 모델이 아니며, GUI 재생은 `checkpoints/go2_teacher.pt`와 `checkpoints/go2_student.pt`를 사용한다.

Student 학습도 같은 `train.py`를 쓰지만, 먼저 완성된 Teacher가 필요하다. 원본은 `logs/rsl_rl/unitree_go2_parkour/<run>/model_*.pt`에서 Teacher를 선택한다. `./run.sh train-student --load_run <teacher_run> --checkpoint <teacher_checkpoint_name>`로 대상을 명시해야 하며, 1회 검증용 Teacher를 최종 Teacher로 사용하면 안 된다.

## 작업 공간과 환경

- `upstream/Isaaclab_Parkour`: 원본 Git checkout와 최소 호환성 수정. 기준 commit은 `d6766d8883cffe427f2d4c087717d4eca1f23d8c`.
- `.venv`: 프로젝트 전용 패키지 설치 위치. `/home/user/isaac/env_isaaclab`의 기존 Isaac Sim/PyTorch를 `.pth`로 읽어 재사용한다. 기존 가상환경 패키지를 교체하지 않았다.
- `checkpoints/manifest.json`: 다운로드 출처, 고정 commit, SHA-256 및 검증한 Git blob 해시.
- `logs/environment.json`: 실제 검증 환경 버전.
- `logs/`: 실제 실행 로그, TensorBoard 기록, JSONL 측정값.
- `artifacts/`: 실제 GUI 녹화 영상.
- `rbq10_parkour/`: 원본을 상속하는 별도 RBQ10 모델·환경·runner 설정과 순서/센서 계약.
- `assets/rbq10/manifest.json`: 공식 모델·보행 정책의 고정 commit, 출처와 SHA-256.
- `patches/phase1_compat.patch`: Phase 1 원본 호환 수정 보존.
- `patches/phase2_port.patch`: Phase 2 추가 upstream 수정.
- `patches/phase1_phase2_combined.patch`: Phase 2 완료 시점의 upstream 대비 변경 보존.
- `patches/phase3_training.patch`: Phase 2 이후 upstream train 진입점의 추가 변경.
- `patches/current_upstream.patch`: Phase 1~3의 현재 upstream 대비 변경 전체.

이 전용 환경은 기존 Isaac Sim 설치에 의존하므로 독립 배포용 가상환경은 아니다. 런처가 프로세스 안에서만 `PYTHONPATH`를 설정한다. 기존 IsaacLab의 `workspaces/rbq10_wmp`, `workspaces/rbq10_pick_place`는 수정하지 않았다.

환경 등록 확인: `./run.sh list`. 정책을 만들지 않고 센서·환경 초기화만 확인: `./run.sh probe-student --num_envs 4`.

공개 Google Drive 링크는 작업 시점에 HTTP 404였으므로 공개 포크에서 모델을 확보했다. 원본 Drive 배포본과 바이트 단위 동일성은 확인할 수 없다. 출처가 다른 Isaac Gym 모델을 호환 모델로 가장하거나 임의 정책을 만들지 않았다.

확인한 Go2 영상: Student Jump/Climb (원PC 기록: `artifacts/student_jump_climb.mp4`), Teacher (원PC 기록: `artifacts/teacher_step.mp4`). 일부 환경에서 정체·실패가 있으며 상세 수치와 한계는 Phase 1 보고서에 포함했다. Phase 3은 headless 학습 중이며 최종 파쿠르 성능·GUI 평가는 아직 수행하지 않았다.
