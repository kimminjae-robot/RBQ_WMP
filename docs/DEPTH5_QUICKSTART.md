# 5뷰 Student: 다른 PC 실행·맵 선택·학습 재개

## MuJoCo만 실행

아래 경로는 **검증한 실제 model800 Export를 포함**하므로 Isaac Sim과 학습 checkpoint를
준비하지 않아도 MuJoCo로 재생할 수 있다. Ubuntu/Python 3.11 기준이며 GUI에는
사용 가능한 OpenGL 환경이 필요하다.

```bash
git clone https://github.com/kimminjae-robot/RBQ_WMP.git
cd RBQ_WMP
python3.11 -m venv .venv-sim2sim
.venv-sim2sim/bin/pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
.venv-sim2sim/bin/pip install -r sim2sim/requirements.txt

# 원본 지형 함수: 고정 commit. rollout에는 Isaac 학습용 patch/venv가 불필요하다.
git clone https://github.com/CAI23sbP/Isaaclab_Parkour.git upstream/Isaaclab_Parkour
git -C upstream/Isaaclab_Parkour checkout d6766d8883cffe427f2d4c087717d4eca1f23d8c

./run.sh sim2sim \
  --policy-dir evaluations/multiview_sim2sim_model800/export \
  --course isaac_platforms --difficulty .5 --terrain-seed 42 \
  --speed .6 --seconds 30 --autonomous --viewer --hold-seconds 60
```

이미 upstream을 사용 중이면 기존 revision/변경 상태를 먼저 확인한다.
학습 환경을 구성했다면 준비된 `upstream/Isaaclab_Parkour`를 그대로 사용한다.
`--autonomous`는 알려진 현재/다음 지도 목표점을 사용하며 지형 종류도 사전 지정한다.
미지 지형의 순수 비전 경로 계획 검증 모드는 아니다.
완주·낙하 때 자동 reset하지 않고 마지막 화면을 `--hold-seconds`만큼 표시한다.

## 맵과 옵션

| `--course` | 지형 |
|---|---|
| `isaac_flat` | 평지 보행 |
| `isaac_hurdle` | 연속 장애물 |
| `isaac_step` | 턱·계단식 플랫폼 |
| `isaac_gap` | 틈새 건너기 |
| `isaac_platforms` | 스테핑 스톤 |
| `isaac_mixed` | 혼합 코스 |

| 옵션 | 의미 |
|---|---|
| `--difficulty 0.5` | 생성 난이도 0~1 |
| `--terrain-seed 42` | 같은 맵/난이도에서 같은 배치 |
| `--speed 0.6` | 전진 속도 명령 m/s. 원본 학습 범위는 주로 0.3~0.8 |
| `--seconds 30` | 최대 시뮬레이션 시간 |
| `--autonomous --viewer` | 목표점 기반 자동 GUI 시연 |
| `--keyboard` | 수동 키보드. autonomous와 동시 사용 불가 |
| `--video artifacts/my_run.mp4` | 실제 render를 MP4로 저장 |
| `--output outputs/my_run.json` | 상태/목표점/종료 결과 저장 |
| `--terrain-edges source` | 기본 2cm 원본 경사 보정 hfield |
| `--contact-profile firm` | 이번 검증의 기본 접촉 설정 |

기존 `flat/hurdle/step`은 별도의 단순 물리 진단 코스다. `--height`는 단순 hurdle/step에만
적용되며 `isaac_*` 코스는 `--difficulty`로 조절한다.
임의 위치/크기의 장애물을 지정하는 custom-map CLI는 아직 없다.

키보드는 위 명령의 `--autonomous --viewer`를 `--keyboard`로 바꾼다.
GUI를 클릭한 뒤 W/S로 속도 증감, A/D로 회전 명령 증감, Space 명령0, R reset, Q 종료다.
키를 놓아도 설정한 명령을 유지하며 넘어지면 R로 다시 시작한다.
`sim2sim-latest`, `demo_autonomous.sh`, `play_stones_medium.sh`는 과거 전방 모델용이다.
5뷰에는 위 `--policy-dir` 또는 직접 Export한 5뷰 디렉터리를 사용한다.

## model700에서 학습 재개

```bash
# 가중치/Adam/counter 전체 보존본. 표준 Python만으로 복원한다.
python3 tools/restore_student_checkpoint.py --directory checkpoints/rbq10_student_multiview
```

복원 파일은 `checkpoints/rbq10_student_multiview/model_700.pt`, SHA-256은
`a7d7feed127e0273ffb5f0f7a3821676409a22c98a1de9af12290e77cfac1799`다.
조각·압축본·복원본 hash를 확인하며 다른 기존 checkpoint를 덮어쓰지 않는다.

[PORTABILITY.md](PORTABILITY.md)에 따라 Isaac Sim5.1 / Lab2.3.2 / Python3.11 및 프로젝트
venv를 구성한다. 그 후:

```bash
./run.sh multiview-student-job start \
  --checkpoint checkpoints/rbq10_student_multiview/model_700.pt \
  --num_envs 192 --iterations 1299 --seed 1
./run.sh multiview-student-job status
```

701회 이후 추가1,299회로 새 분기 목표2,000회에 도달한다. model700의 평가 수치를
model800에서 가져오지 않는다. 원래 PC 학습은 별도로 계속 실행된다.
물리 상태·episode·curriculum/RNG 분포는 재초기화되며 궤적의 bitwise 재현은 보장하지 않는다.

model700 또는 이후 **실제 5뷰 checkpoint**를 새로 Export하려면 별도 MuJoCo venv에
Export 의존성을 추가한다. 이미 배포한 model800 재생만 할 때는 이 설치가 불필요하다.

```bash
.venv-sim2sim/bin/pip install torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu
.venv-sim2sim/bin/pip install gymnasium==1.2.1 rsl-rl-lib==2.3.3
./run.sh export-multiview \
  --checkpoint checkpoints/rbq10_student_multiview/model_700.pt \
  --output outputs/sim2sim/depth5_model700
./run.sh sim2sim --policy-dir outputs/sim2sim/depth5_model700 \
  --course isaac_mixed --difficulty .5 --autonomous --viewer
```

원본 PPO/Reward/Curriculum은 재작성하지 않는다. MuJoCo 환경을 위해 기존 학습 venv를
수정하지 않는다. 검증 도구 전체 실행에는 추가로 Isaac Lab 소스·Warp·trimesh가 필요하며
일반 재생의 필수 의존성과 구분한다.

## 검증 결과와 남은 작업

최신 model800은 중간 난이도18/18, 전체20회 중18회 완주했다. 최고 Step과 Gap은 각각0/1이다.
모터 속도 제한·접촉·지형 모서리와 실물 전방 FOV 차이가 남아 있다.
모든 지형의 안정적인 전이나 실물 적용 완료를 뜻하지 않는다.
[최신 평가·영상](MULTIVIEW_MODEL800_EVALUATION.md) · [카메라 대조·한계](MULTIVIEW_SIM2SIM.md).
