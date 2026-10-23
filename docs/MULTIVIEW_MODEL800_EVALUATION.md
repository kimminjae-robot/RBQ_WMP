# 5뷰 Student model800 — 최신 고정 평가와 README 시연

2026-10-23, 검증 시작 시 최신 저장본 `model_800.pt`를 고정했다. 새 5뷰 분기에서801 update,
전방 model1549의1,550 update로 초기화한 모델이다. 실행 중 학습은 별도로 계속 진행한다.
SHA-256: `e6b33f0aaf510e731a41458a6f3a15289f389c4c5a9cad2cf334e2710403223d`.

## 같은 조건의 MuJoCo 20회

MuJoCo3.3.0 / policy50Hz / physics200Hz / Depth10Hz / 이전 영상 buffer / action1step 지연 /
raw action±4.8 제한 / scale0.25 / firm contact / source slope 2cm hfield를 사용한다.
5개 카메라의 실제 pose·intrinsics와 학습 전처리를 사용하고 Export 출력 오차는0.0이다.
알려진 지도 waypoint와 terrain label, MuJoCo 측정 body velocity/contact를 제공한다.
장애물 height scan/physics truth와 Teacher action은 Student에 공급하지 않는다.

| 지형 | 난이도 | Seed | 완주 / 실행 |
|---|---:|---|---:|
| 평지 | 0.5 | 42/43/44 | 3 / 3 |
| Hurdle | 0.5 | 42/43/44 | 3 / 3 |
| Step | 0.5 | 42/43/44 | 3 / 3 |
| Gap | 0.5 | 42/43/44 | 3 / 3 |
| Platforms | 0.5 | 42/43/44 | 3 / 3 |
| Mixed | 0.5 | 42/43/44 | 3 / 3 |
| Step | 1.0 | 42 | 0 / 1 |
| Gap | 1.0 | 42 | 0 / 1 |

중간 조건18/18, 전체18/20이다. 완주는 순서대로8개 목표점 도달, 최대30초,
전도는 |roll/pitch|>1.5rad 또는 base world-Z<0.15m다. 낙상 시 reset하지 않는다.
이전 model400과 같은 작은 개발용 집합으로, held-out 성공률은 아니다.
이전 중간 Step0/3 → 이번3/3, 중간 전체15/18 →18/18이지만 최고 Step은1/1 →0/1이다.
따라서 모든 지형의 향상이나 학습 수렴으로 판단하지 않는다.

최고 Gap은4.38초·목표2/8에서 낙하했다. 최고 Step은5.12초·목표2/8에서 전도했다.
이번 최고 Step에서 관절 속도69.14rad/s를 관측했으며 Isaac 설정50rad/s와 차이가 있다.
두 실패를 숨기거나 이전 모델의 성공 점수로 바꾸지 않았다.

[원시 20조건 요약](../evaluations/multiview_sim2sim_model800/rollouts/summary.json) ·
[Export manifest](../evaluations/multiview_sim2sim_model800/export/manifest.json).

## README 대표 영상

### Isaac Sim: 높은 Step

model800 / parkour_step / level9 / seed42 / vx0.6, 높이0.45m의 Step 코스다.
실제 Student로8.68초에8/8 목표를 완주했다. 명목 마찰을 사용하고 push·mass/CoM·camera
randomization을 끈 촬영 조건이며, 원본 action 지연과 timing은 유지했다.
12초 전체 영상에는 완주 후 자동 reset이 포함된다. README GIF는0~8.64초, reset 직전까지의
성공 episode를 보여준다. 접촉·몸체 높이·목표점·reset 이력을50Hz로 저장했다.

[GIF](../artifacts/student_depth5_model800_isaac_step_l9.gif) ·
[전체 MP4](../artifacts/student_depth5_model800_isaac_step_l9.mp4) ·
[접촉/상태/종료 원본](../artifacts/student_depth5_model800_isaac_step_l9.json).

### MuJoCo: 중간 스테핑 스톤

동일 model800 / isaac_platforms / difficulty0.5 / seed42 / vx0.6이다.
자동 지도 목표점 주행으로9.52초에8/8 목표를 완주했고20조건 평가의 해당 case와 일치했다.
완주 후 마지막 정지 화면은 종료 표시이며 추가 물리 진행이 아니다.

[GIF](../artifacts/student_depth5_model800_mujoco_platforms.gif) ·
[MP4](../artifacts/student_depth5_model800_mujoco_platforms.mp4) ·
[주행 원본](../evaluations/multiview_sim2sim_model800/video_platforms.json).

두 시뮬레이터의 숫자 seed가 같아도 생성 호출 순서가 달라 같은 배치라고 보장할 수 없다.
이 두 영상은 동일 맵의 PhysX/MuJoCo 짝 비교가 아니다. README의 이전 Teacher 낙상 GIF는
최신 Student의 대표 영상에서 제외하고 과거 Teacher 평가 문서에서 보존한다.

## 남은 차이와 다음 검증

카메라 TF/전처리/지연 검사는 [model400 전이 문서](MULTIVIEW_SIM2SIM.md)에 있다.
실물 전방 FOV는 학습값과 다르며 실제 CameraInfo 보정이 필요하다. 모터 속도 제한,
접촉 응답·수직 모서리 표현, 자기 몸 가림·노이즈·동기화도 완전히 같지 않다.
다음은 최고 Step/Gap의 같은 맵·초기 상태 비교와 held-out 평가다.
[실행/맵 옵션](DEPTH5_QUICKSTART.md) · [현재 상태](STATUS.md).
