# Student 도달 성능 비교

2026-10-23. Student는 620회 업데이트 후 `model_619.pt`를 저장하고 잠시 평가했다. Teacher는 증류 대상으로 선택한 `model_3674.pt`다.

평가는 각 역할/난이도마다 독립 Isaac 프로세스에서 20환경×3 episode=60회를 실행한다. 같은 Student 지형 설정(horizontal scale0.1, simplify), 5종 지형 동일 비율, seed42, nominal friction0.9, 명령 vx0.6m/s, 20초 episode, action delay1 step이다. 두 정책 모두 측정 body velocity와 heading command를 받고 history encoder를 사용한다. Student만 Depth CNN/GRU 특징을 사용한다. push/mass/COM/camera randomization은 평가에서 껐다.

이 조건은 과거 Teacher 전용 level9 58/60 평가와 다르다. Teacher/Student를 이번 조건에서 함께 재평가한 값끼리 비교해야 한다. 단일 seed 소표본이고 MuJoCo/실물 성능이 아니다. 전체 “능력의 몇 %”가 아니라 해당 조건에서의 코스 완주율 비교다.

중간 level4에서 Teacher60/60, Student50/60으로 상대 완주율83.3%다. Student는 평지/허들/턱/징검 플랫폼 각각12/12이며 gap만2/12다. 최고 level9 결과는 평가 완료 후 아래에 기록한다.

중간 징검돌 MuJoCo는 프로젝트의 `play_stones_medium.sh`로 실행한다. 최신 저장 Student, difficulty0.5, seed42, known parkour label, 키보드 모드다. W3회=전진0.3m/s, A/D 회전, Space 명령0, R reset. Isaac의 징검돌 결과가 MuJoCo에서 그대로 재현된다는 의미는 아니다.

평가 도구는 `tools/compare_teacher_student.py`다. 독립 context를 보장하기 위해 한 invocation에서 한 role/level만 지원한다. headless 실행 경로도 스크립트에서 보완했다. 첫 다중-role 평가에서 reset 후 다음 phase가 끝나지 않아 이후 평가는 프로세스를 분리했다. 첫 Teacher level4 점수와 이후 독립 프로세스 점수를 합산하며 score JSON 원본을 보존한다.

## 최종 비교 결과

| 고정 단계 | Teacher | Student | Teacher 대비 완주율 |
| --- | --- | --- | --- |
| 중간 level4 | 60/60 (100%) | 50/60 (83.3%) | 83.3% |
| 최고 level9 | 47/60 (78.3%) | 28/60 (46.7%) | 59.6% |

이는 Student가 모든 능력에서 일괄83% 또는60%라는 뜻이 아니다. 중간 난이도에서는 gap을 제외한 4종이 각각12/12였지만 gap은2/12다. 환경/조건/seed가 바뀌면 결과가 바뀐다. 기존 Teacher 전용58/60은 delay/지형 profile/velocity 입력이 달라 이번표의 분모로 쓰지 않는다.

원본 episode별 결과와 조건은 `evaluations/student_comparison/comparison.json`에 있다. 학습은 평가 snapshot의 실제 depth optimizer와 counter를 복원해 남은380회(총1,000회 목표)로 재개했다.
