# RBQ10 Vision Parkour 성능 개선 과제

기준일: 2026-10-23. 사용자가 현재 성능의 개선 방법을 후속 과제로 남기도록 요청했다.
이 문서는 **앞으로 할 실험과 완료 기준**이다. 이번 문서 업데이트에서 학습을 재시작하거나
아래 목표 성능을 달성한 것은 아니다. 원본 PPO·Reward·Curriculum·Teacher/Student 구조를
유지하고, RBQ10 전용 설정과 검증 도구부터 개선한다.

## 판단: 추가 학습 전에 원인을 분리한다

현재는 두 문제가 함께 있다.

1. **Isaac 안의 모방 성능과 일반화:** 동일 조건의 과거 평가에서 Student는 Teacher보다
   Gap·높은 플랫폼 등에 약했다. 최종 model999의 동등 조건 재평가가 아직 없다.
2. **Isaac→MuJoCo 전이:** 속도 좌표계 오류를 수정한 뒤 평지 성능이 크게 좋아졌지만,
   지형 시드·접촉 설정에 따라 여전히 결과가 달라진다. 학습만으로 bridge 오류를 보상하게
   하면 원인을 가리고 실물 적용까지 이어질 수 있다.

따라서 **최종 정책 재평가와 전이 검증 → 취약 구간 추가 증류 → 필요할 때 Teacher 보강 →
실센서 조건과 연속 코스 검증** 순서로 진행한다. 신경망 대형화나 보상 재작성은 첫 조치가 아니다.

## 확인된 사실과 해석 한계

[평가 원자료를 모은 기준선 JSON](../evaluations/performance_baseline_20261023.json)에
원본 경로·파일 SHA256·조건·episode 결과를 보존했다.

| 근거 | 관측 결과 | 해석 가능한 범위 |
|---|---|---|
| Student 학습 | 1,000 update, 2,304만 transitions, model999 저장·정상 종료 | 정한 학습량 완료. 수렴/실물 준비 완료를 뜻하지 않음 |
| Isaac 동일 조건, **Student619 / Teacher3674**, level4, vx0.6 | Student50/60, Teacher60/60. Gap은 Student2/12, Teacher12/12 | 당시 중간 난이도의 Gap 모방 격차 |
| 같은 비교, level9 | Student28/60, Teacher47/60 | 당시 높은 난이도에서 Teacher와 Student 모두 실패가 있음 |
| **Student999**, level9, vx0.8, 10 episode | 전체4/10, 허들0/2. 촬영한 허들 episode는 목표6에서 낙상 | 작은 고속 시험. 위 vx0.6/60 episode 평가와 직접 비교 불가 |
| MuJoCo **Student941**, difficulty0.5, seed42, firm contact | 징검돌8/8 목표(9.52초), 허들8/8(17.10초) | 선정한 두 코스의 완주. 다중 시드 성공률이 아님 |
| 같은 MuJoCo 설정, 징검돌 seed43/44 | 각각 목표6/8에서 낙상 | 선택한 seed42 외 일반화가 부족함 |
| MuJoCo Gap seed42 | 목표7/8에서 낙상 | Gap 전이/진행 안정화 필요 |
| 동일 Isaac 상태200개를 MuJoCo에 주입 | 몸체 위치 최대오차3.85e-6m, 수정 후 속도 좌표변환 오차6e-16 이하 | 관절 배치·속도 좌표계 검증. 동역학·접촉·Depth가 같다는 뜻은 아님 |

전체 “Teacher 성능의 몇 %”라는 값으로 압축하지 않는다. 학습 로그의 최근 episode 통계,
고정 지형 평가, 다른 엔진 평가, 편집된 GIF는 각각 다른 근거다. 도약 GIF는 성공 구간을
확인하는 자료이며 전체 코스 완주의 증거로 쓰지 않는다. Mixed 전체 코스의 통제된 완주율은 아직 없다.

## 우선순위와 작업 순서

| 우선순위 | 과제 | 먼저 해결할 이유 | 주요 수정 후보 |
|---|---|---|---|
| P0-1 | 최종 checkpoint 동등 조건 평가 및 재현 자료 정리 | 어느 능력이 부족한지, 추가 학습이 나아졌는지 판단할 기준이 없음 | compare_teacher_student.py, evaluation.py, 평가 manifest |
| P0-2 | Depth·관측·PD·접촉 전이 차이 정량화 | 잘못된 입력/물리를 학습 횟수로 보상하지 않기 위해 | student_inputs.py, run_student.py, build_scene.py, terrains.py |
| P1-1 | Gap·플랫폼 중심의 추가 Student 증류 | 과거 평가에서 가장 큰 모방 격차. 최종 재평가 후 조정 | RBQ10 config.py/agents.py/student_runner.py, 원본 learn_vision 재사용 |
| P1-2 | Teacher의 높은 난이도 상한 보강(조건부) | Teacher도 못 푸는 구간을 모방만으로 해결하기 어려움 | RBQ10 Teacher 설정, 원본 PPO/curriculum |
| P1-3 | 실센서·명령 조건과 domain randomization | 현재 알려진 맵 방향 입력과 센서 정답 조건을 실물 조건에 연결 | camera 계약, RBQ10 randomization, command bridge |
| P2 | Mixed·Crawl·실물 전 단계 검증 | 단일 장애물 성공을 연속 코스 성공으로 확대해야 함 | mixed 평가, 3D 천장 지형, 센서/제어 시험 |

### P0-1 — 비교 가능한 기준선과 재현성

- [ ] 현재 로컬의 bridge/자동 시연 수정, 의존성, 코드 revision을 재현 가능한 형태로 정리한다.
      과거 공개 main `32c3792` 이후 로컬 수정이 있으므로 **평가 코드 버전과 파일 존재 여부부터** 맞춘다.
- [ ] model619/941/999와 Teacher3674를 동일 지형·reset·명령·물리·관측 조건으로 비교한다.
      최종 Student가 중간 snapshot보다 좋다고 미리 가정하지 않는다.
- [ ] 기존 평가 도구의 5종/환경 수 제약을 RBQ10 전용으로 확장해 **Mixed 포함 6종**을 평가한다.
      기존 `compare-policies` 명령은 그대로 6종을 지원한다고 가정하지 않는다.
- [ ] 난이도0/4/9, 우선 vx0.6으로 평가하고, vx0.3/0.8과 회전 명령은 별도 조건으로 분리한다.
      Teacher의 estimator/측정 velocity 사용, action delay, history/GRU 초기화 방식도 명시한다.
- [ ] 지형 생성물을 고정해 역할 간 공유하고 지형 seed와 초기 상태 seed를 따로 기록한다.
      개발 seed42~46과 **별도의 보류 seed100~104**를 사용한다. 반복된 같은 맵은 독립 맵으로 세지 않는다.
- [ ] 최종 비교는 각 지형·난이도·모델당 최소50 episode(5맵×10초기 상태)를 제안한다.
      작은 중간 시험과 최종 시험을 분리하고, 완주율에는 표본 수·95% 신뢰구간을 함께 낸다.
- [ ] 성공/낙상/timeout, 목표점 진행, 실패 장애물, 착지 충격, 토크 포화율, 관절 속도,
      추론 지연과 실제 평균속도를 기록한다. episode 성공과 장애물별 도약/착지 성공을 구분한다.
- [ ] checkpoint는 `최신`만으로 고르지 않고 보류 평가와 제약을 기준으로 선정한다.

**완료 기준:** 같은 입력 manifest로 실행·집계되는 표와 원자료가 있고, model999와941의
선택 이유가 조건별 성능과 실패 기록으로 설명된다. 추론용 배포본과 학습 재개용 checkpoint를
구분한다. 100MiB가 넘는 학습 checkpoint는 Git 일반 blob으로 올리지 않고 Release/LFS 등으로
제공하거나 optimizer를 뺀 추론본을 제공한다. 추론본은 원본과 weight/출력 일치 및 SHA256을 검증한다.

### P0-2 — Sim2Sim 차이를 같은 상태에서 측정

**관측과 카메라**

- [ ] 같은 base pose·joint state·지형으로 양쪽 엔진을 고정하고 카메라 origin, optical 축,
      intrinsics, ray 방향, raw metric Depth, crop/resize/normalize, 10Hz 갱신/버퍼 지연을 비교한다.
- [ ] 평면·단일 box·단차/Gap 등 해석 가능한 작은 scene부터 시작한다. raw Depth 오차와
      CNN/GRU latent·action 차이를 따로 기록해 전처리와 정책 문제를 분리한다.
      연속 표면 Depth MAE1cm 이하를 **진단 목표**로 제안하며, 경계/가림 오차는 별도로 숨김없이 보고한다.
- [ ] 동일 관측·latent 입력의 원본 PyTorch/TorchScript action 최대오차1e-5 이하를 유지한다.
      원본 scan/물리 truth 슬롯을 바꿔도 Student 출력이 영향받지 않는 계약도 재검증한다.
- [ ] IMU/COM 선속도 frame·단위·순서, previous action, contact OR history, 명령 heading hold와
      샘플 시점이 동일한지 trace로 검사한다. 수정한 주관성축 오류는 재발 방지 검증에 포함한다.

**지형과 구동/접촉**

- [ ] 원본 wrapper·목표점·수직 경계 보정을 포함한 지형을 비교한다. 2cm hfield의 남은 경사면과
      Isaac simplified mesh 차이가 발 착지/Depth에 만드는 오차를 측정한다.
- [ ] 같은 상태·같은 q_target의 1/20/200 physics-step 응답에서 joint q/dq, torque, base 변위,
      foot force/impulse를 비교한다. 정지 PD와 무접촉 joint sweep, 착지 시험을 순서대로 수행한다.
- [ ] `firm` 접촉은 한 맵을 완주하게 만든 경험적 보정이다. 여러 지형과 독립된 물리 시험으로
      solref/solimp·friction·충돌 필터·solver를 보정하고, 보류 지형에서 다시 검증한다.
- [ ] PhysX의50rad/s 제한과 MuJoCo 차이를 해결할 구동기 모델을 검토한다.
      관측된 속도 폭주를 이 차이 하나의 탓으로 단정하지 않는다. qvel을 사후 강제 잘라
      시험만 통과시키는 방법 대신 실제 토크-속도/응답 제약과 일관된 구현을 택한다.

**완료 기준:** 수정 전/후의 입력·응답 오차표와 재현 가능한 작은 시험이 있고, 동일 checkpoint로
Isaac/MuJoCo 보류 지형별 성공률 차이를 보고한다. 중간 난이도 격차10%p 이내를 후속 목표로
두되, 이를 이미 달성한 값이나 모든 지형의 보장으로 해석하지 않는다.

### P1-1 — 원본 Student 증류를 유지하며 약한 구간 학습

현재 원본 `learn_vision`은 **Student action으로 rollout하고 같은 상태의 Teacher action을
정답으로 사용한다.** 이 기능이 없는 것으로 오해해 새로운 imitation/DAgger 알고리즘을
처음부터 만들지 않는다.

- [ ] P0 재평가로 Gap/플랫폼의 실제 격차를 확인한 뒤, 실패 전후 관측·Depth·목표·Teacher/Student
      action과 적용 q_target/토크를 저장한다. per-joint 오차와 타이밍을 보며 scalar loss만 보지 않는다.
- [ ] Teacher/Student 방향 입력과 measured velocity를 동일하게 유지한다. Teacher가 보는 privileged
      정보를 Student 배포 입력으로 추가해 점수만 올리는 방법은 사용하지 않는다.
- [ ] 원본 curriculum/terrain 생성식을 유지하면서 RBQ10의 난이도 분포·지형 비율을 점검한다.
      Gap·플랫폼 실패 구간의 표본을 늘리되 평지·허들·턱의 회귀도 동시에 평가한다.
- [ ] model999의 실제 optimizer/counter에서 **500 update 단위**로 추가 증류하는 실험을 제안한다.
      우선 추가500~2,000 범위 안에서 중간 평가하며, 이는 개선 보장이나 이번 실행 지시가 아니다.
- [ ] 기존 learning rate1e-3을 기준으로 삼고, plateau/진동이 확인된 경우에만 예컨대3e-4와
      한 변수씩 비교한다. 모델 크기·새 loss·주기·난이도를 한꺼번에 바꾸지 않는다.
- [ ] 중간 loss가 낮아도 보류 성공률이 두 평가 연속 좋아지지 않거나 토크/착지/다른 지형이
      악화되면 추가 시간을 쓰기 전에 원인을 다시 분석한다. 최고 성능 checkpoint는 별도로 보존한다.

**제안 완료 기준:** 보류 중간 난이도에서 지형별90% 이상을 1차 목표로 한다. Teacher 성공률이
80% 이상인 조건에서 Student/Teacher 완주율90% 이상을 목표로 하되, 절대 성공률과 신뢰구간도
같이 보고한다. Teacher가 낮은 조건의 상대비만으로 Student가 충분하다고 판정하지 않는다.

### P1-2 — Teacher 자체가 못 푸는 구간만 조건부 보강

- [ ] 같은 조건의 Teacher 실패를 먼저 재현한다. 과거3674가4832보다 좋았던 작은 시험만으로
      모든 조건에서3674가 우수하다거나, 추가 PPO는 의미 없다고 결론 내리지 않는다.
- [ ] RBQ10 체격·토크/속도·지상고·발 위치에 맞는 장애물 폭/높이와 접근속도를 점검한다.
      Go2 지형의 절대 수치를 유지하는 것과 RBQ10의 물리적 가능성 검증은 구분한다.
- [ ] 원본 PPO/Reward/Curriculum을 재사용해 실패 구간과 착지 전이의 표본·난이도를 조정한다.
      Teacher를 바꾸면 별도 Student 실험으로 증류하고 teacher checkpoint SHA를 기록한다.
- [ ] 원본 reward로 해결되지 않는 근거가 있을 때만 RBQ10 전용 reward 계수/제약 변경을
      별도 ablation으로 검토한다. 토크 상한 확대나 충돌 비활성화로 성공을 만들지 않는다.

**완료 기준:** 동일한 보류 평가에서 개선됐고 기존 능력과 물리 제약을 유지한 Teacher를 선정한다.
Teacher 변경이 필요 없으면 이 과제는 실행하지 않고 Student/전이 개선에 자원을 쓴다.

### P1-3 — 실센서·명령과 학습 조건 맞추기

- [ ] 공식 카메라 외부 좌표는 적용했지만 intrinsics/FOV는 원본 설정을 유지하고 있다.
      실제 OAK-D Depth의 활성 stream 해상도·내부 보정·range를 측정해 계약을 갱신한다.
      새 계약이면 Student를 다시 학습/미세조정하며 이전 정책과 호환성을 임의로 가정하지 않는다.
- [ ] 실측 기반으로 noise/hole/dropout, 지연·jitter, 노출/반사 특성의 영향을 좁은 범위부터
      추가한다. latency·마찰·질량·COM·actuator 오차 randomization을 하나씩 검증한다.
- [ ] 실제 로봇에서 허용된 **측정 선속도와 reference 선속도**는 계속 분리한다.
      contact 추정 오차와 IMU/joint 측정 지연을 반영하되 센서 정답으로 조용히 대체하지 않는다.
- [ ] 알려진 지형 목표점 시연과 실물 joystick 명령을 구분한다. joystick 방향 command의
      heading hold/회전 응답을 학습/배포 양쪽에서 맞춘다. 실물에 terrain 목표점 정답이
      제공된다고 가정하지 않는다. 초기 실험의 flat/non-flat label은 사용자 구성에 맞춰 유지한다.
- [ ] 시간 메모리/GRU 구조 변경은 사용자의 기존 보류 요청에 따라 별도 과제로 남긴다.
      우선 기존 state 초기화/유지와 샘플 주기가 양쪽에서 같은지만 기록한다.

**완료 기준:** 실제 stream/명령 계약을 version/hash로 고정하고, 센서 조건을 바꾼 보류 평가에서
오차와 성공률을 제시한다. 공식 장착 좌표를 맞췄다는 사실만으로 카메라 보정 완료를 선언하지 않는다.

### P2 — Mixed·Crawl과 최종 적용 검증

- [ ] 단일 장애물 구간뿐 아니라 **장애물 사이 연결 구간의 착지·속도·회전**을 포함해 Mixed를 평가한다.
      난이도/배치/명령을 바꾼 보류 맵에서 전체 코스 완주·실패 장애물을 집계한다.
- [ ] Crawl은 현재 heightfield 6종만으로 구현되지 않는다. 낮은 천장은 3D 충돌체와 Depth ray 대상
      mesh로 추가하고, RBQ10 높이/자세 조건·보상·평가가 필요한 부분만 원본 구조에 확장한다.
      수직 벽 부착 등반은 계속 제외한다.
- [ ] 실제 배포는 훈련한 Student의 관절 순서/좌표/단위/PD/주기·지연·GRU 계약을 고정하고
      simulator replay와 실제 센서 기록 재생을 통과한 뒤 단계적으로 검증한다.

**제안 목표:** 최종 highest 난이도80% 이상, 중간 각 지형90% 이상, 양 엔진 차이10%p 이내를
장기 목표로 두고, 물리 제약 위반·센서 의존성과 함께 판단한다. 이는 현재 달성률이 아니며
실물 적용을 자동 승인하는 기준도 아니다.

## 변경 원칙과 근거

- 기존 정상 환경·Teacher snapshot·원본 Go2 태스크는 보존한다. 실험은 새 run/별도 RBQ10 설정으로 한다.
- loss/reward 그래프보다 보류 코스 성능·제약 위반·실패 유형을 checkpoint 선정 근거로 삼는다.
- [원본 Isaaclab_Parkour](https://github.com/CAI23sbP/Isaaclab_Parkour)와
  [원 논문](https://arxiv.org/abs/2309.14341)의 구조를 참고한다. 원 논문의 로봇 성능은 RBQ10 결과가 아니다.
- 평가 근거: [과거 동일 조건 Teacher/Student 비교](STUDENT_PROGRESS.md),
  [원자료 요약](../evaluations/performance_baseline_20261023.json).
  최신 로컬 model941/999와 자동 시연 코드/대용량 가중치는 이번 **문서·과제 업데이트**의 배포물이 아니다.

## GitHub 작업 목록

아래 이슈에서 진행과 완료 기준을 추적한다. 과제는 모두 미완료 상태로 등록한다.

- [이슈 #1 — RBQ10 Student 성능·Sim2Sim 일반화 개선](https://github.com/kimminjae-robot/RBQ_WMP/issues/1)
  에 P0/P1/P2 체크리스트를 등록했다. 보류 seed와 물리 제약을 포함해 완료 여부를 판단한다.
