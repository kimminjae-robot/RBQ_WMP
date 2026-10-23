# 현재 상태와 남은 단계 — 2026-10-23 배포 snapshot

**현재는 전방 1개 + 하향 4개 Depth의 5뷰 Student만 학습 중이다.** 목표는 새 분기
2,000 update이며, 배포용으로 고정한 저장본은 **model700 / 701 update**다.
전방 전용 Student는 사용자 요청으로 1,550 update에서 전체 상태를 저장하고 중단했다.
실시간 진행은 `./run.sh multiview-student-job status`로 확인한다.

## 단계별 상태

| 단계 | 상태 | 확보한 것 / 남은 것 |
|---|---|---|
| Phase 1 — 원본 Go2 | 완료 | 원본 Teacher/Student 실행과 정책 재생 |
| Phase 2 — RBQ10 이식 | 완료 | 공식 모델·관절/PD·접촉 센서·카메라 TF 및 기본 물리 검사 |
| Phase 3 — RBQ10 Teacher | 저장·종료 | 4,833 update까지 진행, 증류 대상으로 model3674 선정 |
| Phase 4 — Vision Student | **5뷰 학습 중** | 전방 모델 1,550 update 보관. 별도 5뷰 분기 2,000 update 목표, model700 전체 재개본 배포 |
| Phase 5 — 통합 검증·배포 | **진행 중·미완료** | 5뷰 MuJoCo 실행과 20조건 검사, 센서 입력 대조. Step/최고 Gap 안정성·실물 검증은 남음 |

## 최신 model800 검증

새 분기801 update인 model800으로 같은 MuJoCo 고정20조건을 재평가했다. 중간 난이도는
6종 각각3/3, 합계18/18이다. 최고 Step/Gap은 각각0/1로 전체18/20이다.
Isaac Sim의0.45m Step은8.68초, MuJoCo 중간 스테핑 스톤은9.52초에8/8 목표를 완주했다.
README 대표 영상을 이 두 최신 Student 시연으로 교체했다.
[최신 평가·영상·한계](MULTIVIEW_MODEL800_EVALUATION.md).

## 이전 model400과 카메라/전이 검사

- **5뷰 model400 MuJoCo:** 중간 난이도 6종 × seed42/43/44의 18회 중 15회 완주.
  Step은 0/3, 나머지 5종은 각각 3/3이다. 추가 최고 난이도 seed42에서 Step 1/1,
  Gap 0/1이다. 전체 16/20이며 작은 개발용 고정 집합의 결과다.
- **카메라:** 5개 MuJoCo camera TF·intrinsics·raw/preprocessed Depth·10Hz 지연 버퍼를
  Isaac 소스와 대조했다. 동일 평면·박스 fixture에서 최대 Depth 차이는 약 0.0123mm.
  실제 파쿠르의 mesh/hfield 모서리 차이는 남는다.
- **수정:** 5뷰 Export/입력 추가, 원본 raw action ±4.8 제한 반영, batch ray 표면 누락을
  검출해 scalar ray로 교체했다. 이전 무효/제한 누락 상태의 점수는 최종 결과에 섞지 않았다.
- **시각화:** 높이별 색상·노란 경계·어두운 틈·로봇/목표점 대비를 적용했다.
  색상 전후 5장면의 물리 배열·100 physics-step·정책 Depth가 정확히 같았다.
- **Isaac 영상:** 이전에 촬영한 model200의 Gap/Step 영상도 포함한다.
  model400의 MuJoCo 평가와 같은 모델·같은 맵의 짝 비교로 취급하지 않는다.

[전이 검사·실패 원인·한계](MULTIVIEW_SIM2SIM.md) ·
[공식 카메라 TF/학습 구조](MULTIVIEW_STUDENT.md) ·
[다른 PC 빠른 시작과 맵 옵션](DEPTH5_QUICKSTART.md).

## GitHub에 포함한 모델

| 모델 | 용도 | 보관 위치 |
|---|---|---|
| Teacher model3674 | 기존 Teacher/증류 기준 | `checkpoints/rbq10_teacher/` |
| 전방 Student model1549 | 중단한 1,550 update 원본 | `checkpoints/rbq10_student_front/`, gzip 조각 복원 |
| 5뷰 Student model800 | **최신 고정 평가·시연** | `evaluations/multiview_sim2sim_model800/export/`, TorchScript |
| 5뷰 Student model400 | **이번 20조건 평가 그대로 실행** | `evaluations/multiview_sim2sim_model400/export/`, TorchScript |
| 5뷰 Student model700 | **701 update 이후 학습 재개** | `checkpoints/rbq10_student_multiview/`, gzip 조각 복원 |

model700의 점수는 이번 20조건에서 아직 측정하지 않았다. model400 점수를 재사용하지 않는다.
전체 재개본에는 Student·frozen Teacher·증류 Adam·counter가 들어 있다.
물리 episode·curriculum 분포·전체 RNG 상태는 재초기화되므로 같은 궤적 재현을 보장하지 않는다.
이전 model941/999 시연은 과거 전방 모델 기록이며 최신 5뷰 성능이 아니다.

## 남은 작업과 우선순위

1. 진행 중인 5뷰 2,000 update 완료 후 별도 평가한다. 개발/보류 seed,
   6종 지형·난이도를 나누고 Teacher/Student를 같은 조건에서 비교한다.
2. **중간 Step 실패를 우선 해결한다.** 해당 3회에서 50rad/s 초과를 측정했다.
   같은 맵·상태·command의 PhysX/MuJoCo 관절 응답·접촉 impulse와 모서리를 대조하고
   실제 구동계에 근거해 조정한다. 속도를 강제로 잘라 성공 결과를 만들지 않는다.
3. **최고 Gap 실패**의 이륙/착지, 발 디딤·표본 및 지연 민감도를 확인한다.
   Teacher도 같은 조건에서 실패하는 구간만 별도 Teacher 보강을 검토한다.
4. **실물 카메라 계약:** 전방 FOV는 학습값 약87°/58°, 실물 명목값127°/79.5°로 차이가 있다.
   실제 CameraInfo 기반 crop/reprojection 또는 별도 학습이 필요하다. 하향 4개의 nominal
   FOV와 TF는 반영했지만 개별 intrinsics/동기화·노이즈·가림은 실측해야 한다.
5. 조이스틱 방향 명령·측정 body velocity·contact 추정과 알려진 지도 목표점 시연을
   구분해 실물 입력을 검증한다. 낮은 천장 Crawl·연속 코스의 보류 지형 검사도 남아 있다.
   시간 메모리 구조 변경은 사용자 보류 요청을 유지하며 수직 벽 부착 등반은 제외한다.

[기존 성능 개선 로드맵](PERFORMANCE_ROADMAP.md)과
[GitHub 후속 과제 #1](https://github.com/kimminjae-robot/RBQ_WMP/issues/1)은 계속 참고한다.
기존 설치·다른 프로젝트·Go2 환경과 실행 중인 학습은 이번 배포로 수정/중단하지 않았다.
