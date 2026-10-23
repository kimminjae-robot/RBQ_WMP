# 현재 상태와 남은 단계 — 2026-10-23

**Student의 1차 학습은 1,000 update에서 정상 종료했다.** 정한 학습량을 완료했으며,
모든 지형의 성능 수렴이나 실물 배포 준비를 완료한 상태는 아니다.

[성능 개선 로드맵](PERFORMANCE_ROADMAP.md) ·
[GitHub 후속 과제 #1](https://github.com/kimminjae-robot/RBQ_WMP/issues/1) ·
[평가 근거 JSON](../evaluations/performance_baseline_20261023.json).

## 단계별 상태

| 단계 | 상태 | 현재 확보한 것 / 남은 것 |
|---|---|---|
| Phase 1 — 원본 Go2 | 완료 | 공개 원본 Teacher/Student 실행 및 정책 재생 |
| Phase 2 — RBQ10 이식 | 완료 | 공식 로봇 모델, 관절/PD/발 접촉/카메라 계약과 물리 검증 |
| Phase 3 — RBQ10 Teacher | 학습 저장·종료 | 4,833 update까지 진행. Student 증류 대상으로 model3674 선정 |
| Phase 4 — Vision Student | 1차 학습 완료·추가 검증 필요 | 1,000 update, 2,304만 transitions, 최종 model999. 동등 조건의 최종 비교 필요 |
| Phase 5 — 통합 검증·배포 | 진행 중·미완료 | 일부 MuJoCo 시연 완주. 다중 시드/Mixed/Gap, Crawl, 실물 검증은 남음 |

## 성능을 판단할 때 구분할 것

- 과거 동일 조건 Teacher3674/Student619 비교는 level4에서60/60 대50/60,
  level9에서47/60 대28/60이다. 중간 Gap은12/12 대2/12였다.
  [조건과 결과](STUDENT_PROGRESS.md). 이를 최신 model999의 점수로 사용하지 않는다.
- 최종 model999의 촬영용 level9/vx0.8/10 episode 결과는4/10이다.
  과거 vx0.6/60 episode와 비교해 향상/퇴보를 단정할 수 없다.
- 로컬 MuJoCo model941/난이도0.5/seed42에서 징검돌·허들8/8 목표 완주를 확인했다.
  징검돌 seed43/44와 Gap seed42에는 낙상이 남아 있다. 특정 시연 성공은 일반화 성공률이 아니다.
- 과거 Teacher 전용 level9의58/60은 지형 profile·action delay·velocity 조건이 다른 별도 시험이다.
  [과거 Teacher 평가](LEVEL9_EVALUATION.md)를 현재 Student나 양 엔진 비교의 분모로 쓰지 않는다.

## 다음 작업

1. 최종 Teacher/Student를 같은 지형·초기 상태·명령·센서/지연 조건으로 재평가한다.
   개발/보류 지형 seed를 나누고 Mixed 포함6종과 난이도별 실패를 기록한다.
2. raw Depth·시간축·구동/접촉·지형 경계 차이를 동일 상태/작은 물리 시험으로 정량화한다.
   기존 좌표 변환이 맞는 것과 동역학이 같은 것은 구분한다.
3. 결과에 따라 원본 Student 증류를500 update 단위로 추가하고, Gap·플랫폼 표본과
   curriculum을 조정한다. Teacher도 못 푸는 조건만 별도 Teacher 보강을 검토한다.
4. 실제 카메라 intrinsics/FOV/stream·지연/노이즈, joystick/측정 velocity/contact 추정 조건을 맞춘다.
   시간 메모리/GRU 구조 변경은 사용자의 보류 요청을 유지한다.
5. Mixed 연결 구간과 3D 낮은 천장 Crawl을 추가 검증한다. 수직 벽 부착 등반은 제외한다.

각 작업의 체크리스트와 **제안** 완료 기준은 [로드맵](PERFORMANCE_ROADMAP.md)에 있다.
이번 후속 과제 등록에서 새 학습이나 성능 개선을 실행한 것은 아니다.

## GitHub 배포 범위

이번 업데이트는 문서·후속 과제·평가 요약이다. 공개 main의 기존 실행 코드는
`32c3792` 시점의 기준이며 이후 로컬 자동 시연/bridge 수정과 model941/999가 모두 배포된 것은 아니다.
재현 가능한 코드·의존성·가중치 배포를 P0 작업에 포함했다. 새 checkpoint를 받은 것으로
가정하고 기존 다운로드/실행 명령을 안내하지 않는다.

저장소의 기존 Teacher snapshot은 model3674이며 과거 재개/평가 안내는
[README](../README.md)에 있다. 기존 환경·다른 프로젝트·Go2 태스크는 보존한다.
