# Teacher 마지막 단계 고정 평가

- 모델: `checkpoints/rbq10_teacher/model_3674.pt` — 3,675 PPO updates, 실제 프로젝트 학습 모델.
- SHA-256: `073bf99cb7f4b4afdcef9c0c338c4bc426fcab57d497af6a5fc2be9c6304b7af`.
- 레벨: 10단계 중 index 9, difficulty **1.0**. 원본 10행 generator의 마지막 행이며 reset 뒤에도 이 단계에 고정했다.
- 평가: seed 42, 20개 환경, 환경당 최초 3개의 완전한 episode, 총 60회. 지형별 4개 환경 × 3회 = 12회, 지형별 두 개의 mesh 배치다.
- 원본 Teacher의 estimator + history encoder (`hist_encoding=True`) 추론, action 12개, physics 0.005s / 제어 0.02s.
- 전진 명령 0.6m/s, heading 0, 마찰 0.9, push·질량/COM randomization 없음. 원본 roughness와 관절 초기 scale 0.95~1.05 유지.
- 원본 terrain/reward/종료 조건을 재사용했다. 새 `fixed_terrain_level=None` 기본값은 원래 학습 curriculum을 보존하고 평가 때만 9를 지정한다.

## 결과

| 지형 | 완주 | 낙상 | 시간 초과 |
| --- | --- | --- | --- |
| flat | 12 | 0 | 0 |
| hurdle | 12 | 0 | 0 |
| step | 10 | 1 | 1 |
| gap | 12 | 0 | 0 |
| parkour 발판 | 12 | 0 | 0 |

총 58/60 (96.7%)의 **코스 완주**다. 최종 goal index 8 이상이고 낙상이 없을 때 성공으로 기록했다. 원본 낙상 기준은 roll/pitch 절댓값 >1.5 rad 또는 world z<−0.25m이다. 나머지는 20초 제한에 따른 시간 초과로 분류했다. 원본의 합쳐진 `time_out` flag에 의존하지 않고 auto-reset 직전 실제 terminal state를 읽었다.

Hurdle 높이는 원본 difficulty 1 식에서 0.20~0.40m, step은 단차 0.45m(3단 최고 1.35m), gap 폭은 0.80m다. 이는 생성 설정의 값이며 모든 장애물에 대해 실측 성공률을 별도 산출한 것은 아니다.

## 영상과 데이터

[전체 50초 MP4](../artifacts/rbq10_teacher_3674_level9.mp4)는 flat → hurdle → step → gap → 발판을 각 10초씩 같은 실제 simulation에서 추적한다. HUD에 checkpoint·level·difficulty·terrain·simulation time을 표시했다. 실패와 자동 재시도도 포함하며 성공 장면만 골라 만든 영상이 아니다. 평가는 55.76 simulation seconds까지 진행해 모든 episode를 수집했다.

[8초 GIF](../artifacts/rbq10_teacher_3674_level9.gif)는 전체 영상의 Step 구간 발췌다. [JSON](../evaluations/level9_model3674/results.json), [CSV](../evaluations/level9_model3674/results.csv)에 60개 실제 episode의 결과·goal index·길이·최대 접촉력·추정 torque가 있다.

## 해석의 한계

동일 seed·명목 물리 조건·소수 배치의 반복 시험이다. 무작위 seed/외란/센서 지연/실물로 일반화했다고 주장하지 않는다. Hurdle/Gap 완주는 로봇 몸체 길이와 stepping으로 가능할 수 있으므로 공중 점프 성공률과 구분한다. Implicit PD torque는 Isaac Lab의 근사값이고 contact force도 제어 시점 표본이다. 충격·모터 제약 검증을 완료한 결과는 아니다. 이 모델은 privileged Teacher이며 Vision Student가 아니다.
