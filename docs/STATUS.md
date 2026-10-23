# 현재 진행 상황과 남은 단계 — 2026-10-23 KST

## 현재

Phase 1·2를 완료했고 Phase 3 RBQ10 Teacher 학습을 진행 중이다. 배포 snapshot은 **3,675회 업데이트 후의 `model_3674.pt`**, 누적 541,900,800 transitions이다. 원본 PPO, Reward, Curriculum, RMA/history encoder를 재사용했다.

2026-10-22 20:25 KST에 디스크 여유 2GiB 미만으로 자동 저장·중단했다. 손상 없이 모델·auxiliary optimizer/RMA counter를 검증했다. **2026-10-23 09:14 KST**, 사용자가 여유 공간 약 22GB를 확보한 것을 확인하고 같은 모델에서 총 50,000회 목표로 재개했다. 최신 진행도는 각 PC의 `./run.sh teacher-job status`로 확인한다. 저장소 모델은 고정 snapshot이며 실행 중인 모델과 자동 동기화되지 않는다.

## 단계별 상태

| 단계 | 상태 | 확보한 결과 / 남은 것 |
| --- | --- | --- |
| Phase 1 — 원본 Go2 | 완료 | 실제 Teacher/Student 실행, 사전학습 정책 재생, 점프·Climb 확인. 원본 Drive 대신 pinned public fork 모델 사용 |
| Phase 2 — RBQ10 이식 | 완료 | 공식 40.684kg/12관절 모델, PD·foot·depth 계약, 정지·제조사 보행 정책의 평지·낮은 step 물리 검증 |
| Phase 3 — RBQ10 Teacher | 진행 중 | 실제 PPO 학습·checkpoint·안전 재개. snapshot 3674의 최고 단계 60개 episode 평가 완료 |
| Phase 4 — Vision Student | 미시작 | 완성 Teacher 선정 후 원본 distillation, 실센서 observation 계약, depth 노이즈·지연·물리 오차 검증 |
| Phase 5 — 최종 검증·배포 | 미완료 | Student의 평지·Jump·Climb·Leap·연속 코스 평가, recurrent export·MuJoCo Sim2Sim 검증 |

현재 모델은 **시뮬레이션 지형 정답 정보를 활용하는 Teacher**다. 실제 로봇 센서만 사용하는 Vision Student는 아직 없다.

## 마지막 단계 성능

10개 단계 중 9단계(difficulty 1.0)에 고정해 실제 Teacher 추론으로 20환경 × 3회 = 60 episode를 실행했다. [상세 결과](LEVEL9_EVALUATION.md), [영상](../artifacts/rbq10_teacher_3674_level9.mp4).

| 지형 | 완주 | 실패 |
| --- | --- | --- |
| 평지 | 12/12 | 없음 |
| Hurdle | 12/12 | 없음 |
| Step·턱 | 10/12 | 낙상 1, 시간 초과 1 |
| Gap·틈새 | 12/12 | 없음 |
| 발판 코스 | 12/12 | 없음 |

합계 58/60이다. 고정 seed와 명목 조건의 작은 시험이며 광범위한 일반화 성공률·독립적인 점프/착지 성공률·실물 성공률로 해석하지 않는다. 영상은 실패·auto-reset 재시도를 포함한다.

## 다음 작업

1. Teacher를 우선 총 5,000회까지 이어 학습하고 동일한 고정 단계 조건과 다른 seed/명령/마찰 조건으로 비교한다. 원본 RMA 정규화 counter가 아직 ramp 구간이므로 snapshot 3674를 최종 모델로 확정하지 않는다. 실제 재개는 원래 총 50,000회 목표를 유지한다.
2. Step 낙상·정체, 착지 접촉력과 모터 포화·관절 속도 위반을 분석한다. 학습 지표의 상승만으로 제약 충족을 판단하지 않는다.
3. 충분히 검증된 Teacher checkpoint를 선정한 뒤 Phase 4를 진행한다. 원본 Student의 simulator base velocity/terrain-type 정보, history/yaw/목표 입력 중 실물에서 얻을 수 없는 항목을 감사해야 한다.
4. Depth noise·시간 지연·도메인 랜덤화와 센서 장착 실측을 적용하고 Student를 학습한다.
5. 최종 GUI 평가와 MuJoCo Sim2Sim을 위해 ONNX/PyTorch의 관절/observation 순서, depth 축, recurrent hidden state 입출력을 수치 검증한다.
6. Crawl용 낮은 천장 지형·정책·평가와 모든 장애물을 섞은 연속 코스 검증은 아직 미구현/미검증이다. 수직 벽 부착 등반은 현재 범위에서 제외한다.

이번 배포에 Teacher·영상이 포함됐다고 Phase 4/5나 실물 적용까지 완료됐다는 뜻은 아니다. 기존 작업 공간과 기본 Isaac 환경은 보존했다.

## MuJoCo 초기 bridge (2026-10-23)

Teacher TorchScript export, 공식 URDF 변환, 관측값 bridge, headless 실행을 구현했다. 관측값 fixture와 10초 정지 PD는 통과했으나 학습 정책은 짧은 실행에서 넘어져 안정적인 Sim2Sim 전이는 미완료다. Student 및 기존 Isaac 학습 설정은 바꾸지 않았다. [실행/한계](SIM2SIM.md), [추가 발견된 관측값 문제와 실물 적용 차이](DEPLOYMENT_GAPS.md).

## Student Sim2Sim 입력 계약 적용

MuJoCo 기본 실행/export를 Student로 변경했다. measured/reference velocity 분리, joystick yaw-rate 변환, 수동 terrain label, MuJoCo contact, 학습 카메라 위치/자세/전처리를 적용했다. 시간/GRU 동작은 유지한다. RBQ10 Student checkpoint가 없어 실제 Student rollout 성능은 미검증이며 Teacher를 대신 사용하지 않는다. [입력 계약과 검증](SIM2SIM.md).
