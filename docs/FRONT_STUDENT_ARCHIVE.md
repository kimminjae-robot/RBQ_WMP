# 전방 Depth Student 저장·중단 기록 — 2026-10-23

사용자 요청에 따라 기존 전방 전용 학습을 **13:09:25 KST에 중단 요청**했다.
현재 증류 update를 완료한 뒤 **model1549, 누적1550 update**를 저장했고,
**13:09:32 KST, 종료 코드0**으로 학습 프로세스와 관리 프로세스가 종료됐다.
저장 파일을 다시 로드해 다음 iteration1550과 depth Adam 상태가 보존됐음을 확인했다.
기존 로그·과거 checkpoint는 삭제하지 않았다.

GitHub에는 전체 파일을 압축한 25개 조각과 manifest를 포함한다. 원본 파일은112,833,075바이트,
SHA256은 `708a3591052e5696c92c82cfd50c387ff1be86416d5520fd43b33356c3b565bd`다.
압축본은102,506,694바이트다. 조각별·압축본·복원본 SHA256을 모두 검증하며,
별도 임시 폴더에서 원본 checkpoint가 무손실 복원됨을 실제로 확인했다.

```bash
python tools/restore_student_checkpoint.py
# checkpoints/rbq10_student_front/model_1549.pt
```

Student actor, depth CNN/GRU, Teacher/estimator, optimizer, 완료 counter가 들어 있다.
Teacher로 대체한 파일이나 inference 전용 가중치가 아니다. 이미 다른 파일이 있으면
복원 도구가 덮어쓰지 않는다. 지형 curriculum·시뮬레이터 episode·GRU runtime state는
원본 재개 방식에 따라 초기화되므로 trajectory를 정확히 이어가는 checkpoint는 아니다.

| 보존 정보 | 파일 |
|---|---|
| 전체 checkpoint 복원 정보·SHA256 | `checkpoints/rbq10_student_front/manifest.json` |
| gzip 조각25개 | `checkpoints/rbq10_student_front/parts/` |
| 실제 작업 종료·process identity | `checkpoints/rbq10_student_front/training_stop.json` |
| 마지막 완료 update의 loss와 지형별 지표 | `checkpoints/rbq10_student_front/latest.json` |
| 실행한 env/agent 설정 | `checkpoints/rbq10_student_front/params/*.yaml` |

마지막 action loss는0.52281, yaw loss0.16739, 합계0.69020이었다. 이는 증류 손실이며
완주율 또는 Teacher 성능 달성률이 아니다. training curriculum에서 수집한 지형별 통계도
고정 난이도의 보류 평가와 구분해야 한다.

현재 진행하는 실험은 [5뷰 Student 분기](https://github.com/kimminjae-robot/RBQ_WMP/tree/experiment/depth5-student-20261023)다.
기존 전방 학습을 자동 재개하지 않는다. 새 분기와 기존 태스크/실험/체크포인트를 구분한다.

기존 front 모델을 명시적으로 재생/전이 검증하려면:

```bash
./run.sh export-sim2sim --checkpoint checkpoints/rbq10_student_front/model_1549.pt \
  --output outputs/sim2sim/front1550
```

이 문서는 저장과 복원 기록이다. 추가 학습2,000회 완료 또는 새 카메라 성능 향상을 의미하지 않는다.
