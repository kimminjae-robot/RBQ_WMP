# 최신 Student MuJoCo 키보드 조작

```bash
cd ~/Desktop/coop_ws/rbq10_vision_parkour
./run.sh sim2sim-latest --course isaac_flat --difficulty 0 --terrain-label flat
```

가장 최근 저장된 Phase4 Student checkpoint를 선택해 export한 뒤 MuJoCo GUI를 연다. Teacher/Go2를 대신 사용하지 않는다. 학습 중인 메모리가 아니라 완전히 저장된 snapshot을 사용하며 시작 후 자동으로 모델을 교체하지 않는다. 기존 export 경로를 직접 쓰려면 `./run.sh sim2sim --policy-dir PATH --keyboard ...`도 가능하다.

| 키 | 동작 |
| --- | --- |
| W / ↑ | 전진속도 명령 +0.1m/s |
| S / ↓ | 전진속도 명령 −0.1m/s |
| A / ← | 좌회전속도 명령 +0.2rad/s |
| D / → | 우회전속도 명령 −0.2rad/s |
| Space | 전진/회전 명령을 0으로 |
| R | 초기 위치, action/history/Depth buffer/GRU state를 리셋하고 다음 입력 대기 |
| Q / Esc | 종료하고 기록 저장 |

뷰어 창에 포커스를 둔다. 키를 누를 때마다 목표값을 증감하는 방식이며 키를 떼면 0이 되는 방식은 아니다. vx 제한 ±0.8m/s, yaw-rate 제한 ±1rad/s다. 원본 학습은 전진 중심이므로 후진 성능은 보장하지 않는다. 처음과 R 이후에는 W/A/S/D 입력을 기다리며 물리를 진행하지 않는다. 넘어지면 물리를 일시 정지하고 R 또는 종료를 기다린다. Space는 명령을 0으로 만들며 물리를 강제로 정지하는 키가 아니다. 기본 session은 최대 1시간이며 `--seconds`로 바꿀 수 있다.

측정 선속도와 MuJoCo 접촉 입력은 유지한다. 이 조작 변경은 Student 학습 프로세스에 영향을 주지 않는다. 외부 measured velocity를 공급할 경우 `--input-json ... --velocity-source external`을 함께 쓰면 measured 항목을 받고 command는 키보드를 우선한다.

## 영상/사진 저장

```bash
./run.sh sim2sim-latest --course isaac_hurdle --difficulty 0 --terrain-label parkour --video artifacts/manual_test.mp4 --screenshot artifacts/manual_test.png
# GUI 없이 고정 명령 테스트/촬영:
MUJOCO_GL=egl ./run.sh sim2sim-latest --no-keyboard --course isaac_flat --difficulty 0 --terrain-label flat --speed 0.3 --seconds 12 --video artifacts/flat_test.mp4
```

촬영은 실제 MuJoCo state와 Student Depth 추론을 사용한다. MP4는 25fps이며 종료하면 writer를 닫는다. GUI 없는 고정 명령 시험은 fall 시 종료하고 `FALL - STOPPED` 마지막 장면을 2초 유지한다. 이 유지 프레임은 추가 시뮬레이션 시간이 아니다.

## 2026-10-23 실제 테스트

시험 시작 당시 최신 저장 snapshot은 **model_400.pt**였다. 실제 Student export 후 `isaac_flat`, difficulty0, seed42, 수동 flat label, vx=0.3m/s, yaw-rate=0으로 시험했다.

- 약 **5.66초에 fall**. 안정적인 Sim2Sim 성공 결과가 아니다.
- 최대 관절 속도 약 84.28rad/s로, 기존에 남겨둔 PhysX/MuJoCo 속도 제한 차이도 여전히 존재한다. 이 시험만으로 낙상 원인을 한 가지로 단정하지 않는다.
- 실제 GUI 키보드 session에서 속도/회전 명령 변화, 6번의 시작/리셋 episode, 4번의 fall, GUI 종료를 기록했다. 고정 명령 영상과 별도 session이다.
- Student 학습은 계속 실행했다. 이 snapshot은 아직 초기 증류 모델이며 추가 학습과 Isaac/MuJoCo 관측/동역학 비교가 필요하다.

[실제 영상](../artifacts/student_model_400_mujoco_flat.mp4), [GIF](../artifacts/student_model_400_mujoco_flat.gif), [최종 장면](../artifacts/student_model_400_mujoco_flat.png), [결과 요약](../evaluations/student_visual/summary.json).

## 최신 저장 snapshot 추가 시험

키보드 확인 후 새로 저장된 `model_500.pt` 를 자동 선택해 재시험했다.

동일한 `isaac_flat`, difficulty0, vx0.3m/s 조건에서 **7.66초에 fall**했다. 두 번의 단일 시행 차이는 통계적 개선 판정이 아니다. 최종 제공 영상은 이 최신 snapshot이며 학습은 유지했다.

[최신 영상](../artifacts/student_latest_mujoco_flat.mp4), [최신 GIF](../artifacts/student_latest_mujoco_flat.gif), [결과](../evaluations/student_visual/latest_summary.json).

## 중간 난이도 징검돌 바로 실행

```bash
~/Desktop/coop_ws/rbq10_vision_parkour/play_stones_medium.sh
```

원본 `parkour` 징검 플랫폼을 MuJoCo 높이장으로 변환한 `isaac_platforms`, difficulty0.5, seed42를 사용한다. 최신 저장 Student를 자동 선택하며 키보드 입력 대기로 시작한다. W를 3번 누르면 전진속도 명령 0.3m/s다. A/D로 좌우 방향을 맞추고 Space로 명령 0, R로 리셋한다. 실제 로봇/Isaac 대비 MuJoCo 전이는 아직 검증 중이므로 안정적인 코스 완주를 보장하는 명령은 아니다.
