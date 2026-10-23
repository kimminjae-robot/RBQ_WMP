# RBQ10 Student 자동 지형 시연 (2026-10-23)

> 이 문서는 이전 **전방 전용 model941** 시연 기록이다. 현재 5뷰 model400 시연과
> model700 재개본은 [5뷰 빠른 시작](DEPTH5_QUICKSTART.md)을 사용한다.

```bash
cd ~/Desktop/coop_ws/rbq10_vision_parkour
./demo_autonomous.sh
# 동일 명령: ./play_stones_medium.sh
# 허들:
./demo_autonomous.sh --course isaac_hurdle
```

창이 열리면 키보드 입력 없이 출발한다. 목표점 8개를 따라가고, 성공/낙상 시
3초 동안 마지막 물리 상태를 보여준 뒤 종료한다. 이 마지막 정지 시간은 시뮬레이션
시간에 포함하지 않는다. 낙상 후 자동 리셋으로 실패를 숨기지 않는다.

기본값은 **검증한 model941 Student, seed42, difficulty0.5, vx0.6**이다.
로컬 `checkpoints/rbq10_student/model_941.pt`를 사용하며 실행 때마다 원본 Student actor와
Depth CNN/GRU를 export한다. 이후 학습으로 최신 checkpoint가 달라져도 기본 시연 조건은
바뀌지 않는다. 최신 정책을 시험하려면 아래 별도 명령을 사용한다.
이번 배포에서 model941 원본 가중치를 새로 추가하지 않았다. 보유한 해당 전방 checkpoint가
없는 새 PC에서는 동봉한 model400 5뷰 Export를 사용한다.

```bash
./run.sh sim2sim-latest --autonomous --viewer --course isaac_platforms --difficulty .5
# GUI 없이 기본 시연과 정확히 같은 조건:
MUJOCO_GL=egl ./demo_autonomous.sh --headless
# 영상:
MUJOCO_GL=egl ./demo_autonomous.sh --headless --video artifacts/auto_test.mp4
```

새 맵/정책의 완주를 보장하는 명령은 아니다. `outputs/sim2sim/autonomous_demo.json`에
종료 사유, 목표점 진행, 낙상, 상태 이력이 기록된다. 자동 모드는 완주 때만 exit0이며 실패/미완료는 exit2다. 기본 시연은 카메라/조이스틱 외부
입력이나 Teacher 관절 action 없이 동작한다.

## 실제 검증 결과

다음 결과는 모두 입력/리셋 없이 실제 Student 추론과 MuJoCo 물리를 실행한 결과다.

| Student | 지형 | 난이도 | Seed | 목표점 | 결과 |
|---|---|---|---|---|---|
| model941 | 징검돌 | 0.5 | 42 | 8/8 | 9.52초 완주 |
| model941 | 허들 | 0.5 | 42 | 8/8 | 17.10초 완주 |
| model941 | 징검돌 | 0.5 | 43 | 6/8 | 18.62초 낙상 |
| model941 | 징검돌 | 0.5 | 44 | 6/8 | 9.62초 낙상 |
| model941 | Gap | 0.5 | 42 | 7/8 | 13.10초 낙상 |

기본 징검돌/허들 조건의 완주는 최종 코드에서 다시 확인했다.
선정된 시연 맵에서의 성공이며 임의 지형 전체의 안정성을 의미하지 않는다.
새 시드/Gap의 안정화와 50rad/s 모터 속도 제한, live Depth parity 검증이 남아 있다.

[자동 시연 영상](../artifacts/student_autonomous_mujoco_stones.mp4),
[GIF](../artifacts/student_autonomous_mujoco_stones.gif),
[징검돌 결과](../evaluations/autonomous/validated_stones.json),
[허들 결과](../evaluations/autonomous/validated_hurdle.json).

## 구현 변경과 근거

- `goal_follower.py`: 원본 terrain generator가 생성한 현재/다음 목표점의 yaw 오차를
  Student 관측의 6:8에 넣는다. 0.2m 도달 범위, >0.1초 누적 도달 시간, 목표 갱신 순서,
  5 policy-step heading hold를 원본 `ParkourEvent`와 맞췄다. 원본 Torch 메서드를
  실제 실행하는 비교에서 602스텝/8목표점의 인덱스와 heading이 일치했다.
  이는 알려진 맵 목표점 기반 시연 모드이며 실물의 순수 비전 내비게이션 검증은 아니다.
  Depth를 관절 action으로 바꾸는 Student와 가중치/추론 구조는 유지했다.
- `terrains.py`: 기존 raw terrain 함수 재사용에 더해, 생략돼 있던 원본
  `parkour_field_to_mesh`의 1픽셀 외곽과 inner/full 크기 기반 목표점 보정을 반영했다.
  4종 지형의 원본 wrapper 실행 결과와 목표점 오차 <=4.5e-16m를 확인했다.
  world Y 원점은 -2m, 전체 크기는 16x4m다. 기존 atlas는 이전 raw profile의 그림이다.
- 원본 `convert_height_field_to_mesh` 함수 자체를 실행해 경계 이동을 적용하고
  표면을 2cm hfield로 변환한다. MuJoCo hfield는 수직 벽을 표현할 수 없어
  남는 전이 폭은 최대 한 셀(2cm)이다. 이전 10cm 경사면과 다르다. 장애물 높이/폭과
  생성 난이도는 임의로 낮추지 않았다. 원본 Student mesh simplification까지 동일하지는 않다.
- `contact-profile=firm`: MuJoCo 기본 solref=(.02,1), solimp=(.9,.95,.001,.5,2) 대신
  solref=(.01,1), solimp=(.95,.99,.001,.5,2)를 사용한다. 접촉 시간상수와 impedance를
  더 단단하게 조정해 해당 코스의 전이를 개선한 경험적 설정이다. PhysX와 물리가
  같아졌다는 의미가 아니다. PD/토크/질량/armature는 변경하지 않았다.
  [MuJoCo solver 설명](https://mujoco.readthedocs.io/en/3.3.0/modeling.html#solver-parameters).
  기존 접촉 비교는 `--contact-profile default`, 이전 넓은 hfield 경계는
  `--terrain-edges hfield`로 시험할 수 있다.
- MuJoCo 적분 직후 `mj_forward`로 pose/contact/velocity를 갱신해 현재 qpos/qvel과
  관측 시점을 맞췄다. 지난 진단의 COM 주관성축→trunk link 축 수정도 유지한다.
- 목표점 구체는 충돌/Depth ray에 영향을 주지 않는 site다. 지나간 목표는 녹색,
  현재 목표는 청록색, 미래 목표는 주황색으로 표시한다.

학습 알고리즘, reward, curriculum, Teacher/Student loss, 정책 가중치를 시연용으로
바꾸지 않았다. 기존 키보드 모드는 아래와 같이 사용할 수 있다.

```bash
./run.sh sim2sim-latest --keyboard --course isaac_platforms --difficulty .5
```

## 점프 GIF

[최신 Student model999의 Isaac Sim GIF](../artifacts/student_isaac_jump.gif)는
level9/9 허들, reference vx0.8, seed42, env2의 약4.8~7.6초 구간을 1.5배 느리게
재생한다. 영상의 시간 표시와 foot contact 측정은 실제 시뮬레이션 값이며
AIRBORNE는 4개 발 모두 접촉력이 2N 이하일 때만 표시한다.
최고난이도 해당 전체 episode는 목표점6에서 낙상했다. GIF는 도약 장면이며
전체 코스 완주 증거가 아니다.

[원본 전체 MP4](../artifacts/student_isaac_high_hurdle.mp4),
[프레임별 접촉](../artifacts/student_isaac_high_hurdle.states.json),
[평가 결과](../evaluations/student_jump/high_hurdle_results.json).

Student 학습은 총1,000회 update를 완료했고 model999를 저장했다.
MuJoCo 기본 시연에는 반복 검증한 model941를 사용한다.
