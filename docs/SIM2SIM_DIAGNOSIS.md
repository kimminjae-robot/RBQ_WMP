# Isaac Student 촬영 및 MuJoCo 낙상 진단 (2026-10-23)

실제 RBQ10 Student model_764.pt (765 updates), SHA256
503740d55cce8681538832b14848b64e469f7b5dd047a4c8d5bd87acd6e44834.
Teacher/더미 정책 대체 없이 원본 Student actor와 Depth CNN/GRU를 사용했다.

## Isaac Sim 영상

[GIF](../artifacts/student_latest_isaac.gif), [MP4](../artifacts/student_latest_isaac.mp4).
중간 level4/9 징검돌을 따라 촬영한 실제 Isaac Sim 영상 17.16초다.
동일 프로세스에서 5종 지형 각2환경, 1episode씩 10/10 성공했다.
고정 seed42, vx0.6, measured velocity, Student action delay1, random pushes/mass/COM/camera 없음.
작은 nominal 평가이며 전체 난이도/실물 성능을 뜻하지 않는다.

## 확인한 구현 오류와 수정

MuJoCo mj_objectVelocity(... mjOBJ_BODY, flg_local=1)는 URDF link 축이 아니라
주관성축(ximat) 기준의 COM 속도를 반환한다. 기존 bridge는 이를 학습 입력인
trunk link 축(xmat) 기준으로 잘못 사용했다. 각속도와 측정 선속도가 모두 영향받았다.

run_teacher.py의 공유 body_velocity()에서 세계 좌표 COM 속도를 얻고 xmat.T로 회전하도록 수정했다.
run_student.py의 측정 선속도도 같은 함수를 사용한다. COM 위치는 유지하며
단순히 mjOBJ_XBODY로 바꾸어 link 원점 속도를 넣는 수정은 하지 않았다.
원본 MuJoCo 구현: https://github.com/google-deepmind/mujoco/blob/3.3.0/src/engine/engine_support.c

실제 Isaac 상태 200개를 MuJoCo에 주입한 결과:

- 몸체 위치 최대 오차 3.85e-6m, 관절 이름/제한 일치.
- 수정 전 각속도 최대 오차 1.968rad/s, 선속도 1.415m/s.
- 수정 후 둘 다 6e-16 이하 (동일 상태에서 좌표 변환 비교).
- 이 검증은 접촉/동역학이 동일함을 뜻하지 않는다.

## 동일 checkpoint/조건 수정 전후

MuJoCo3.3.0, isaac_flat, difficulty0.5, seed42, flat label, vx0.6, yaw-rate0:

| 상태 | 결과 | 최대 관절속도 |
|---|---|---|
| 수정 전 | 2.56초 낙상 | 56.89rad/s |
| 수정 후 | 15초 무낙상, 전진11.82m, 횡변위0.014m | 12.75rad/s |

단일 시행이며 완전한 Sim2Sim 검증은 아니다. 수정 후 중간 징검돌
isaac_platforms difficulty0.5, vx0.6, yaw-rate0 시험은 **5.04초에 여전히 낙상**했다.

## 아직 남아 있는 차이

1. Isaac 평가의 heading 입력은 원본 terrain goal 방향이다. MuJoCo는 joystick
   yaw-rate를 0.5초 heading error로 변환한다. yaw-rate0이면 두 heading error가
   계속0이어서, 징검돌 중심/이탈을 자동 보정하지 않는다. Depth latent만 바꾼
   Student도 원본의 이 heading 보조입력을 사용한다. 따라서 Isaac goal-following과
   MuJoCo 무조작 직진은 동일 임무가 아니다. 키보드 A/D로 조종 가능하지만 실험에서
   joystick-heading 인터페이스 검증이 필요하다.
2. MuJoCo hfield는 가파른 경계가 삼각 경사면으로 연결된다. Isaac의
   slope-threshold 수직경계 보정/단순화 mesh와 접촉면이 동일하지 않다.
3. PhysX implicit PD와 MuJoCo position servo, 접촉 solver/마찰/접촉 오프셋,
   self-collision 필터 차이가 남아 있다. 토크/이득/armature/dt는 설정상 대응시켰다.
4. PhysX 50rad/s joint velocity limit은 MuJoCo에서 아직 강제되지 않는다.
   수정 후 평지/징검돌 단일시험은 각각12.75/39.11rad/s로 제한 아래였다.
   과거 속도폭주를 이 차이 하나로 설명해서는 안 된다.
5. Depth extrinsics는 제조사 front Depth와 일치하지만, 실제 센서의 광학 보정은
   별도다. 두 시뮬레이터 간 같은 map/state의 raw Depth pixel 비교는 아직 안 했다.

## 재현

```bash
./run.sh compare-policies --teacher checkpoints/rbq10_teacher/model_3674.pt \
 --student /ABS/PATH/model_764.pt --output evaluations/student_isaac_visual/results.json \
 --levels 4 --roles student --num_envs 10 --episodes 1 \
 --video artifacts/student_latest_isaac.mp4 --trace evaluations/student_isaac_visual/state_trace.npz --headless
.venv-sim2sim/bin/python tools/check_sim2sim_state_parity.py \
 --trace evaluations/student_isaac_visual/state_trace.npz \
 --manifest outputs/sim2sim/diagnostic764/manifest.json \
 --output evaluations/student_isaac_visual/state_parity.json
./run.sh sim2sim-latest --course isaac_flat --terrain-label flat
./play_stones_medium.sh
```

영상/trace를 위해 학습을 안전 저장한 뒤 model764 optimizer/counter에서
남은235 update를 재개했다. 총 목표1000은 유지한다.
원본 네트워크/학습 loss/정책 가중치는 이번 수정으로 바뀌지 않는다.

## 후속 수정: 자동 시연

[목표점 자동 시연 및 최종 결과](AUTONOMOUS_DEMO.md)를 추가했다.
raw terrain wrapper/목표 좌표, 경계 변환, 접촉 응답을 수정했고
model941/seed42 중간 징검돌·허들에서 각각8/8 목표 완주를 확인했다.
다른 시드와 Gap에서는 실패가 남아 있다.
