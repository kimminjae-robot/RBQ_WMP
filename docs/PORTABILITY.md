# 다른 PC에서 설치·평가·학습 재개

## 런타임 경계

먼저 Isaac Lab v2.3.2와 그 설치에 사용한 Python 환경을 준비한다. 검증 조합은 Sim 5.1.0, Python 3.11, Torch 2.7.0+cu128, torchvision 0.22이다. Isaac Lab package metadata는 `0.54.2`로 표시된다. Simulator 자체는 이 저장소에서 배포하지 않는다.

`tools/setup_env.py --isaac-python ...`는 기존 Python의 site-packages 위치를 조회하고 새 프로젝트 `.venv`에서 읽는다. RSL-RL·terrain·녹화 의존성은 새 venv에만 설치하며 기존 환경의 패키지를 uninstall/upgrade하지 않는다. GPU driver나 Ubuntu 설정도 변경하지 않는다.

```bash
python3 tools/setup_env.py --isaac-python /path/to/isaac/venv/bin/python --check-only
```

다른 조합은 자동 허용하지 않는다. `--allow-runtime-mismatch`는 미검증 조합을 명시적으로 허용할 때만 사용한다. 기존 `.venv`도 덮어쓰지 않는다. 다른 설치 위치는 `--venv /새/경로`와 `RBQ10_PYTHON=/새/경로/bin/python ./run.sh ...`로 지정한다.

설치·런처·URDF 경로는 프로젝트 위치에서 계산한다. 원래 PC의 env.yaml/agent.yaml은 참고 snapshot이고 실제 config 로딩에 사용하지 않는다. 새 PC에서 기존 `/home/user/...` 경로를 재현할 필요가 없다.

## 재개 계약

`./run.sh import-teacher`는 포함된 `model_3674.pt`의 SHA-256·iteration·RBQ10 URDF hash·관절 순서·PD·제어 주기·auxiliary optimizer 상태를 확인한다. 기존 runner의 `logs/rsl_rl/rbq10_extreme_parkour/imported_teacher_3674/`로 복사하며 다른 파일을 덮어쓰지 않는다.

`teacher-job start --resume <해당 경로> --iterations N`은 **추가 N회**를 학습한다. 완료한 3,675회 이후 총 5,000회는 N=1,325, 총 50,000회는 N=46,325다. PPO/estimator/history optimizer, adaptive learning rate, RMA counter와 transition 누계를 복원한다.

Simulator 물리 상태·episode·terrain curriculum 분포·전체 RNG state는 저장하지 않는다. Parameter/optimizer 연속성을 유지하지만 같은 trajectory를 bitwise 재현하지는 않는다. 새 GPU·driver·PhysX에서도 평가 도구로 다시 확인해야 한다.

## 저장 공간과 녹화

기존 PC는 2026-10-22 20:25 KST에 디스크 2GiB 미만을 감지해 저장 후 정상 종료했다. 다음 날 사용자가 공간을 확보한 뒤 같은 checkpoint에서 재개했다. 모델은 약 10MB씩 원본 주기 100 iteration마다 저장된다. 로그와 Simulator cache도 공간을 사용하므로 새 PC에서는 디스크 여유 20GB 이상을 권장한다.

16GB GPU에서는 대규모 학습과 RTX 녹화를 동시에 하지 않는다. 안전 중단 → 평가/녹화 → 해당 checkpoint 재개 순서로 진행한다. Status/TensorBoard 확인은 학습 중에도 가능하다.

## 선택적인 Phase 1 Go2 모델

새 설치에 `--download-go2`를 추가하거나 이미 설치한 환경에서 다운로드 함수만 실행한다.

```bash
.venv/bin/python - <<'PY'
import json
from tools.setup_env import ROOT, download
for item in json.loads((ROOT / 'checkpoints/manifest.json').read_text()):
    download(item['source_url'], ROOT / 'checkpoints' / item['file'], item['sha256'])
PY
```

이 모델은 Go2 재현용이며 RBQ10 정책이 아니다. 원본 Drive 배포본과 동일성은 미확인이다. 출처 fork commit과 hash는 manifest에 고정했다.

## 수행한 이식 검증

새 폴더 `/tmp/rbq10_portable_rehearsal`에서 설치 스크립트로 원본 checkout/patch·독립 venv를 구성하고, 모델 import와 level 9의 10개 episode를 headless로 실행했다. 모두 정상 종료했다. 이 검사는 **같은 PC의 새 경로·새 venv**에서 수행했으며 별도 물리 PC에서 검증했다는 의미는 아니다. 기존 기본 환경의 RSL-RL 3.1.2도 변경되지 않았다.
