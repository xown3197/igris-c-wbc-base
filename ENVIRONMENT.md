# WBC_base environment reuse guide

이 문서는 현재 IGRIS-C torso-root Isaac Lab 환경을 PPO 이외의 정책에서도
재사용하기 위한 기준점이다. 환경 경계는 Gym task이며, RSL-RL 설정은 기본 학습기
메타데이터일 뿐 환경을 RSL-RL에 종속시키지 않는다.

## 기준 경로

- 로컬 bundle: `/home/tjkim/Documents/ChatGPT/IGRIS-C/WBC_base`
- 로컬 Python package: `/home/tjkim/Documents/ChatGPT/IGRIS-C/WBC_base/src`
- 로컬 custom-hand USD:
  `/home/tjkim/Documents/ChatGPT/IGRIS-C/WBC_base/assets/igris_c_custom_hand_torso/usd/igris_c_custom_hand_torso.usd`
- A100 runtime bundle: `/home/tjkim2/WBC/wbc_base_20260913/runtime/WBC_base`
- A100 manufacturer upstream:
  `/home/tjkim2/WBC/wbc_base_20260913/runtime/robros_lab_public`
- A100 IsaacLab: `/home/tjkim2/WBC/wbc_base_20260913/runtime/IsaacLab_exact`
- A100 RSL-RL 5.4.2 overlay:
  `/home/tjkim2/WBC/wbc_base_20260913/runtime/rsl_overlay`

`WBC_base` 디렉터리 전체를 복사해야 한다. USD wrapper만 떼어내면 상대경로로
참조하는 `usd/configuration/`과 mesh가 빠질 수 있다.

## 사용할 task

- `WBC-Base-Flat-IGRISC-CustomHand-v0`: 공식 domain randomization을 유지하는 본 학습
- `WBC-Base-Flat-IGRISC-CustomHand-Nominal-v0`: randomization-light 64-env 실험/smoke
- `WBC-Base-Flat-IGRISC-CustomHand-Play-v0`: 1-env, zero-command 배치 확인
- 손 교체 전 제조사 외형이 필요하면 위 이름에서 `CustomHand-`를 뺀 세 task 사용

다른 알고리즘이 현재 정책과 같은 입력/출력을 사용하면 task ID만 그대로 소비하면
된다. 고정 계약은 다음과 같다.

- action: 다리 12축 + 허리 3축 = 15
- actor observation: 58값 x 2-frame history = 116
- command: `[vx, vy, wz, height, roll, pitch, yaw]`
- root/IMU 의미: torso `base_link`; pelvis 참조 body는 `Link_Waist_Pitch`
- quaternion: `wxyz`
- physics/policy: 200 Hz / 50 Hz
- target: 제조사 default joint pose + `0.25 * action`
- reset history: 첫 관측을 두 history slot에 복제

관측 순서와 관절 순서는 `src/wbc_base/contract.py`가 유일한 기준이다.

## 다른 학습기에서 여는 최소 순서

Isaac Sim 앱을 먼저 만든 뒤 task를 import한다.

```python
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from wbc_base.compat import install_action_term_cfg_alias

install_action_term_cfg_alias()
import robros_lab_public.tasks  # noqa: F401, E402
import wbc_base.tasks  # noqa: F401, E402

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg

task_id = "WBC-Base-Flat-IGRISC-CustomHand-v0"
env_cfg = parse_env_cfg(task_id, device="cuda:0", num_envs=4096)
env = gym.make(task_id, cfg=env_cfg)
```

필요한 Python 경로는 `WBC_base/src`와 제조사 저장소의
`source/robros_lab_public`이다. 학습기 쪽에서 Gym vector environment의 reset/step
출력을 자신의 rollout 형식으로 연결하면 된다. 등록 정보의
`rsl_rl_cfg_entry_point`는 RSL-RL 기본값이므로 다른 학습기는 무시해도 된다.

## 새 정책의 변경 범위

1. **알고리즘만 변경:** task와 환경 설정은 수정하지 않는다. 새 학습기의 actor가
   116차원 입력과 15차원 출력을 사용하도록 한다.
2. **network/history만 변경:** 환경 관측 116차원은 유지하고 학습기 내부에서
   encoder 또는 recurrent state를 추가한다.
3. **관측이나 action 의미 변경:** `WBCBaseCustomHandFlatEnvCfg`를 상속한 새 cfg와
   새 task ID를 만든다. 기존 task를 덮어쓰지 않아야 실험 분모와 checkpoint lineage가
   섞이지 않는다.
4. **손 접촉/손가락 제어 학습:** 현재 asset을 그대로 사용하면 안 된다. 커스텀 손은
   고정 visual CAD이고 질량/충돌은 제조사 `Left_Hand`/`Right_Hand` surrogate다.

## 복사 후 검사

```bash
cd /path/to/WBC_base
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests
python scripts/verify_environment_bundle.py
python scripts/verify_upstream.py /path/to/robros_lab_public
```

첫 두 명령은 CPU에서 bundle과 정책 계약을 검사한다. 실제 GPU 학습 전에는
`CustomHand-Play-v0` zero-action settle과 짧은 nominal smoke를 별도로 실행한다.
smoke 성공은 보행 품질이나 실로봇 안전 검증을 뜻하지 않는다.

