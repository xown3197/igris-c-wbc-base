# WBC_base

IGRIS-C의 제조사 공개 torso-root Isaac plant 위에서 새로 학습하는 15축
하지/허리 WBC 기준선이다. 기존 pelvis-root USD, 기존 정책 가중치, 기존
observation normalizer를 가져오지 않는다.

다른 정책/학습기에서 재사용할 경로, task ID, 최소 통합 코드는
[`ENVIRONMENT.md`](ENVIRONMENT.md)에 정리되어 있다. 복사 가능한 기계 판독 계약은
`environment_manifest.json`이다.

## 현재 범위

- 기준 upstream: `robrosinc/robros_lab_public@b72d87190ee6893cf6f3db8dc417d1750518040a`
- plant: upstream `IGRIS_C_LOCOMOTION_CFG` 그대로
- floating root: torso인 `base_link`
- pelvis/heading anchor: `Link_Waist_Pitch`
- policy action: 다리 12축 + 허리 3축, 총 15축
- 상지 8축: policy action을 소비하지 않고 공식 default pose를 hold
- actor observation: 58값 frame 2개, 총 116값
- physics/policy 주기: 5 ms / 20 ms
- PPO actor/critic/optimizer: scratch

현재 7D command는 `[vx, vy, wz, height, roll, pitch, yaw]`이다. 첫 기준선에서는
앞의 세 속도만 변하고 body target은 `[0.94, 0, 0, 0]`으로 고정한다. 이후 기존
body-pose curriculum을 같은 계약에 추가할 수 있다.

## 의도적으로 가져오지 않는 것

- `igris_c_corrected.unlocked.usd` 및 1 kg dummy root
- pelvis에서 torso를 합성하는 관측 변환
- vendor foot collision override
- 고정 5 ms custom target delay
- 기존 BC/PPO checkpoint와 running statistics

공식 plant의 `DelayedPDActuatorCfg(min_delay=0, max_delay=2)`를 사용하므로 별도
target delay를 중복 적용하지 않는다.

## MuJoCo 배치 시작 계약

과거 평가에서 사용한 최대 15회 RESET은 제조사 공개 Isaac 학습 로직이 아니라,
정책 투입 전에 유효한 기립 상태를 고르는 외부 readiness gate였다. 따라서 학습
reset에는 이 선택 절차를 넣지 않는다. 배치 시에는 fresh state와 제어 mode를 확인하고
측정 `q0`를 mode 전환 동안만 hold한다.

정책이 활성화된 뒤에는 IsaacLab과 동일하게 첫 fresh 관측을 2개 history slot에
복제한다. 과거 MuJoCo 코드처럼 history를 0으로 시작하지 않는다. 또한 측정된 기립
`q0`는 action offset이 아니다. action target은 항상 제조사 공개 default pose에
`0.25 * action`을 더한다. 기계 판독 가능한 전체 조건은 `deployment_contract.json`에
고정했다.

## 로컬 계약 검사

```bash
cd WBC_base
PYTHONPATH=src pytest -q
python scripts/verify_upstream.py /path/to/robros_lab_public
```

두 번째 명령은 upstream Git commit과 학습에 사용하는 allowlist 파일의 SHA-256만
검사한다. 저장소의 `public/` 디렉터리는 학습 입력으로 사용하지 않는다.

## Isaac 통합

Isaac Sim을 시작한 다음 `import wbc_base.tasks`가 실행되어야 Gym task가 등록된다.
공식 `train.py`를 복사하거나 수정하지 않는다. `run_verified_upstream.py`가 잠긴
SHA-256을 확인한 뒤 메모리에서만 `wbc_base.tasks` import를 추가한다. import 위치는
upstream AppLauncher가 시작된 뒤다.

제조사 공개 `mdp/actions.py`는 고정된 IsaacLab `f4aa17f`에서 re-export되지 않는
`isaaclab.envs.mdp.actions.ActionTermCfg`를 import한다. `wbc_base.compat`는 task import
직전에 이를 원래 정의인 `isaaclab.managers.ActionTermCfg`로 alias한다. 이 shim은
plant, 관측, reward 또는 action 계산을 바꾸지 않는다.

```python
import robros_lab_public.tasks
import wbc_base.tasks
```

등록 task:

- `WBC-Base-Flat-IGRISC-v0`: 공식 domain randomization을 유지한 scratch 학습
- `WBC-Base-Flat-IGRISC-Nominal-v0`: 64-env randomization-light plant/학습 smoke
- `WBC-Base-Flat-IGRISC-Play-v0`: 1-env zero-command 확인
- `WBC-Base-Flat-IGRISC-CustomHand-v0`: 교정된 고정 커스텀 손 외형을 쓰는 공식 randomization task
- `WBC-Base-Flat-IGRISC-CustomHand-Nominal-v0`: 같은 손 에셋의 64-env nominal 학습/smoke
- `WBC-Base-Flat-IGRISC-CustomHand-Play-v0`: 같은 손 에셋의 1-env 배치 확인

커스텀 손 task는 제조사 torso-root, 23개 가동 관절, 정책 I/O를 유지한다. 손
CAD 하위 관절은 고정되어 있으며 현재 질량과 충돌은 제조사 `Left_Hand`와
`Right_Hand` surrogate를 유지한다. 따라서 이 task는 손 외형 및 장착 배치가 포함된
WBC 학습용이고, 실제 커스텀 손의 정밀 물리 모델이라고 해석하지 않는다.

## 커스텀 손 원본 및 no-camera 결정 (2026-09-13)

손 형상의 기준 저장소는
`sejong-rcv/Project_Humanoid/igris/igris_c_description`이다. 확인한 GitHub
`main`은 `fffe0ba8f145882d21bdbf38d26f22965b9b837d`이고, 손 포함 URDF
`igris_c_rcv_hand_pelvis.urdf`의 SHA-256은
`f06f037e61a756b528bae73daf57f65da4462b1e525c33d511f4995b30bede6d`이다.

현재 WBC asset은 PyRoki에 포함되어 있던 이전 snapshot
`20ea37be83ed30bf02583a1720fe7ac2f616aae048d2e72088b365770a97696f`에서
만들었다. 두 snapshot의 공통 손 link 110개, joint 110개 및 STL 34개는 동일하며,
최신 원본에만 wrist-camera link 4개, fixed joint 4개, STL 4개가 추가되어 있다.
root mount 변환과 8개 손 관절 정의는 바뀌지 않았다. 카메라 link에는 collision이
없고 URDF상 추가 inertial 합계는 한쪽 약 0.587 g이다.

따라서 현재 보행 baseline은 의도적으로 **동일한 손 본체 + wrist-camera 형상 없음**
lineage로 유지한다. 카메라가 없는 결과를 최신 손 전체 형상 결과로 부르지는 않는다.
현재 변환기는 손 관절을 fixed로 만들고 제조사 손 collision surrogate를 유지하므로,
원본의 정밀 손 동역학 또는 접촉 모델과도 구분한다. 최신 카메라 형상을 추가할 때는
기존 asset/task를 덮어쓰지 않고 versioned asset과 새 task ID로 분기한다.

실제 GPU 학습 전에는 upstream 검증, 1-env no-policy settle, 64-env 32-update smoke를
순서대로 수행한다. smoke 성공은 걷기 성공이나 sim-to-real 검증을 뜻하지 않는다.
Nominal 프로파일도 공식 actuator의 `0~2` physics-step delay 샘플링은 유지하므로
완전 결정론적 plant라고 부르지 않는다.

실행 예시:

```bash
# 1 env, 10 s, command/action zero settle
python scripts/zero_action_settle.py \
  --upstream-root /path/to/robros_lab_public \
  --output evidence/zero_action_settle.json \
  --headless --device cuda:0

# 64 env x 32 PPO updates, scratch nominal smoke
python scripts/run_verified_upstream.py train \
  --upstream-root /path/to/robros_lab_public \
  --isaaclab-root /path/to/IsaacLab \
  --output-root evidence/nominal_smoke \
  --task WBC-Base-Flat-IGRISC-Nominal-v0 \
  --num_envs 64 --max_iterations 32 --seed 42 \
  --headless --device cuda:0
```

WBC zero-action 종료가 배치 오류인지 확인할 때는 동일 스크립트에
`--task Isaac-Velocity-Phase-Flat-IGRISC_Play`를 주어 제조사 task를 A/B control로
실행한다. 둘 다 feedback policy가 없는 open-loop 검사이므로 장시간 기립 성공 자체를
학습 전제조건으로 해석하지 않는다.

## 아직 옮기지 않은 방법론

현재 버전은 plant와 policy I/O 경계를 먼저 고정한다. 아래 항목은 별도 변화로
추가하여 원인을 섞지 않는다.

- 5개 이산 명령의 `20 s stand -> 20 s move -> 10 s stand` scheduler
- 시작 heading 유지 및 cross-velocity 진단 reward
- WalkV5 measured-q 목표 완화와 PJS/GLOBAL HARD projector
- 상지 자세, 손목 하중 및 FALCON-inspired curriculum

## 2026-09-13 A100 smoke 결과

정확한 제조사 dependency commit과 Isaac Sim 5.1에서 1-env A/B 및 64-env x
32-update nominal smoke를 완료했다. zero-action 최초 종료는 WBC 78 step, 제조사
phase task 79 step으로 사실상 같았다. 32-update 학습은 exit 0, 49,152 transition,
checkpoint 전체 tensor finite로 runtime 계약을 통과했다. 이 결과는 보행 품질이나
sim-to-real 성능을 입증하지 않는다. 상세 수치는 `evidence/a100_20260913/summary.json`에
기록했다.

같은 seed/분모로 별도의 fresh scratch 500-update nominal run도 완료했다. 최종
rolling mean episode length는 142.11 step(약 2.84 s), mean reward는 4.19로
늘었지만 timeout 종료는 0이고 base-height 종료는 0.96875였다. 따라서 이는
학습 실행 기준선은 통과했지만 기립 성공은 아니다. `model_499.pt`의 76개 tensor
leaf, 1,325,557개 값은 모두 finite였다. 32-update smoke와 500-update run은 서로
다른 fresh lineage이며 checkpoint 번호를 이어서 해석하지 않는다. 상세 수치와
SHA-256은 `evidence/a100_20260913/nominal_500_summary.json`에 기록했다.

교정된 no-camera custom-hand asset으로도 별도의 fresh scratch 500-update nominal
run을 완료했다. 최종 rolling mean episode length는 107.01 step(약 2.14 s), mean
reward는 2.96이었고, timeout 종료는 0이며 base-height 종료는 0.984375였다. 따라서
reward와 생존 시간은 증가했지만 기립 성공은 아니다. 최종 `model_499.pt`의 SHA-256은
`399dffcd15557414dd689ccd26130152936a9102063ebfd7848e1a4c9b6f8369`이고 모든
checkpoint tensor는 finite였다. 이 결과와 손 없는 nominal run의 절대 reward 차이는
서로 다른 fresh lineage이므로 손 또는 카메라의 인과 효과로 해석하지 않는다.
