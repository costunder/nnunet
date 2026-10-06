# Basic CP–v2.22 비교 난수 정정 (2026-09-22)

## 원인과 수정

v2.22의 nnU-Net 경로는 seed=42를 설정했지만, Original Basic CP는 기본 nnU-Net의 비결정적 worker와 seeds=None을 그대로 사용했다. 따라서 기존 상태를 동일한 난수 조건이라고 말할 수 없었다. 데이터 분할과 모델 설정 일치만으로 동일 실험 환경이라고 설명한 것은 잘못이다.

공통 `comparison_randomness.py`를 두 트레이너에 연결했다. 비교 학습 seed 기본값은 기존 v2.22의 42다. 모델 초기화 및 epoch/worker/batch별 CT 선택, foreground oversampling, crop, 기본 영상 증강에 공통 난수 일정을 적용한다. CP는 별도 RNG를 사용한다. CP 내부 draw 수가 달라도 다음 CT/기본 증강의 난수 일정은 밀리지 않는다. CPU worker에서 CUDA RNG를 초기화하지 않는다.

두 경로 모두 ordered MultiThreadedAugmenter를 사용하며, 매 epoch에 자기 loader worker만 재생성하고 train/validation 각각 warm-up 한 batch를 버린다. 같은 epoch를 재개할 때도 동일하다. 두 실행의 seed, worker 수, physical batch, case ID 순서가 같아야 공통 일정이 성립한다. `--seed`는 양쪽 기본 42이며 `--workers`는 양쪽 필수 인자다. 실행 receipt와 로그에 실제 값을 기록한다. 기존 split ID는 바꾸거나 seed=42로 다시 생성하지 않는다.

cuDNN benchmark=False/deterministic=True를 공통 적용했다. Native CPU GaussianBlur의 FFT/일반 연산 시간 측정 선택은 보존했다. 이 때문에 동일 난수의 DEBUG 입력에서도 최대 약 1.8e-7 수치 오차를 관측했고 실제 기본 증강 검사는 rtol=1e-5/atol=1e-6, GT exact로 검사한다. GPU 전체 학습의 bitwise 재현성까지 검증한 것은 아니다. CP로 영상·마스크와 crop 대상이 달라지므로 같은 난수 seed가 같은 입력 영상이나 공간 변환 결과를 뜻하지 않는다.

## 여전히 다른 CP 정책

| 항목 | Original Basic CP online | v2.22 |
|---|---|---|
| donor: 복사할 종양의 출처 | 같은 CT 내 종양 | training CT들의 공통 donor pool |
| 종양 크기 제한 | 없음 | 체적 기준 등가 직경 20mm 이하 |
| CP 시도 확률 | 매 case 방문에 시도; 배치 성공률 100%라는 뜻은 아님 | 0.5 |
| 위치 선택 | 최대 4,000번 무작위 탐색 중 첫 유효 위치 | 128개 후보의 GNN 점수 argmax |
| nnU-Net crop | 증강한 전체 GT를 기준으로 일반 crop | CP 시 붙인 종양 bbox 주변 crop |

따라서 두 방법은 현재 **전체 파이프라인 비교**다. GNN 위치 선택만의 ablation이라고 주장할 수 없다. 그 효과만 분리하려면 donor/크기/확률/후보/crop을 동일하게 두고 선택 점수만 바꾼 별도의 대조군이 필요하다. 이를 원본 Basic CP라고 부르거나 원본을 바꿔서 맞추지 않는다. 이번 수정은 위 정책을 변경하거나 새 대조군을 실행하지 않았다.

로컬 준비 자료의 raw CT/GT 131개 SHA와 outer train105/val26, inner train84/val21 분할은 서로 같다. 기존 split에서 outer seed270869/inner1042를 유지했다. ResEncM, patch128³, physical batch2, 250epochs도 유지했다. 근거: `work/v222_train_20260922/basic_cp_comparison_audit.json`. 이 파일은 **수정 전** 감사이므로 seed 불일치 표시는 보존한다. 과거 서버 Basic CP 점수0.6454의 데이터·split·seed 동등성까지 확인한 것은 아니다.

## 검증과 실행 상태

- DEBUG 난수 검사9개: Python/NumPy/Torch RNG 복원, CP 소비량 개입, 실제 native/online loader loop 및 nnU-Net 기본 transforms, worker/epoch/seed 분리, 2-worker 순서, train/validation epoch 재개, 실제 전체 ResEncM 초기 state_dict 일치.
- Basic CP 원본 CT/GT 동등성 등8개와 v2.2–v2.22 회귀39개 통과. 합계56개. 최초 2-worker 검사는 DEBUG mock의 pickle 문제로 실패했고 직렬화 가능한 fixture로 고친 뒤 통과했다.
- L0/L1/L2 및 원본 Basic CP reference 소스는 보존했다. 기존 준비 manifest는 source hash가 달라져 재준비가 필요하며 과거 manifest/checkpoint의 hash를 새 값으로 위조하지 않는다.
- 전체 GNN/nnU-Net 학습 및 전체 평가는 미실행이다. 새 성능 점수나 production checkpoint는 없다. 전체 데이터의 case/환자 동일성·주석 provenance 계약 문제와 production 자원/처리량 보정은 별도 미완료 항목이다.
- 수정 전 보존: `versions/v2.22/before_comparison_seed_fix_20260922/`; 수정 검증: `versions/v2.22/verification_comparison_seed_20260922/`.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다. 수정 전 소스를 보존했다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 검토했다. native batch2 유지; 2-worker DEBUG 검증. Production 처리량 보정 미완료.
- [x] GPU, CPU, RAM 자원을 확인했다. RTX5070Ti 16GB/CPU16logical/RAM64GiB. 이번 검사는 CPU RNG/loader 검사다.
- [x] OOM이 발생하지 않았으며 모델 축소를 적용하지 않았다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 production에 사용하지 않았다. 합성 fixture는 명시적인 DEBUG다.
- [x] 핵심 모델의 forward/loss/gradient/optimizer 회귀 검사가 통과했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
