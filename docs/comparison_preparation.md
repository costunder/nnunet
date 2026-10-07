# 비교 실험의 CPU 입력 대기 개선

대상은 기존 `selected`, `native`, `native_fixed`, `native_listwise` 10 mm 실험이다. 학습 입력을 준비하는 반복 CPU 계산을 재사용한다. 모델, loss, 후보, 실제 graph view, epoch, physical batch, optimizer와 저장된 진행 상태는 변경하지 않는다.

## 서버에서 확인한 병목

사용자가 제공한 `native_fixed` epoch 2의 완료된 14개 update 기록 기준이다. 서버 파일을 직접 읽거나 서버 프로세스를 조작한 결과가 아니다.

| 측정 | 시간 |
|---|---:|
| 입력 대기 + update | 1,531.92초 |
| 입력 대기 | 1,408.66초 (91.95%) |
| update | 123.26초 |
| forward / backward | 26.73 / 53.35초 |
| CPU → GPU 전송 | 0.31초 |
| 체크포인트 저장 (update에 포함) | 39.32초 |

CPU 입력 생성의 1,497.85초 중 source 준비가 405.49초, 기존 미분류 조립 구간이 909.68초였다. 반면 canonical local graph는 7.61초, sampled views는 26.02초였다. 이 입력 시간은 GPU와 겹칠 수 있으므로 위 전체 시간에 더하지 않는다. 메모리 압력 이벤트는 0이었다. 이는 일반 LRU eviction이나 NFS 지연이 없다는 뜻은 아니다.

미분류 조립 구간에는 상위 graph 생성, 처음 읽는 source/lesion helper, raw identity 검사, patch 조립, collate가 포함된다. 전체 909.68초를 한 함수의 비용이라고 단정하지 않는다. 새 기록은 이 구간을 더 세분한다.

## 적용하는 실행 변경

- **source 준비:** 최초에는 검증된 원본 `choose_source_tumor`와 `prepare_local_source`를 그대로 실행한다. 선택된 종양의 작은 patch mask, 정확한 centroid와 metadata, 준비된 원본 source graph를 저장한다. 다시 읽을 때 full mask를 정확히 복원하고 기존 RAM LRU와 예산으로 관리한다. 전체 CT 크기의 source mask를 파일에 반복 저장하지 않는다.
- **간 통계와 source 방향:** 원본 `_liver_raw` 반환 배열을 환자/region별로, `_principal_axis` 반환값을 source별로 재사용한다. 공식이나 부동소수점 집계 순서를 다시 구현하지 않는다. 다른 병변의 mask는 source 축 캐시에 섞이지 않는다.
- **측정:** 상위 graph, prototype graph, binding, collate를 별도로 기록한다. 내부 helper 시간은 부모 구간과 겹치는 세부 측정으로 표시하고 총시간에 중복 합산하지 않는다.

새 파일은 각 실험이 소유한 `data/source_prepared`와 `data/compact_upper_v1`에 들어간다. 기존 source, checkpoint, canonical cache와 manifest를 덮어쓰거나 삭제하지 않는다. raw SHA, 원본 code/scope, CT shape/spacing, 설정, source identity를 결속하며 손상된 결과를 조용히 새 값으로 대체하지 않는다. 이미 실행 중인 프로세스에는 자동 적용되지 않는다.

## A6000 VRAM과의 관계

서버에서 관측된 training peak는 11.12 GiB다. 남는 VRAM이 있지만 지금 대기의 대부분은 전송 이전의 CPU 계산이다. 전송 0.31초를 줄이는 device prefetch만으로 1,408.66초를 해결할 수 없다. 입력을 제때 공급하는 것이 먼저다.

이 패치는 VRAM을 인위적으로 채우지 않으며 현재 비교 실험의 batch를 중간에 바꾸지 않는다. 입력 대기가 줄어든 뒤 physical batch 또는 activation 저장 정책을 비교하려면 최대 크기 graph를 포함한 실제 CUDA 측정이 필요하다. 관측된 peak 하나만으로 나머지 VRAM 전부를 안전하게 사용할 수 있다고 판단하지 않는다. 학습 중 바뀌는 CNN/GNN embedding을 오래 캐시해 gradient나 가중치 갱신을 건너뛰지도 않는다.

## 검증 범위

검증 결과는 `validation/comparison_preparation/verification.json`에 기록한다. 로컬 상세 자료는 `work/runs/v1.9/preparation/debug-20261008/`에 보관한다. CPU 단위 검사와 실제 CT/CUDA DEBUG 검증을 production 처리량·40 epoch 학습·추천 품질과 구분한다. 실행 결과 없이 서버 epoch 시간이나 가속 배수를 약속하지 않는다.

관련 단위 검사 104개가 통과했다. 실제 CT 세 환자의 보존된 fixture에서 train8 epoch 1/2와 val129를 cold/hot/메모리 eviction/새 provider 재열기의 네 조건으로 비교했다. 총 12개 배치 대조에서 값·dtype·shape·stride·PyG metadata·전역 RNG와 원본 source/준비 graph/helper 값이 같았다. 학습 graph view의 실제 epoch를 고정하지 않았다.

위 입력 일치 검사의 대조군은 보존된 원본 직렬 provider다. 그 검사의 시간에는 이전에 배포한 병렬 view와 국소 mask 추출 효과도 포함되므로, 이번 패치만의 개선 수치로 사용하지 않는다.

추가로 **현재 배포 경로의 병렬 view·국소 mask·메모리 예산을 그대로 적용한 대조군**과 비교했다. 같은 실제 CT train8 epoch 2 배치를 세 번씩 준비하고 실행 순서를 교대했을 때 중앙값은 기존 9.492초, 새 경로 3.176초였다. 입력 준비 시간 약 66.5% 감소다. RAM eviction 후에는 30.490초 → 17.159초였다. val129 한 쌍은 20.920초 → 19.089초로 개선이 작았다. 검증의 후보 graph 생성·sampling까지 모두 없어진 것이 아니다. 총 여섯 쌍의 입력·graph metadata·RNG가 같았다. 로컬 Windows SSD의 세 환자 DEBUG fixture이며 서버 NFS, 네 실험 동시 실행 또는 epoch 전체 시간의 가속 배수가 아니다. 새 경로의 disk cache는 앞선 검증에서 생성한 정확한 완료 결과를 사용했다. 새 source를 처음 만들 때는 원래 계산과 저장 비용이 여전히 든다.

전체 원본 모델 10,434,532 parameters로 실제 CUDA optimizer update 두 번과 129후보 validation 세 번을 수행했다. 1,085개 parameter tensor의 gradient가 모두 유한했다. 완료 checkpoint 직접 재개와 복제된 완료 실험의 실제 재개 launcher는 각각 추가 update 0개였고 원본 checkpoint·cache·17개 결속 helper는 보존됐다. RTX 5070 Ti에서 관측된 DEBUG peak는 CUDA 2.199 GiB, RSS 16.37 GiB다. 서버 A6000의 최대 graph나 production batch 한계를 측정한 값이 아니다. 상세 입력/학습/성능 기록을 검증 폴더에 함께 보관한다.

세 source의 새 준비 캐시는 총 12,895,896 bytes, 간 통계/축 캐시는 7,587 bytes였다. 이는 전체 서버 cohort 저장량이 아니다. source payload에 전체 CT mask가 포함되지 않는지 실제 파일을 읽어 확인했다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch와 병렬화 가능성을 검토하고 비교 실험의 기존 측정값을 유지했다.
- [x] 서버 제공 GPU·CPU·RAM 기록과 로컬 자원을 확인했다.
- [x] 모델 축소보다 측정된 CPU 입력 병목을 먼저 조사했다.
- [x] DEBUG 검증과 production 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 학습 경로에 사용하지 않았다.
- [x] 변경 입력의 forward·loss·gradient·optimizer 연결을 실제 CUDA로 검사했다.
- [x] 실행 변경과 유지되는 설정을 보고했다.
- [x] 단위·smoke 결과와 전체 학습·전체 평가 미수행을 검증 기록에 구분했다.
