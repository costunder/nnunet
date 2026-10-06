# 저장 공간 정리 — 2026-09-23

사용자 지시: 필요 없는 파일을 정리한다. 원본 CT/GT, 현재 학습 입력, 모델 가중치, 로그·지표·버전 이력은 보존했다. 이번 작업은 모델 변경이나 학습 실행이 아니다.

## 실제 정리 결과

- 실제 SHA256로 동일한 내용임을 확인한1,713개 중복 경로를 NTFS hardlink로 공유했다. 기존 경로와 파일 내용, manifest SHA, dtype/shape는 유지한다. 해제된 중복 저장량101.735GiB: Basic CP 배열49.583GiB, CT 복사본52.152GiB.
- 실패한 context 준비와 교체된 DEBUG 실행의 재생성 가능한 NPY/NPZ11,870개를 개별 삭제했다.9.441GiB. 삭제 목록/원래 SHA를 보존했다. 원본 이미지나 가중치 확장자를 삭제 대상으로 사용하지 않았다.
- 중복 inode를 한 번만 계산한 프로젝트 파일 내용 합계:563.181GiB →452.014GiB. 이 값은 파일 내용 기준이고 NTFS allocation/slack까지 합친 값은 아니다.
- D드라이브 여유 공간 실측:1,299,694,891,008bytes →1,419,118,116,864bytes. **119,423,225,856bytes(119.423GB /111.222GiB) 증가**했다.
- 파일 경로마다 중복 합산하는 논리 크기는553.749GiB다. 하드링크 경로를 각각 더하는 탐색기/도구에서는 이 값이 보일 수 있다. 동일 데이터를 실제 디스크에 여러 벌 남긴 것이 아니다.

## 삭제 범위

아래의 `.npy`, `.npz` 파일만 고정 목록으로 열거하고 삭제했다. 재귀 폴더 삭제 명령은 쓰지 않았다. 각 경로의 절대 위치가 workspace와 지정된 폐기 cache root 내부인지 확인했다.

- `work/v222_train_20260922/case_benchmark_run1/context/patches/`: 이미 폐기한 blind-radius 설정에서 context 준비에 실패한 실행. 학습이 시작되지 않은 부분 캐시. 현재 raw-CT L0 캐시와 별개다.
- `work/shared_donor_fix_20260919/real_cp_DEBUG1/bank/`
- `work/shared_donor_fix_20260919/real_cp_DEBUG2/bank/`
- `work/shared_donor_fix_20260919/real_cp_DEBUG3/bank/`

위 DEBUG bank의 기존 JSON/CSV/log/verification 지표는 남겼다. 삭제된 payload를 다시 로딩하려면 해당 DEBUG 입력을 재생성해야 한다. 부모 위치에 `*.storage_cleanup.json`으로 폐기 사실과 감사 목록을 연결했다. 예전 성공/실패 기록을 현재 완전한 bank가 존재한다는 뜻으로 읽으면 안 된다.

## 보존과 검증

현재 Basic CP 코드가 reference_inputs/preparation/raw_cases 세 종류를 모두 읽으므로 통째로 삭제하지 않았다. 불필요한 물리적 복사본만 공유했고 reader 코드와 manifest를 변경하지 않았다. dtype float64를 float32로 바꾸거나 모델/입력 규모를 줄이지 않았다. 하드링크 파일도 기존과 같은 불변 입력으로 취급하며 새 내용은 새 파일로 게시해야 한다.

검증 결과:

- 1,713개 공유 경로가 실제로 동일한 file identity인지 확인.
- 11,870개 폐기 캐시 파일이 삭제됐는지 확인.
- Basic CP105개 case의 manifest 검증 통과.2,205개 배열의 파일 존재/shape/dtype와 manifest SHA 확인.
- liver_1/liver_22/liver_108을 production RawBankStore로 실제 로딩. 각21배열과 원본 CT/GT SHA 재검증. 가장 큰 baseline 배열의 case(liver_22) 포함.
- 현재 L0 cache14,102개 patch 경로 모두 보존.
- frozen v1 검증: `74dcc2cf03d2d40d1f582223321d96004333f661`.

## 감사 파일

- `work/storage_cleanup_20260923/plan.json`: 원본 SHA/size/mtime/inode, 정확한 source/destination와 삭제 목록.
- `work/storage_cleanup_20260923/actions.jsonl`: 각 변경의 intent와 완료 기록. 기존 파일 내용 확인 후 하드링크를 만들고 atomic replace했으며 모델 출력/평가값을 수정하지 않았다.
- `work/storage_cleanup_20260923/result.json`: 실제 정리 전후 byte 수와 드라이브 여유 공간.
- `work/storage_cleanup_20260923/verification.json`: 정리 후 현재 데이터 로딩 검증.
- `tools/cleanup_verified_storage.py`, `tools/verify_storage_cleanup.py`: 이번 고정 범위의 계획/실행/검증 도구. 기존 output을 덮어쓰거나 부분 실패를 숨기지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 원본 데이터와 기존 실험 결과를 파괴적으로 변경하지 않았다. 명시적으로 폐기한 재생성용 캐시만 삭제했다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] 학습 batch는 변경하지 않았고 독립 파일 해시 검증은4workers로 병렬 수행했다.
- [x] 디스크 사용량과 실행 중인 프로젝트 Python 프로세스를 확인했다. GPU 학습은 실행하지 않았다.
- [x] OOM 대응이나 모델 축소가 필요하지 않은 저장 정리 작업이었다.
- [x] DEBUG 캐시와 현재 학습 입력을 구분했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] forward/loss/optimizer 경로를 변경하지 않았으며 production 데이터 로더를 검증했다.
- [x] 실제 변경 사항과 해제된 용량을 명확하게 보고했다.
- [x] 저장 검증과 전체 학습/평가를 구분했다.
