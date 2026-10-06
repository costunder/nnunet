# r4 검토 반영: 전달 ZIP의 단독 재검사

첨부 r4 검토 전체를 읽고 대조했다. 새로운 P0/P1 모델 오류는 보고되지 않았다. 실제416 CUDA 결과·초기 weights·원본 source·sampling contract는 이미 결속돼 있다. 이번 변경은 전달 자료를 개발 PC의 `work/validation` 없이 검사하게 만드는 수정이다.

## 수정 내용

`tools/build_v1_runtime416_review.py`에 read-only `--bundle-input`을 추가했다. 입력은 ZIP 또는 `manifest.json / repository / evidence`를 포함한 압축 해제 root다. ZIP은 추출하거나 실행하지 않고 actual member bytes를 읽는다. 외부 manifest inventory·파일 길이·SHA와 ZIP CRC를 확인한 뒤 기존 원본202-file archive 검사 및 runtime416 report/branch/snapshot/계약 검사를 그대로 실행한다. manifest SHA는 계산해 기록하며 외부 서명처럼 인증됐다고 주장하지 않는다. 디렉터리 입력의 외부 CRC는 NOT_APPLICABLE이다.

두 artifact 검사 파일은 명시적 bundle input이나 압축 해제 위치를 감지하면 실제 bundle bytes만 읽는다. 감지한 bundle이 잘못됐을 때 개발 workspace로 fallback하지 않는다. ZIP 중복 경로·상위 디렉터리 경로·absolute/drive/backslash 경로·symlink·누락/추가 멤버·hash/길이 불일치도 거부한다.

default Python/pytest가 만든 `repository` 내부의 `__pycache__` bytecode와 `.pytest_cache`는 비증거 파일로 구분하며 디렉터리 검증 결과에 그 경로를 기록한다. manifest에 포함된 파일은 이 예외로 생략하지 않으며 그 외 추가 파일은 계속 거부한다. 원본 sampler·runner·model·loss·데이터·profile과 기존 r3/r4 ZIP 및 원시 GPU 결과는 변경하지 않는다. 이번에는 새로운 CT/GPU 학습을 실행하지 않는다.

## 압축 해제 폴더에서 실행

압축을 푼 폴더를 현재 위치로 두고 다음을 실행한다. 코드·계약 검사에 필요한 기존 Python/Torch/NumPy 의존성을 사용한다. PyG 신경망 연산·CUDA·CT·checkpoint·workspace cache가 필요한 검사가 아니다.

```bash
python repository/tools/build_v1_runtime416_review.py --bundle-input .
python -m unittest discover -s repository/tests -p test_v1_runtime416_review_bundle.py -v
python -m unittest discover -s repository/tests -p test_v1_nested_review_bundle.py -v
python -m unittest discover -s repository/tests -p test_v1_review_bundle_input.py -v
```

pytest가 있는 환경이라면 `repository` 안에서 위 세 test 파일만 지정해 실행해도 같은 결속 검사를 수행한다. 전체 tests 폴더는 실제 CT/CUDA 또는 개발 fixture가 필요한 다른 검사들도 포함하므로 이 standalone 범위와 구분한다. 이번 환경에서는 pytest가 설치돼 있지 않아 unittest로 실행한다.

ZIP을 직접 검사할 때는 다음처럼 ZIP 경로를 입력한다. `--bundle-input`과 `--output`은 함께 쓸 수 없다. input 검증은 새 ZIP·checkpoint를 만들지 않는다.

```bash
python repository/tools/build_v1_runtime416_review.py --bundle-input /path/to/handoff.zip
```

`select()`의 개발 workspace 기반 새 ZIP 생성 동작은 유지한다. `--output`은 새 ZIP 생성용, `--bundle-input`은 기존 자료 검증용으로 분리한다. r5는 이 전달본의 개정 번호이며 모델은 원본 v1.0과 명시적 strict-nested416 그대로다.

## 이번 파일 검사 결과

개발 workspace에서 기존 runtime416 결속 8개 및 nested 결속 6개, 새 bundle-input 회귀 14개가 PASS했다. 합계 28개는 파일·JSON·SHA·계약 검사이며 신경망이나 정확도 검사를 새로 실행했다는 뜻이 아니다. 새 회귀 검사는 실제 r4 ZIP을 사용하고, 별도 임시 자료의 변조·CRC 손상·누락·추가·중복·위험 경로 및 측정 코드 교체를 거부하는지 확인했다. 기존 ZIP과 원시 결과를 보존했다. 전달본의 독립 재실행 결과는 별도 standalone verification receipt에 기록한다.

## 아직 다른 검증이 필요한 범위

파일 결속 재실행 성공은 v1 ranking 품질이나 CP 효용의 검증이 아니다. 현재 실제 GPU 증거는 기존104/416 DEBUG 결과를 그대로 사용한다. native vs strict-nested416의 전체84/21 ranking 학습·case별 paired 지표·whole-epoch 시간·실제128 candidate inference/transform·Basic CP80 대비 nnU-Net은 미실행이다. sampler를 다시 변경하거나 임의 품질 통과 기준을 추가하지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 GPU 측정 설정을 유지했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 이번 변경은 파일·계약 검사에 한정한다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 새로운 축소나 fallback을 추가하지 않았다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 기존 실제 CUDA 검사와 코드 경로를 유지했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
