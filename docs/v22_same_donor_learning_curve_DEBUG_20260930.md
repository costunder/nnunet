# Same-donor learning curve — actual CT DEBUG

결론: 작은 학습 표본의 순위는 개선됐다. 별도 검증 표본의 순위는 악화됐다. 일반화 성능이 개선됐다는 근거로 사용할 수 없다.

2026-09-30 RTX 5070 Ti에서 `tools/measure_same_donor_learning_debug.py`를 실행했다. 실제 CT 학습 4case × 2후보, 검증 1case × 2후보다. 학습 8관측 전체를 8회 반복해 32 optimizer update를 수행했다. 이것은 명시적 DEBUG 학습 가능성 검사이며, production 후보128/전체 데이터 학습이 아니다.

| Update | Train rank loss | Train MRR | Validation rank loss | Validation MRR |
|---:|---:|---:|---:|---:|
| 0 | 0.713837 | 0.625 | 0.617756 | 1.000 |
| 4 | 0.581983 | 1.000 | 0.723398 | 0.500 |
| 8 | 0.407816 | 1.000 | 0.839475 | 0.500 |
| 12 | 0.143143 | 1.000 | 1.081635 | 0.500 |
| 16 | 0.019868 | 1.000 | 1.141348 | 0.500 |
| 20 | 0.001481 | 1.000 | 1.901808 | 0.500 |
| 24 | 0.000167 | 1.000 | 2.550794 | 0.500 |
| 28 | 0.000014 | 1.000 | 3.148197 | 0.500 |
| 32 | 0.00000172 | 1.000 | 3.635236 | 0.500 |

Train R@1은 25%→100%, validation R@1은 100%→0%다. 검증은 단 한 case이므로 전체 일반화 실패율을 추정할 수 없다. 그러나 이번 결과를 ‘검증 성능도 오른다’고 보고해서는 안 된다. 작은 표본을 반복 학습한 과적합 양상이며, 이것만으로 원인이나 전체 학습 결과를 확정하지 않는다. 기존 모델과 동일 조건 A/B는 수행하지 않았으므로 수정 효과의 인과적 크기도 측정하지 않았다.

모델 5,535,830 parameter, FP32, AdamW 원래 lr/weight decay/gradient clip, retained activation, physical batch2, support patients2, workers4, RAM cache4GiB, CUDA budget8GiB, RSS budget24GiB다. CNN·SAGE·L1·L2 구조 및 학습 loss는 현재 구현 그대로다. 각 pass 후 전체 DEBUG train support를 갱신했다. 평가에서는 train-only support와 기존 환자 제외 규칙을 사용했고, 평가 전후 RNG를 복원했다. 후보와 평가 case는 모든 snapshot에서 동일하다.

측정 구간 39.07초(데이터셋/모델 초기 로드 제외; calibration·32update·refresh·평가 포함), peak CUDA 1,061,577,216 bytes. 실제 물리32/후보128/서버 epoch 비용으로 외삽하지 않는다. production checkpoint 및 ready 표시는 생성하지 않았다. 테스트 통과를 성능 통과로 바꾸지 않는다.

원시 결과: `work/same_donor_curve_DEBUG_20260930/report.json`. 각 snapshot에 case별 실제 종양 순위를 보존했다. 재현 실행은 다음과 같다(출력 폴더는 새 경로여야 한다).

```powershell
.\.venv\Scripts\python.exe -B -u tools/measure_same_donor_learning_debug.py --cache work/region_frozen_reuse_cli_20260929/cache/index.json --fine-cache work/same_donor_learning_DEBUG_20260930/cache/index.json --output work/same_donor_curve_DEBUG_20260930 --debug-passes 8
```

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] production 그래프와 데이터 규모를 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다. 소규모 검사는 DEBUG로 명시했다.
- [x] 기존 DEBUG physical batch2 및 batched GPU 경로를 사용했다. production 배치 최적화 검사는 아니다.
- [x] GPU 및 자원 예산을 확인하고 CUDA peak를 기록했다. RSS 제한을 매 update 검사했다.
- [x] OOM을 숨기거나 모델을 줄이지 않았다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 실제 loss/backward/gradient 검사/optimizer를 실행했다.
- [x] 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] DEBUG 학습과 전체 학습·전체 평가를 구분했다. 전체 학습·전체 평가는 미실행이다.
