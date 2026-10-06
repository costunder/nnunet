# V1.7 C

native CNN 입력/L0와 원래 source-anchor8후보 학습을 결합한다. 원래 GT·sample 순서와 full own-task support는 유지하되 native 입력에는 erasure와 두 view를 만들어 넣지 않는다.

[실행](run.py) · [서버](server.sh) · [설정](config.md) · [공통 구성](../README.md) · [결과](../results.json)

서버 launcher의 `CP_ARM=C`를 선택한다. physical batch의 단위는 curriculum samples이며 sample당8후보다. own-task BEST는 원래8후보 기준이다. 사용자 terminal 보고로40/40epoch·3040updates를 완료했고, epoch 중앙값은2.98분이었다.

| C own8후보 validation | MRR | top1 | margin | loss |
| --- | ---: | ---: | ---: | ---: |
| 초기 | 0.422751 | 0.194444 | -0.009976 | 3.814999 |
| BEST epoch20 | 0.986111 | 0.972222 | 6.795793 | 0.424519 |
| 마지막 epoch40 | 0.918982 | 0.861111 | 6.229910 | 0.625842 |

학습 중 full128 MRR/Hit@1은 own8 BEST epoch20에서0.195344/0.125000, 마지막 epoch40에서0.108725/0이었다. 가장 높은 full128 MRR0.252844(epoch19)는 참고값이며 checkpoint 선택 기준은 아니다. 이후 표준화한 공통21case P+128U 재평가의 MRR0.195354/Hit@1 0.125000은 [results.json](../results.json)에 별도 기록했다.

출처는 이 대화에서 사용자가 직접 제공한 서버 terminal 결과다. 서버 checkpoint·raw JSON을 독립 확인했다는 뜻은 아니다.

기존 actual CT/CUDA2-update DEBUG와 resume 검사는 전체40epoch 품질 결과가 아니다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 실행 계약을 안내했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 기존 기록만 참조하며 새 측정은 없다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 문서 정리에 OOM은 없다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈의 forward, loss, gradient와 optimizer 연결에 관한 기존 증거를 구분했다. 새 학습 검사는 없다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다. 원본 실행 파일은 유지했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
