# Notion 연구 기록 — 2026-09-29

사용자 요청으로 **연구용 → 소형암 CP 추천**을 만들었다.

- [연구용](https://app.notion.com/p/3eabea17210b81c2913cde71d642cab5)
- [소형암 CP 추천](https://app.notion.com/p/3eabea17210b81c8969dd61f2aef459e)

## 기록 목록
- [01 · v1 — 기존 HierCP 기준선](https://app.notion.com/p/3eabea17210b81838581c6bb30c35c52)
- [02 · v2.0 → v2.1 — view 정렬에서 환자 간 정렬로](https://app.notion.com/p/3eabea17210b8110bfb1e727bb08fbe9)
- [03 · v2.2 초기 — 수작업 특징에서 CT-only CNN으로 정정](https://app.notion.com/p/3eabea17210b81dbac2dfcfdf236aa26)
- [04 · v2.21 — data/label·T/F/U 의미 계약](https://app.notion.com/p/3eabea17210b814b8750e261c17e3be8)
- [05 · v2.22 r1/r2 — 관측 과제와 군집 기반 L2](https://app.notion.com/p/3eabea17210b81c38a50f364b849cb6c)
- [06 · r3/r4·풀링·CNN 후보 — 변경 및 반려 이력](https://app.notion.com/p/3eabea17210b8182b307e741241519c3)
- [07 · paired r5/r6 + observed_rank_v1 — 현재 학습 목표](https://app.notion.com/p/3eabea17210b8110b6b1f60922b41f7f)
- [08 · 9/29 L0 최적화 — EZ-SP·GraphSAGE·고정 영역](https://app.notion.com/p/3eabea17210b819da1bff616a725cfc4)
- [09 · 데이터·누수 방지·Basic CP 비교 계약](https://app.notion.com/p/3eabea17210b81b8afa5fc24eb286c88)
- [10 · 과거 실험 결과 — 소형 병변·전체 성능·ablation](https://app.notion.com/p/3eabea17210b810baaddcab51b78553c)
- [11 · References·미검증 항목·기록 관리](https://app.notion.com/p/3eabea17210b81f1aac9f9f713c8e500)

## 검증 및 범위
상위/프로젝트 구조와 상세 11페이지를 생성 후 다시 fetch했다. 모두 연구용 / 소형암 CP 추천 아래에 있으며 본문·근거 파일 절이 존재하고 truncated/unknown block 없음. 총13페이지(연구용1, 프로젝트1, 상세11).
버전별 L0/L1/L2, 손실·추천·변경 이유·폐기 이력·원시 자료 경로·검증 범위를 기록했다. 과거 소형암 결과는 복구 후 EXPERIMENT_RECORD/SMALL_TUMOR_RESULTS 기준을 사용했다. 복구 전 docs/results_summary를 전체 결과로 사용하지 않았다.
현재 working-tree 기록과 GitHub 배포 상태를 구분하고, 과거 성능과 최신 DEBUG 비용을 섞지 않았다. 모델/설정/학습/서버 상태를 변경하지 않았다. 의료영상·가중치 바이너리와 검토 ZIP은 Notion에 업로드하지 않았다.
노션 본문 원고와 URL·조회 검증 정보는 같은 폴더 notion_research_snapshot_20260929.json에 보존한다. 자동 동기화는 설정하지 않았다.

## 작업 완료 체크리스트
- [x] 서버 또는 원격 세션 종료 위험 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [ ] Physical batch 및 병렬화 신규 검토 — 문서화 작업으로 해당 없음.
- [ ] GPU/CPU/RAM 신규 측정 — 이번 작업에서 학습·평가를 실행하지 않았다.
- [ ] OOM 원인 신규 조사 — 이번 작업에서 모델 실행 없음.
- [x] DEBUG와 최종 설정/결과를 구분했다.
- [x] Dummy·placeholder·random fallback을 결과로 제시하지 않았다.
- [ ] Forward/loss/gradient/optimizer 신규 검증 — 이전 검증 보고서만 전재했다.
- [x] 기록 범위와 출처, 변경 사항을 명확하게 보고했다.
- [x] Smoke와 전체 학습·전체 평가를 구분했다.

