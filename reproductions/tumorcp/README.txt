TumorCP / KiTS19 paper reproduction
=================================

Purpose
Reproduce TumorCP* + ImgDA on the official labelled KiTS19 cohort. This is
separate from the project's liver Basic CP and nnU-Net v2 experiments.

Sources
Paper: https://arxiv.org/html/2107.09843
Code: https://github.com/YaoZhang93/TumorCP
Code revision: 0fd9da71712f0b1467f32206e96823cc880256bb
Labels: https://github.com/neheller/kits19
Label revision: 456b293435b25303bbc7995f094f83a0afdff0eb
CT: https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging
CT revision: 65f1f295873a326230153c7e1de0c7dba10f0b29

Protocol
210 annotated patients; 168 training / 42 held-out. Official-code fold 0,
KFold(5, shuffle=True, random_state=12345). The paper's exact 42 IDs were not
published in the checked source, so equality to the paper split is unverified.
Official Generic_UNet, automatic 3d_fullres plans, batch 2, accumulation 1,
500 epochs x 250 optimizer updates. No liver data or ResEncM substitution.
Training monitor uses training patients only. Primary evaluation is the fixed
epoch-500 FINAL on all 42 held-out full volumes, without holdout-selected
postprocessing. Published tumor Dice reference: 84.75 +/- 20.87 percent.
An implemented pipeline or successful smoke is not evidence of that accuracy.

Files
config.json           Fixed research settings and explicit unverified limits.
requirements.txt      Private-environment dependency pins; existing CUDA Torch.
sources.json          Public upstream file URLs, sizes and pinned digests.
download.py           Full-cohort CT download, resumable and hash checked.
runtime.py            Copies preserved upstream into an isolated runtime.
paper_trainer.py      Paper settings on the official trainer/network/loss.
run.py                Complete preparation, 500-epoch training and FINAL42.
smoke.py              Separate actual-CT/CUDA gradient and resource checks.
test_runtime.py       Source/contract checks; not model-quality evaluation.
liver.json            Single paper-profile liver experiment; all131,105/26.
liver_remote.py       Independent liver deployment and PBS submission.
liver_prepare.pbs     Existing liver CT -> original full-resolution preparation.
liver_train.pbs       Actual CT smoke -> full500 -> fixed FINAL26.
test_liver.py         Split and metric-scope unit checks; generated tiny arrays.
upstream/             Local official checkout, ignored by this repository.
env/                  Private local environment, ignored; not training evidence.

Released code versus paper
The released loader has independent 0.8 intra/inter gates. The paper profile
uses one 0.8 CP gate, followed by equal intra/inter selection, at most one paste.
Inter donors come only from a different training patient. The profile passes
the paper's spatial transform arguments explicitly and uses blur sigma 0.5-1.
All source patches and reasons are recorded in runtime_receipt.json. Original
upstream bytes remain untouched. Compatibility patches do not shrink the model.

Storage
Keep one verified raw CT/label copy. Task040 inputs use hardlinks where allowed.
Preprocessed arrays and checkpoints belong to this reproduction only. No prior
liver experiments, remote sessions or unrelated processes are modified.

Single liver experiment
The separately requested liver run uses liver.json: all131 MSD Task03 patients,
the original project105/26 split, official nnU-Net v1 fullres/B2/500epochs and
one paper TumorCP arm. --dataset-profile binds the original split file SHA and
IDs. Raw inputs use existing imagesTr/labelsTr. A tumor-free intra recipient
keeps its training sample and skips only that impossible CP event; actual CP
attempts, pasted events and skips are logged. Final26 reports all-patient Dice
(empty GT and prediction both empty =1) and positive-only tumor Dice separately.
This is a paper-setting transfer to liver, not KiTS19 accuracy reproduction or
the project's ResEncM250 matched comparison. The KiTS19 jobs use their original
pinned checkout and are unaffected by the separate liver checkout/output.

작업 완료 체크리스트
- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 검토하는 별도 CUDA smoke를 구현한다.
- [ ] 새 작업에 할당된 GPU, CPU, RAM 활용 상태를 실제 실행에서 확인했다.
- [ ] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [ ] 실제 CT/CUDA에서 핵심 모듈의 loss, gradient, optimizer 연결을 검증했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 기록한다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분한다.
