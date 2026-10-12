"""Render the existing Result schema 1 without running training or metric math.

Input values are observations supplied by an evidence collector. A cell may be
``value`` or ``{value, source, status, reason}``. Missing values remain JSON null.
An online_evaluation directory is admitted only through its completion hashes.
An inline online_evaluation bundle requires the collector's explicit output-hash
verification, and its scope is recorded rather than claimed as a local recheck.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "config/nnunet_result_report_schema_v1.json"
FORMAT = "nnunet_completed_result_layout_v1"
MISSING = "실제 근거에 해당 값 없음"
NO_METRICS = "추가 병변 평가 completion/output SHA 검증 근거 없음"
CHECKLIST = (
    "서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.",
    "사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.",
    "모델 깊이와 너비를 편의상 축소하지 않았다.",
    "그래프와 데이터 규모를 편의상 축소하지 않았다.",
    "숨겨진 subset, cap, fast mode를 추가하지 않았다.",
    "physical batch size와 병렬화 가능성을 실제로 검토했다.",
    "GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.",
    "OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다.",
    "디버그 설정과 최종 설정을 분리했다.",
    "dummy, placeholder, random fallback을 사용하지 않았다.",
    "핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.",
    "실제 실행 설정과 변경 사항을 명확하게 보고했다.",
    "smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.",
)


class EvidenceError(ValueError):
    """A supplied completed artifact or comparison claim is inconsistent."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise EvidenceError(f"JSON object required: {path}")
    return value


def _cell(value: Any, reason: str = MISSING, source: str | None = None) -> dict:
    if isinstance(value, dict) and "value" in value:
        result = copy.deepcopy(value)
        result.setdefault("source", source)
        result.setdefault("status", "RECORDED" if result["value"] is not None else "UNAVAILABLE")
    else:
        result = {"value": copy.deepcopy(value), "source": source,
                  "status": "RECORDED" if value is not None else "UNAVAILABLE"}
    if result["value"] is None:
        result["reason"] = result.get("reason") or reason
    return result


def _value(value: Any) -> Any:
    return value.get("value") if isinstance(value, dict) and "value" in value else value


def _row(columns: list[str], values: dict, *, reason: str = MISSING,
         source: str | None = None) -> dict:
    cells = {key: _cell(values.get(key), reason, source) for key in columns}
    return {**{key: cell["value"] for key, cell in cells.items()},
            "_missing": {key: cell["reason"] for key, cell in cells.items() if cell["value"] is None},
            "_evidence": cells}


def _finite(value: Any, trail: str = "evidence") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise EvidenceError(f"Nonfinite input must be null with a reason: {trail}")
    if isinstance(value, dict):
        for key, item in value.items():
            _finite(item, f"{trail}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite(item, f"{trail}[{index}]")


def _online(bundle: dict, schema: dict) -> dict | None:
    if not bundle:
        return None
    if bundle.get("directory"):
        folder = Path(bundle["directory"]).resolve()
        completion_path = folder / "completion.json"
        if not completion_path.is_file():
            if bundle.get("status") == "VERIFIED_COMPLETE":
                raise EvidenceError("Completed evaluator claim lacks completion.json")
            return {"status": "INCOMPLETE", "reason": "추가 평가 completion.json 없음",
                    "directory": str(folder), "experiment": bundle.get("experiment")}
        completion = _read(completion_path)
        outputs = completion.get("outputs", {})
        required = set(schema["existing_metric_artifact_names"]) - {"completion.json"}
        if (completion.get("format") != "online_eval_completion_v1"
                or completion.get("complete") is not True
                or completion.get("actual_data_evaluation") is not True
                or not isinstance(outputs, dict) or not required.issubset(outputs)):
            raise EvidenceError("Incomplete evaluator completion certificate/output inventory")
        artifacts = []
        for name, expected in outputs.items():
            if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
                raise EvidenceError("Evaluator output must be a direct file name")
            path = folder / name
            if not path.is_file() or not isinstance(expected, str) or _sha(path) != expected:
                raise EvidenceError(f"Evaluator output SHA mismatch/missing: {name}")
            artifacts.append({"path": str(path), "sha256": expected, "status": "LOCAL_BYTES_VERIFIED"})
        summary = _read(folder / "summary.json")
        if (completion.get("summary_sha256") != outputs.get("summary.json")
                or completion.get("evaluation_contract_sha256") != summary.get(
                    "evaluation_contract", {}).get("contract_sha256")):
            raise EvidenceError("Evaluator summary/evaluation-contract binding differs")
        csv_rows = {}
        for name in ("criterion_summary.csv", "size_recall_summary.csv", "patient_cluster_statistics.csv"):
            with (folder / name).open(newline="", encoding="utf-8-sig") as stream:
                csv_rows[name] = list(csv.DictReader(stream))
        artifacts.append({"path": str(completion_path), "sha256": _sha(completion_path),
                          "status": "LOCAL_BYTES_VERIFIED"})
        return {"status": "VERIFIED_COMPLETE", "experiment": bundle.get("experiment"),
                "summary": summary, "completion": completion, "artifacts": artifacts,
                "csv_rows": csv_rows, "verification_scope": "local completion and all published output bytes"}
    verification = bundle.get("verification", {})
    if verification.get("outputs_sha256_verified") is not True:
        if bundle.get("status") == "VERIFIED_COMPLETE":
            raise EvidenceError("Inline completed evaluator bundle lacks collector output-hash proof")
        return {"status": "UNVERIFIED", "reason": NO_METRICS,
                "experiment": bundle.get("experiment")}
    summary, completion = bundle.get("summary", {}), bundle.get("completion", {})
    if (completion.get("format") != "online_eval_completion_v1"
            or completion.get("complete") is not True
            or completion.get("actual_data_evaluation") is not True
            or not completion.get("summary_sha256")
            or completion.get("summary_sha256") != completion.get("outputs", {}).get("summary.json")
            or completion.get("evaluation_contract_sha256") != summary.get(
                "evaluation_contract", {}).get("contract_sha256")
            or not verification.get("source")):
        raise EvidenceError("Inline evaluator certificate/binding or collector proof source missing")
    return {**copy.deepcopy(bundle), "status": "VERIFIED_COMPLETE",
            "verification_scope": "collector output-byte verification; no local prediction/checkpoint recheck"}


def _native_value(native: dict, name: str) -> Any:
    summary = native.get("summary", {})
    labels = native.get("dataset_labels", native.get("labels", {}))
    selector = native.get(name + "_label")
    if not labels or not summary:
        return None
    if selector is None:
        matches = [value for key, value in labels.items() if key.casefold() == name]
        if len(matches) != 1:
            return None
        selector = matches[0]
    elif isinstance(selector, str) and selector in labels:
        selector = labels[selector]
    if selector not in labels.values():
        raise EvidenceError(f"Native {name} label is not in the observed dataset label mapping")
    key = str(selector) if not isinstance(selector, (list, tuple)) else "(" + ", ".join(map(str, selector)) + ")"
    metric = summary.get("mean", {}).get(key, {})
    return metric.get("Dice") if isinstance(metric, dict) else None


def _native_reason(native: dict, name: str, experiment: str) -> str:
    if not native.get("summary"):
        return experiment + "의 native validation/summary.json 근거 미제공"
    if not native.get("dataset_labels", native.get("labels")):
        return experiment + "의 실제 dataset.json label mapping 미제공; label 의미를 추정하지 않음"
    return experiment + "의 실제 label mapping/native summary에 " + name + " Dice 미확인"


def _native_check(experiment: dict) -> None:
    native = experiment.get("native", {})
    status = _value(experiment.get("completion", {}).get("best_native_evaluation_status", native.get("status")))
    if status != "VERIFIED_COMPLETE":
        return
    receipt = native.get("receipt", {})
    verified = native.get("verification", {})
    if (verified.get("best_receipt_verified") is not True or not verified.get("source")
            or receipt.get("complete") is not True
            or receipt.get("original_checkpoint_load_called") is not True
            or receipt.get("original_full_validation_called") is not True
            or receipt.get("final_checkpoint_fallback") is not False
            or receipt.get("checkpoint_name") != "checkpoint_best.pth"
            or not receipt.get("checkpoint_sha256")
            or not receipt.get("summary_sha256")
            or not receipt.get("validation_cases")
            or set(receipt.get("predictions_sha256", {})) != set(receipt["validation_cases"])):
        raise EvidenceError("VERIFIED_COMPLETE native evaluation requires its verified actual BEST receipt")


def _comparison(experiment: dict, overall: dict) -> dict:
    result = {**overall, **experiment.get("comparison", {})}
    status = result.get("status", "PENDING_VERIFICATION")
    if status == "MATCHED_VERIFIED":
        checks = result.get("admission", {})
        required = ("cohort", "gt_spacing", "own_best", "basic_best", "metric_implementation",
                    "aggregation_matching_empty_mask", "native_training", "cp_policy")
        if any(checks.get(key) is not True for key in required) or not result.get("source"):
            raise EvidenceError("MATCHED_VERIFIED requires all explicit comparison admissions and a proof source")
    elif status not in ("HISTORICAL_REFERENCE", "NOT_COMPARABLE", "PENDING_VERIFICATION"):
        raise EvidenceError(f"Unknown comparison status: {status}")
    result["status"] = status
    result.setdefault("reason", "Basic BEST·실험 통제 비교 검증 미완료")
    return result


def _stats_reason(stats: dict, fallback: str) -> str:
    details = []
    for key in ("bootstrap_inference", "permutation_inference"):
        item = stats.get(key)
        if isinstance(item, dict):
            details.append(key + ": " + json.dumps(item, ensure_ascii=False, sort_keys=True))
    return fallback + ("; " + "; ".join(details) if details else "")


def build_report(evidence: dict, *, schema_path: Path = SCHEMA_PATH) -> dict:
    """Build ordered raw-value tables. Never evaluate a metric or infer an epoch index."""
    _finite(evidence)
    schema = _read(Path(schema_path))
    if schema.get("format") != FORMAT or schema.get("result_schema_version") != 1:
        raise EvidenceError("Existing Result schema 1 is required")
    experiments = copy.deepcopy(evidence.get("experiments", []))
    if not experiments and evidence.get("experiment"):
        experiments = [copy.deepcopy(evidence)]
    if not experiments or any(not isinstance(item, dict) or not item.get("experiment") for item in experiments):
        raise EvidenceError("At least one explicitly named experiment is required")
    names = [item["experiment"] for item in experiments]
    if len(set(names)) != len(names) or "Basic CP" in names:
        raise EvidenceError("Experiment names must be unique and distinct from Basic CP")
    tables = {name: [] for name in schema["tables"]}
    basic = copy.deepcopy(evidence.get("basic", {}))
    basic["experiment"] = "Basic CP"
    online = _online(evidence.get("online_evaluation", {}), schema)
    if online and online.get("experiment") not in names:
        raise EvidenceError("Evaluator bundle must be bound to an exact report experiment")
    metadata, references, artifact_rows = [], [], copy.deepcopy(evidence.get("artifacts", []))
    if online:
        artifact_rows.extend(online.get("artifacts", []))
    criterion_order, bins = schema["detection_criteria_order"], schema["size_bins_order"]
    summaries = {}
    comparisons = {}
    for experiment in experiments:
        name = experiment["experiment"]
        _native_check(experiment)
        comp = _comparison(experiment, evidence.get("comparison", {}))
        comparisons[name] = comp
        matching_online = online if online and online.get("experiment") == name else None
        summary = matching_online.get("summary", {}) if matching_online and matching_online["status"] == "VERIFIED_COMPLETE" else {}
        summaries[name] = summary
        completion = copy.deepcopy(experiment.get("completion", {}))
        completion.update(experiment=name, comparison_status=comp["status"])
        if matching_online:
            completion["lesion_metrics_status"] = matching_online["status"]
        elif _value(completion.get("lesion_metrics_status")) == "VERIFIED_COMPLETE":
            raise EvidenceError("Lesion metric completion requires its verified evaluator bundle")
        completion.setdefault("lesion_metrics_status", "INCOMPLETE")
        training = copy.deepcopy(experiment.get("training", {}))
        for key in ("planned_epochs", "completed_epochs"):
            if key not in training and key in completion:
                training[key] = completion[key]
            if key not in completion and key in training:
                completion[key] = training[key]
        native = experiment.get("native", {})
        receipt = native.get("receipt", {})
        if "best_current_epoch_raw" not in training and "best_current_epoch" in receipt:
            training["best_current_epoch_raw"] = receipt["best_current_epoch"]
        if "predicted_cases" not in completion and receipt.get("validation_cases"):
            completion["predicted_cases"] = len(receipt["validation_cases"])
        if (_value(completion.get("best_native_evaluation_status")) == "VERIFIED_COMPLETE"
                and _value(completion.get("expected_cases")) is not None
                and len(receipt["validation_cases"]) != _value(completion["expected_cases"])):
            raise EvidenceError("Native BEST receipt case count differs from the observed full cohort")
        training["experiment"] = name
        tables["completion_and_best_provenance"].append(_row(schema["tables"]["completion_and_best_provenance"], completion))
        tables["nnunet_training_record"].append(_row(schema["tables"]["nnunet_training_record"], training))
        metadata.append({"experiment": name, "settings": experiment.get("settings", {}),
                         "native": native, "training_details": {key: value for key, value in training.items()
                            if key not in schema["tables"]["nnunet_training_record"]},
                         "comparison": comp, "verification": experiment.get("verification", {})})
    for item in [basic, *experiments]:
        name, native = item["experiment"], item.get("native", {})
        summary = (online or {}).get("summary", {}) if name == "Basic CP" else summaries[name]
        if not online or online.get("status") != "VERIFIED_COMPLETE":
            summary = {} if name == "Basic CP" else summary
        role = "basic" if name == "Basic CP" else "hier"
        seg = copy.deepcopy(item.get("segmentation", {}))
        seg["experiment"] = name
        if summary:
            seg.setdefault("evaluation_cases", summary.get("validation_cases"))
            seg.setdefault("case_macro_tumor_dice", summary.get("case_tumor_dice", {}).get(role + "_mean"))
            seg.setdefault("lesion_mean_dice", summary.get("lesion_dice_quality", {}).get(role + "_mean"))
            seg.setdefault("metric_source_and_aggregation", "online_eval_v2 summary: case_tumor_dice 환자 평균; lesion_dice_quality 전체 GT 병변 가중 평균(unmatched 0)")
        for key in ("case_macro_tumor_dice", "lesion_mean_dice"):
            if seg.get(key) is None:
                seg[key] = _cell(None, "검증된 추가 평가 summary에 " + key + " 값 없음" if summary else NO_METRICS)
        for name_label in ("liver", "tumor"):
            seg.setdefault("native_" + name_label + "_dice", _cell(_native_value(native, name_label),
                _native_reason(native, name_label, name), source="native summary.mean + observed dataset.json labels"))
        if name == "Basic CP":
            seg.setdefault("checkpoint", basic.get("checkpoint_origin_status", "UNVERIFIED / original BEST linkage unverified"))
        else:
            seg.setdefault("checkpoint", native.get("receipt", {}).get("checkpoint_name"))
        tables["whole_tumor_segmentation"].append(_row(schema["tables"]["whole_tumor_segmentation"], seg, reason=NO_METRICS))
        for criterion in criterion_order:
            criterion_summary = summary.get("criteria", {}).get(criterion, {})
            det = copy.deepcopy(item.get("detection", {}).get(criterion, {}))
            raw_det = criterion_summary.get("basic_cp" if role == "basic" else "hiercp", {})
            for key, value in raw_det.items():
                det.setdefault(key, value)
            if "gt" not in det and isinstance(det.get("tp"), int) and isinstance(det.get("fn"), int):
                det["gt"] = _cell(det["tp"] + det["fn"], source="published TP + FN count inventory (no metric recomputation)")
            det.update(experiment=name, criterion=criterion)
            tables["whole_lesion_detection"].append(_row(schema["tables"]["whole_lesion_detection"], det, reason=NO_METRICS))
            for size_bin in bins:
                size = copy.deepcopy(item.get("size_detection", {}).get(criterion, {}).get(size_bin, {}))
                raw_size = criterion_summary.get("by_size", {}).get(size_bin, {})
                for key, source_key in (("gt", "gt"), ("detected", role + "_detected"), ("recall", role + "_recall")):
                    if source_key in raw_size:
                        size.setdefault(key, raw_size[source_key])
                if "missed" not in size and isinstance(size.get("gt"), int) and isinstance(size.get("detected"), int):
                    size["missed"] = _cell(size["gt"] - size["detected"], source="published GT minus detected count inventory (no metric recomputation)")
                size.update(experiment=name, criterion=criterion, size_bin=size_bin)
                tables["size_stratified_detection"].append(_row(schema["tables"]["size_stratified_detection"], size, reason=NO_METRICS))
    basic_seg = tables["whole_tumor_segmentation"][0]
    for experiment in experiments:
        name, summary, comp = experiment["experiment"], summaries[experiment["experiment"]], comparisons[experiment["experiment"]]
        arm_seg = next(row for row in tables["whole_tumor_segmentation"] if row["experiment"] == name)
        entries = []
        for metric, source_key in (("case_macro_tumor_dice", "case_tumor_dice"), ("lesion_mean_dice", "lesion_dice_quality")):
            block = summary.get(source_key, {})
            stats = block.get("statistics", {}) if source_key == "case_tumor_dice" else block
            entries.append((metric, None, "all", basic_seg.get(metric), arm_seg.get(metric), stats))
        for criterion in criterion_order:
            block = summary.get("criteria", {}).get(criterion, {})
            for metric in ("recall", "precision", "f1", "fp_per_case"):
                entries.append((metric, criterion, "all", block.get("basic_cp", {}).get(metric),
                                block.get("hiercp", {}).get(metric), block.get("statistics", {}).get(metric, {})))
        for criterion in criterion_order:
            for size_bin in bins:
                block = summary.get("criteria", {}).get(criterion, {}).get("by_size", {}).get(size_bin, {})
                entries.append(("size_recall", criterion, size_bin, block.get("basic_recall"),
                                block.get("hier_recall"), block.get("statistics", {})))
        for metric, criterion, size_bin, basic_value, arm_value, stats in entries:
            reason = _stats_reason(stats, comp["reason"] if comp["status"] != "MATCHED_VERIFIED" else "원래 evaluator에 해당 통계 미산출/undefined")
            values = dict(experiment=name, metric=metric, criterion=criterion, size_bin=size_bin,
                          basic_value=basic_value, experiment_value=arm_value,
                          comparison_status=comp["status"], evidence_or_unavailable_reason=reason)
            produced = dict(experiment_minus_basic=stats.get("difference"), ci95_low=stats.get("ci_low"),
                            ci95_high=stats.get("ci_high"), patient_swap_p=stats.get("permutation_p"),
                            wilcoxon_p=stats.get("wilcoxon_p") if metric == "case_macro_tumor_dice" else None)
            if comp["status"] == "MATCHED_VERIFIED" and summary:
                values.update(produced)
            elif comp["status"] == "HISTORICAL_REFERENCE" and summary:
                reference = _row(schema["tables"]["basic_cp_comparison"], {**values, **produced}, reason=reason)
                reference["_original_statistics"] = copy.deepcopy(stats)
                references.append(reference)
            row = _row(schema["tables"]["basic_cp_comparison"], values, reason=reason)
            if criterion is None:
                row["_missing"]["criterion"] = "전체 분할 Dice에는 병변 검출 criterion 적용 없음"
            if metric != "case_macro_tumor_dice":
                row["_missing"]["wilcoxon_p"] = "기존 evaluator는 이 metric의 Wilcoxon을 산출하지 않음"
            row["_original_statistics"] = copy.deepcopy(stats)
            tables["basic_cp_comparison"].append(row)
    return {"format": FORMAT, "result_schema_version": 1,
            "kind": "rendered_evidence_report_not_metric_execution",
            "source_evidence_kind": evidence.get("kind", "COLLECTED_EVIDENCE"),
            "layout_sha256": _sha(Path(schema_path)), "section_order": schema["section_order"],
            "table_columns": schema["tables"], "detection_criteria_order": criterion_order,
            "size_bins_order": bins, "tables": tables, "historical_reference_comparison": references,
            "experiment_metadata": metadata, "basic_metadata": basic,
            "online_evaluation": online, "artifacts": artifact_rows,
            "verification": copy.deepcopy(evidence.get("verification", {})),
            "limitations": copy.deepcopy(evidence.get("limitations", [])),
            "checklist": copy.deepcopy(evidence.get("checklist", {})),
            "ratio_scale": "0_to_1", "difference_direction": "experiment_minus_basic",
            "metric_math_executed": False, "training_executed_by_renderer": False}


def render_text(report: dict) -> str:
    """Render template headers and table order, including missing rows and reasons."""
    reasons: dict[str, str] = {}

    def unavailable(reason: str) -> str:
        if reason not in reasons:
            reasons[reason] = f"N{len(reasons) + 1:02d}"
        return "NA [" + reasons[reason] + "]"

    def display(value: Any, reason: str = MISSING, *, fp: bool = False) -> str:
        if isinstance(value, dict) and "value" in value:
            reason = value.get("reason") or reason
        value = _value(value)
        if value is None:
            return unavailable(reason)
        if isinstance(value, bool):
            result = "true" if value else "false"
        elif isinstance(value, float):
            result = f"{value:.{3 if fp else 4}f}"
        elif isinstance(value, (dict, list, tuple)):
            result = json.dumps(value, ensure_ascii=False)
        else:
            result = str(value)
        return result.replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    def field(row: dict, key: str) -> str:
        return display(row.get(key), row.get("_missing", {}).get(key, MISSING), fp=key == "fp_per_case")

    def paired(row: dict, first: str, second: str) -> str:
        return field(row, first) + "/" + field(row, second)

    def table(headers: list[str], rows: list[list[str]]) -> list[str]:
        return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                *["| " + " | ".join(row) + " |" for row in rows], ""]

    def prose(value: Any, reason: str = MISSING) -> str:
        if value is None or value == {} or value == []:
            return unavailable(reason)
        if isinstance(value, dict) and "value" not in value:
            return "; ".join(f"{key}={display(item)}" for key, item in value.items())
        return display(value, reason)

    tables = report["tables"]
    title = " / ".join(item["experiment"] for item in report["experiment_metadata"])
    lines = [f"# nnU-Net 학습·BEST 평가 결과 — {title}", "", f"Result schema: {FORMAT}",
             "단위: 원본 비율 0~1. NA [Nxx]는 08절의 미산출·미검증 사유를 참조합니다.", "",
             "## 01 완료 상태 및 BEST 출처", ""]
    lines += table(["실험", "학습 상태", "완료/목표 epoch", "BEST native 평가 상태", "예측 case/전체", "병변 지표 상태", "비교 상태"], [
        [field(r, "experiment"), field(r, "training_status"), paired(r, "completed_epochs", "planned_epochs"),
         field(r, "best_native_evaluation_status"), paired(r, "predicted_cases", "expected_cases"),
         field(r, "lesion_metrics_status"), field(r, "comparison_status")] for r in tables["completion_and_best_provenance"]])
    for meta in report["experiment_metadata"]:
        native, details = meta["native"], meta["training_details"]
        receipt = native.get("receipt", {})
        tr = next(r for r in tables["nnunet_training_record"] if r["experiment"] == meta["experiment"])
        lines += [f"{meta['experiment']} 평가 checkpoint: {prose(receipt.get('checkpoint_name'))}",
                  "BEST epoch: 원시 current_epoch=" + field(tr, "best_current_epoch_raw") + "; index 설명=" + prose(details.get("best_epoch_index_definition"), "원시 current_epoch의 0/1-based 정의 근거 없음; 추정하지 않음"),
                  "BEST 출처/SHA·실제 로드 증거: " + prose({key: receipt.get(key) for key in
                    ("checkpoint_path", "checkpoint_sha256", "original_checkpoint_load_called", "original_full_validation_called", "validation_cases", "predictions_sha256", "summary_sha256")}),
                  "학습 완료 증거: " + prose(native.get("training_receipt")), ""]
    lines += ["## 02 실험 설정 및 GNN·CP 요약", ""]
    for meta in report["experiment_metadata"]:
        settings = meta["settings"]
        lines += ["실험: " + meta["experiment"] + "; " + prose(settings.get("experiment")),
                  "GNN: " + prose(settings.get("gnn")), "GNN BEST: " + prose(settings.get("gnn_best")),
                  "CP: " + prose(settings.get("cp")), "nnU-Net: " + prose(settings.get("nnunet")), ""]
    lines += ["## 03 nnU-Net 학습 기록", ""]
    lines += table(["실험", "목표/완료 epoch", "BEST 원시 epoch", "BEST 선택 기준", "최근 loss epoch", "train_loss", "val_loss", "마지막 온라인 Pseudo dice", "CP 적용/시도(율)", "실제 학습시간", "peak VRAM/RSS", "측정 범위"], [
        [field(r, "experiment"), paired(r, "planned_epochs", "completed_epochs"), field(r, "best_current_epoch_raw"),
         field(r, "best_selection_rule"), field(r, "latest_loss_epoch"), field(r, "train_loss"), field(r, "val_loss"),
         field(r, "online_pseudo_dice"), paired(r, "cp_applied", "cp_attempted") + "(" + field(r, "cp_apply_rate") + ")",
         field(r, "training_seconds") + " s", paired(r, "peak_vram_gib", "rss_gib") + " GiB",
         field(r, "measurement_scope")] for r in tables["nnunet_training_record"]])
    for meta in report["experiment_metadata"]:
        details = meta["training_details"]
        lines += [meta["experiment"] + " 시작/종료 시각: " + prose(details.get("started_at")) + "/" + prose(details.get("ended_at")),
                  "온라인 Pseudo dice 배열과 실제 class 순서: " + prose(details.get("pseudo_dice_class_order")),
                  "epoch time 측정: " + prose(details.get("epoch_time")), ""]
    lines += ["온라인 Pseudo dice는 아래 BEST 전체 평가 성능과 별도로 기록합니다.", "", "## 04 전체 종양 분할 성능", ""]
    lines += table(["실험", "checkpoint", "평가 case 수", "전체 종양 Dice(case macro)", "병변 평균 Dice", "native 간 Dice", "native 종양 Dice", "metric 출처/집계"], [
        [field(r, k) for k in report["table_columns"]["whole_tumor_segmentation"]] for r in tables["whole_tumor_segmentation"]])
    lines += ["## 05 전체 병변 검출 성능 — 고정 6개 기준", ""]
    lines += table(["실험", "criterion", "GT 수", "TP", "FP", "FN", "Recall", "Precision", "F1", "FP/case"], [
        [field(r, k) for k in report["table_columns"]["whole_lesion_detection"]] for r in tables["whole_lesion_detection"]])
    lines += ["Basic CP와 각 arm의 6개 기준을 고정 순서로 모두 표시하며 사후 primary endpoint를 선택하지 않습니다.", "", "## 06 크기별 검출 성능 — 고정 3구간 × 6개 기준", ""]
    lines += table(["실험", "criterion", "size_bin", "GT 수", "검출수", "미검출수", "Recall"], [
        [field(r, k) for k in report["table_columns"]["size_stratified_detection"]] for r in tables["size_stratified_detection"]])
    lines += ["소형: GT 26-connectivity 병변 체적의 등가구 직경 <=10mm. CP donor 자격 <=20mm와 구분합니다.",
              "GT 크기 구간의 FP/precision 및 소형 Dice는 별도 산출 근거가 없으면 보고하지 않습니다.", "", "## 07 Basic CP 대비 차이 및 통계", ""]

    def compare_table(rows: list[dict]) -> list[str]:
        formatted = []
        for row in rows:
            fp = row["metric"] == "fp_per_case"
            cells = [field(row, key) for key in ("experiment", "metric", "criterion", "size_bin")]
            cells += [display(row.get(key), row["_missing"].get(key, MISSING), fp=fp) for key in
                      ("basic_value", "experiment_value", "experiment_minus_basic")]
            cells += ["[" + display(row.get("ci95_low"), row["_missing"].get("ci95_low", MISSING), fp=fp) + ", "
                      + display(row.get("ci95_high"), row["_missing"].get("ci95_high", MISSING), fp=fp) + "]"]
            cells += [field(row, key) for key in ("patient_swap_p", "wilcoxon_p", "comparison_status", "evidence_or_unavailable_reason")]
            formatted.append(cells)
        return table(["실험(version/arm/run ID)", "metric", "criterion", "size_bin", "Basic 값", "실험 값", "실험-Basic", "95% CI", "patient-swap p", "Wilcoxon p", "비교 상태", "근거/미검증 사유"], formatted)

    lines += compare_table(tables["basic_cp_comparison"])
    if report["historical_reference_comparison"]:
        lines += ["역사적 참고 차이 — HISTORICAL_REFERENCE: 원래 Basic training-BEST linkage/실험 통제가 미검증인 참고값이며 matched BEST 차이가 아닙니다.", ""]
        lines += compare_table(report["historical_reference_comparison"])
    lines += ["차이 방향은 실험-Basic입니다. FP/case의 음수는 개선 방향입니다. CI/p는 원래 evaluator 값만 표시합니다.",
              "p값은 원래 산출물의 보정 상태를 따릅니다; online_eval_v2의 고정 sensitivity suite p는 unadjusted입니다. 사후 주평가지표를 선택하지 않습니다.", "", "## 08 원본 산출물·검증 범위·제한", ""]
    online = report.get("online_evaluation") or {}
    summary = online.get("summary", {})
    lines += ["추가 평가 검증 범위: " + prose(online.get("verification_scope")),
              "Evaluator version/source SHA·평가 계약 SHA: " + prose({"version": summary.get("version"), "source_files": summary.get("source_files", summary.get("source_sha256")),
                  "evaluation_contract_sha256": summary.get("evaluation_contract", {}).get("contract_sha256")}),
              "prediction_origin_status 원문: " + prose(summary.get("prediction_origin_status")),
              "자기 BEST 증거 / Basic BEST 증거: " + prose({"own": [m["native"].get("verification", {}) for m in report["experiment_metadata"]],
                                                            "basic": report["basic_metadata"].get("native", {}).get("verification", {})}), ""]
    if report["artifacts"]:
        lines += ["실제 근거 산출물:", ""]
        for artifact in report["artifacts"]:
            lines.append("- " + prose(artifact))
        lines.append("")
    else:
        lines += ["실제 근거 산출물: " + unavailable("실제 산출물 경로/SHA 목록 미제공"), ""]
    verification = report["verification"]
    for label, key in (("정적 검사", "static_checks"), ("단위/smoke 검사", "unit_smoke_checks"),
                       ("전체 학습", "full_training"), ("BEST 전체 native 평가", "full_best_native_evaluation"),
                       ("추가 병변 metric", "lesion_metrics")):
        lines.append(label + ": " + prose(verification.get(key)))
    lines += ["제한: " + prose(report["limitations"]),
              "이 renderer는 표시 문서만 생성합니다. 학습·예측·metric 계산을 실행하거나 기존 산출물을 덮어쓰지 않습니다.", "", "NA 사유:", ""]
    lines += [f"- [{identifier}] {reason}" for reason, identifier in reasons.items()]
    lines += ["", "## 작업 완료 체크리스트", ""]
    checks = report.get("checklist", {})
    for index, text in enumerate(CHECKLIST, 1):
        item = checks.get(str(index), {})
        checked = item is True or isinstance(item, dict) and item.get("verified") is True
        suffix = " " + str(item.get("reason")) if isinstance(item, dict) and item.get("reason") else ""
        lines.append(f"- [{'x' if checked else ' '}] {text}{suffix}")
    return "\n".join(lines) + "\n"


def render_report(evidence: dict) -> tuple[dict, str]:
    report = build_report(evidence)
    return report, render_text(report)


def write_report(evidence: dict, output_stem: Path) -> tuple[Path, Path]:
    """Publish a new pair exclusively; remove only this call's partial output."""
    report, text = render_report(evidence)
    stem = Path(output_stem).resolve()
    text_path, json_path = Path(str(stem) + ".txt"), Path(str(stem) + ".json")
    if text_path == json_path or text_path.exists() or json_path.exists():
        raise FileExistsError("Fresh report paths required; existing results are never overwritten")
    stem.parent.mkdir(parents=True, exist_ok=True)
    created: list[tuple[Path, int]] = []
    try:
        for path, content in ((json_path, json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
                              (text_path, text)):
            # O_EXCL is the final race-safe check, including against symlinks.
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                created.append((path, os.fstat(stream.fileno()).st_ino))
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
    except BaseException:
        for path, inode in reversed(created):
            if path.exists() and path.stat().st_ino == inode:
                path.unlink()
        raise
    return text_path, json_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output-stem", required=True, type=Path,
                        help="Fresh path without .txt/.json; neither output may already exist")
    args = parser.parse_args(argv)
    text_path, json_path = write_report(_read(args.evidence), args.output_stem)
    print(json.dumps({"format": FORMAT, "text": str(text_path), "json": str(json_path),
                      "metric_math_executed": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
