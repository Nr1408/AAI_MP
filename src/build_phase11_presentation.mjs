import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = path.resolve(path.join(path.dirname(fileURLToPath(import.meta.url)), ".."));
const SKILL_DIR = "C:\\Users\\Nishit\\.codex\\plugins\\cache\\openai-primary-runtime\\presentations\\26.904.11930\\skills\\presentations";
const TMP_DIR = path.join(ROOT, "results", "freetwinev", "phase11_build");
const FINAL_PPTX = path.join(ROOT, "results", "freetwinev", "phase11_presentation.pptx");
const FIG_DIR = path.join(ROOT, "results", "freetwinev", "figures");
const RUNTIME_PYTHON = "C:\\Users\\Nishit\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe";
const RUNTIME_NODE_MODULES = "C:\\Users\\Nishit\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\node\\node_modules";
const RUNTIME_BIN_DIR = "C:\\Users\\Nishit\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\node\\bin";
const ARTIFACT_TOOL = "C:\\Users\\Nishit\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\node\\node_modules\\@oai\\artifact-tool\\dist\\artifact_tool.mjs";

process.env.RUNTIME_NODE_MODULES = RUNTIME_NODE_MODULES;
process.env.RUNTIME_BIN_DIR = RUNTIME_BIN_DIR;
process.env.RUNTIME_PYTHON = RUNTIME_PYTHON;

const { Presentation, PresentationFile } = await import(pathToFileURL(ARTIFACT_TOOL).href);
const { resolvePresentationFont, applyPresentationChartFont, finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools", "artifact_tool_utils.mjs")).href,
);

const family = resolvePresentationFont({ fontFamily: "Aptos" });
const W = 1280;
const H = 720;
const navy = "#182B3A";
const teal = "#147C7C";
const green = "#1C7C54";
const amber = "#B36B00";
const slate = "#4B5563";
const light = "#F6F8FA";
const line = "#D8DEE6";

function addText(slide, text, position, style = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  box.text = text;
  box.text.style = {
    typeface: family,
    fontSize: style.fontSize ?? 24,
    bold: style.bold ?? false,
    color: style.color ?? navy,
    autoFit: "shrinkText",
    alignment: style.alignment ?? "left",
  };
  return box;
}

function title(slide, text, kicker = "") {
  if (kicker) {
    addText(slide, kicker, { left: 64, top: 34, width: 850, height: 28 }, { fontSize: 16, color: teal, bold: true });
  }
  addText(slide, text, { left: 64, top: 58, width: 980, height: 58 }, { fontSize: 34, bold: true });
  slide.shapes.add({
    geometry: "rect",
    position: { left: 64, top: 122, width: 1138, height: 2 },
    fill: teal,
    line: { fill: teal, width: 0 },
  });
}

function footer(slide, n) {
  addText(slide, `Phase 11 final presentation | ${n}`, { left: 64, top: 684, width: 360, height: 22 }, { fontSize: 12, color: "#6B7280" });
}

function bulletList(slide, items, x, y, w, fontSize = 22, gap = 44) {
  items.forEach((item, idx) => {
    slide.shapes.add({
      geometry: "ellipse",
      position: { left: x, top: y + idx * gap + 8, width: 10, height: 10 },
      fill: teal,
      line: { fill: teal, width: 0 },
    });
    addText(slide, item, { left: x + 24, top: y + idx * gap, width: w - 24, height: gap - 4 }, { fontSize, color: navy });
  });
}

function addMetric(slide, label, value, x, y, color = teal) {
  addText(slide, value, { left: x, top: y, width: 190, height: 54 }, { fontSize: 30, bold: true, color });
  addText(slide, label, { left: x, top: y + 50, width: 210, height: 48 }, { fontSize: 15, color: slate });
}

async function addImage(slide, file, alt, position) {
  const blob = await fs.readFile(path.join(FIG_DIR, file));
  slide.images.add({
    blob,
    contentType: "image/png",
    alt,
    fit: "contain",
    position,
  });
}

function addTable(slide, values, position, widths = undefined) {
  const table = slide.tables.add({
    rows: values.length,
    columns: values[0].length,
    left: position.left,
    top: position.top,
    width: position.width,
    height: position.height,
    values,
    ...(widths ? { columnWidths: widths } : {}),
  });
  table.styleOptions = { headerRow: true, bandedRows: true };
  for (let c = 0; c < values[0].length; c += 1) {
    table.getCell(0, c).fill = navy;
    table.getCell(0, c).text.style = { typeface: family, fontSize: 13, color: "#FFFFFF", bold: true };
  }
  for (let r = 1; r < values.length; r += 1) {
    for (let c = 0; c < values[0].length; c += 1) {
      table.getCell(r, c).text.style = { typeface: family, fontSize: 12, color: navy };
    }
  }
  table.borders.assign({ style: "solid", fill: line, width: 1 });
  return table;
}

const presentation = Presentation.create({ slideSize: { width: W, height: H } });

function slideBase() {
  const slide = presentation.slides.add();
  slide.background.fill = "#FFFFFF";
  return slide;
}

// 1
{
  const slide = slideBase();
  slide.background.fill = light;
  addText(slide, "AI-enabled battery thermal forecasting", { left: 74, top: 86, width: 900, height: 76 }, { fontSize: 43, bold: true });
  addText(slide, "FreeTwinEV P45B 1s4p battery module", { left: 78, top: 168, width: 720, height: 42 }, { fontSize: 25, color: teal, bold: true });
  addText(slide, "Forecasting, risk interpretation, cooling guidance, and dashboard", { left: 78, top: 228, width: 760, height: 72 }, { fontSize: 24, color: navy });
  await addImage(slide, "01_runs_matrix_overview.png", "FreeTwinEV run matrix overview", { left: 725, top: 250, width: 460, height: 310 });
  addText(slide, "AAI Mini Project, KJS-CES-02", { left: 78, top: 570, width: 520, height: 32 }, { fontSize: 18, color: slate });
  slide.speakerNotes.textFrame.setText("Source: Phase 0 dataset assessment and persisted FreeTwinEV figures.");
}

// 2
{
  const slide = slideBase();
  title(slide, "Project objective", "Problem framing");
  bulletList(slide, [
    "Predict future maximum cell-surface temperature using measurements available at time t.",
    "Use 60-second and 300-second forecast horizons.",
    "Keep validation for development and ID18 and ID19 as held-out test runs.",
    "Translate frozen forecasts into risk and cooling decision-support outputs.",
  ], 82, 188, 1040, 25, 70);
  footer(slide, 2);
  slide.speakerNotes.textFrame.setText("The project prioritizes leakage prevention, validation discipline, and traceable outputs.");
}

// 3
{
  const slide = slideBase();
  title(slide, "Dataset and provenance", "Physical evidence");
  addMetric(slide, "physical runs", "14", 86, 178);
  addMetric(slide, "observations", "22,247", 320, 178);
  addMetric(slide, "sampling interval", "5 s", 585, 178);
  addMetric(slide, "cell temperature sensors", "12", 810, 178);
  bulletList(slide, [
    "The previous EV battery CSV was rejected after leakage and continuity checks.",
    "FreeTwinEV provides documented physical battery-module experiments.",
    "Current, voltage, SOC, coolant flow, inlet temperature, ambient temperature, and cell sensors support causal timing at t.",
  ], 86, 340, 1010, 22, 56);
  footer(slide, 3);
  slide.speakerNotes.textFrame.setText("Source: reports/freetwinev_dataset_assessment.md and reports/freetwinev_eda.md.");
}

// 4
{
  const slide = slideBase();
  title(slide, "Forecasting formulation", "Target definition");
  addText(slide, "Training target", { left: 86, top: 174, width: 260, height: 32 }, { fontSize: 20, color: teal, bold: true });
  addText(slide, "DeltaT_h(t) = T_cell_max(t + h) - T_cell_max(t)", { left: 86, top: 214, width: 1040, height: 46 }, { fontSize: 29, bold: true });
  addText(slide, "Absolute reconstruction", { left: 86, top: 318, width: 300, height: 32 }, { fontSize: 20, color: teal, bold: true });
  addText(slide, "T_future_pred = T_cell_max(t) + predicted_DeltaT_h(t)", { left: 86, top: 358, width: 1060, height: 46 }, { fontSize: 29, bold: true });
  addTable(slide, [
    ["Horizon", "Steps", "Purpose"],
    ["60s", "12", "Operational response"],
    ["300s", "60", "Strategic thermal envelope"],
  ], { left: 172, top: 486, width: 810, height: 126 }, [180, 160, 470]);
  footer(slide, 4);
  slide.speakerNotes.textFrame.setText("Source: Phase 2 target and feature definition.");
}

// 5
{
  const slide = slideBase();
  title(slide, "Frozen split and leakage controls", "Validation design");
  addTable(slide, [
    ["Split", "Runs"],
    ["Train", "ID01, ID02, ID11, ID12, ID13, ID14, ID15, ID17, ID20, ID21"],
    ["Validation", "ID16, ID22"],
    ["Held-out test", "ID18, ID19"],
  ], { left: 84, top: 168, width: 1110, height: 176 }, [180, 930]);
  bulletList(slide, [
    "Run-level split. No row shuffle.",
    "StandardScaler fitted on training data only.",
    "Rolling and lag features use time t or earlier only.",
    "ID18 and ID19 never inform model selection, threshold derivation, or recommendation rules.",
  ], 104, 410, 1010, 21, 48);
  footer(slide, 5);
  slide.speakerNotes.textFrame.setText("Source: preprocessing_config.json and split_manifest.csv.");
}

// 6
{
  const slide = slideBase();
  title(slide, "Extended 21-feature representation", "Feature engineering");
  bulletList(slide, [
    "Electrical state: current, voltage, and SOC.",
    "Cooling state: liquid flow, inlet temperature, and ambient temperature.",
    "Thermal state: current hotspot, cell spread, and plate mean.",
    "Derived physics terms: current squared, power, rolling load history, thermal velocity, and gradients.",
  ], 84, 175, 620, 22, 62);
  await addImage(slide, "10_feature_relationships.png", "Feature relationship diagnostic", { left: 735, top: 168, width: 410, height: 390 });
  addText(slide, "Ablation outlet-temperature features were not promoted into the selected model.", { left: 84, top: 575, width: 1030, height: 34 }, { fontSize: 19, color: amber, bold: true });
  footer(slide, 6);
  slide.speakerNotes.textFrame.setText("Source: feature_manifest.json and reports/freetwinev_target_feature_definition.md.");
}

// 7
{
  const slide = slideBase();
  title(slide, "Model selection evidence", "Validation only");
  addTable(slide, [
    ["Horizon", "Selected model", "Validation MAE", "Validation RMSE", "Validation R2"],
    ["60s", "Extended Ridge, alpha = 1.0", "0.1598", "0.7228", "0.7890"],
    ["300s", "Extended OLS", "2.2582", "4.5396", "0.5027"],
  ], { left: 75, top: 168, width: 1130, height: 142 }, [130, 390, 200, 210, 200]);
  bulletList(slide, [
    "Tree ensembles were evaluated with grouped cross-validation.",
    "They did not beat the selected Extended linear models on validation.",
    "Selection occurred before held-out test access.",
  ], 110, 380, 930, 24, 58);
  footer(slide, 7);
  slide.speakerNotes.textFrame.setText("Source: Phase 4, Phase 5, and Phase 6 reports.");
}

// 8
{
  const slide = slideBase();
  title(slide, "Held-out test performance", "Final evaluation");
  addTable(slide, [
    ["Horizon", "Model", "Test MAE", "Test RMSE", "Test R2", "Max AE"],
    ["60s", "Extended Ridge", "0.2271", "0.8256", "0.8060", "12.2504"],
    ["300s", "Extended OLS", "1.9302", "4.9649", "0.5836", "43.3812"],
  ], { left: 78, top: 160, width: 1125, height: 142 }, [120, 300, 175, 185, 160, 185]);
  await addImage(slide, "35_phase6_per_run_mae.png", "Phase 6 per-run MAE", { left: 145, top: 345, width: 460, height: 250 });
  await addImage(slide, "32_phase6_residual_distributions.png", "Phase 6 residual distributions", { left: 660, top: 345, width: 460, height: 250 });
  footer(slide, 8);
  slide.speakerNotes.textFrame.setText("Source: results/freetwinev/phase6_results.csv and phase6_results_per_run.csv.");
}

// 9
{
  const slide = slideBase();
  title(slide, "Temporal behavior", "Forecast stability");
  await addImage(slide, "40_phase7_horizon_comparison.png", "Phase 7 horizon comparison", { left: 88, top: 164, width: 500, height: 390 });
  await addImage(slide, "38_phase7_rolling_error.png", "Phase 7 rolling error", { left: 650, top: 164, width: 500, height: 390 });
  addText(slide, "60s is tighter. 300s has useful signal but larger local error spikes.", { left: 100, top: 592, width: 950, height: 36 }, { fontSize: 23, bold: true, color: teal });
  footer(slide, 9);
  slide.speakerNotes.textFrame.setText("Source: Phase 7 temporal forecasting report and figures 38 and 40.");
}

// 10
{
  const slide = slideBase();
  title(slide, "Thermal-risk layer", "Interpretability");
  bulletList(slide, [
    "Critical uses the documented 70 deg C cell-surface cutoff.",
    "P75, P90, and P99 bands come from validation forecasts only.",
    "Secondary signals include predicted DeltaT, recent trend, cell spread, and sustained elevated state.",
    "Held-out ID19 produced 9 critical rows at 60s after rules were frozen.",
  ], 88, 170, 560, 21, 58);
  await addImage(slide, "41_phase8_temperature_risk_overview.png", "Phase 8 risk overview", { left: 690, top: 165, width: 455, height: 360 });
  addText(slide, "Risk categories are decision-support indicators, not certified safety labels.", { left: 88, top: 595, width: 1010, height: 34 }, { fontSize: 20, color: amber, bold: true });
  footer(slide, 10);
  slide.speakerNotes.textFrame.setText("Source: phase8_thresholds.json and reports/freetwinev_phase8_thermal_risk.md.");
}

// 11
{
  const slide = slideBase();
  title(slide, "Cooling recommendation layer", "Decision support");
  addTable(slide, [
    ["Recommendation", "Meaning"],
    ["MAINTAIN", "Normal risk"],
    ["INCREASE_COOLING", "Elevated, active load, low flow"],
    ["HIGH_COOLING_ATTENTION", "High risk or unsupported Elevated state"],
    ["CRITICAL_PROTECTIVE_ACTION", "Critical risk"],
  ], { left: 70, top: 158, width: 635, height: 215 }, [255, 380]);
  addTable(slide, [
    ["Horizon", "Flow Spearman", "Inlet-temp Spearman"],
    ["60s", "-0.4325", "0.4195"],
    ["300s", "-0.4865", "0.5451"],
  ], { left: 745, top: 174, width: 420, height: 136 }, [110, 145, 165]);
  await addImage(slide, "46_phase9_recommendation_counts.png", "Phase 9 recommendation counts", { left: 350, top: 405, width: 540, height: 190 });
  addText(slide, "Associations are observational and confounded by operating regime.", { left: 120, top: 625, width: 990, height: 30 }, { fontSize: 19, color: amber, bold: true });
  footer(slide, 11);
  slide.speakerNotes.textFrame.setText("Source: Phase 9 cooling recommendation report and validation associations.");
}

// 12
{
  const slide = slideBase();
  title(slide, "Dashboard", "Interface layer");
  bulletList(slide, [
    "Streamlit and Plotly dashboard over persisted Phase 6-9 artifacts.",
    "Filters by run, horizon, time, risk category, and recommendation state.",
    "Shows forecast trajectories, risk timelines, recommendation transitions, and final metrics.",
    "Does not train models, tune thresholds, or generate new predictions.",
  ], 86, 178, 1000, 24, 64);
  footer(slide, 12);
  slide.speakerNotes.textFrame.setText("Source: reports/freetwinev_phase10_dashboard.md and phase10_dashboard_audit.json.");
}

// 13
{
  const slide = slideBase();
  title(slide, "Limitations", "Scope");
  bulletList(slide, [
    "Fourteen physical runs from one laboratory module setup.",
    "Only ID18 and ID19 serve as held-out stress-test runs.",
    "The 300s model has materially larger RMSE and tail errors than the 60s model.",
    "P75/P90/P99 risk bands are descriptive validation percentiles.",
    "Cooling guidance is not causal proof and not a certified controller.",
  ], 90, 176, 1030, 24, 64);
  footer(slide, 13);
  slide.speakerNotes.textFrame.setText("Source: Phase 6-10 limitations sections.");
}

// 14
{
  const slide = slideBase();
  slide.background.fill = navy;
  addText(slide, "Final conclusion", { left: 78, top: 78, width: 700, height: 58 }, { fontSize: 42, bold: true, color: "#FFFFFF" });
  addText(slide, "The project delivers a reproducible thermal forecasting and decision-support prototype with protected held-out evaluation.", { left: 82, top: 178, width: 920, height: 92 }, { fontSize: 28, color: "#FFFFFF" });
  addMetric(slide, "held-out 60s MAE", "0.2271 deg C", 100, 330, "#8DE2D1");
  addMetric(slide, "held-out 60s R2", "0.8060", 440, 330, "#8DE2D1");
  addMetric(slide, "held-out 300s MAE", "1.9302 deg C", 710, 330, "#F4C47A");
  addText(slide, "Best framing: validated academic decision support, not a certified EV battery safety controller.", { left: 100, top: 570, width: 1000, height: 44 }, { fontSize: 22, color: "#FFFFFF", bold: true });
  slide.speakerNotes.textFrame.setText("Source: Phase 6 final evaluation and final report limitations.");
}

const requirements = {
  explicitTotalSlideCount: 14,
  requiredNativeTableOwnerSlides: [4, 5, 7, 8, 11],
  requiredNativeChartOwnerSlides: [],
};
const fontPolicy = { basis: "design", families: [family] };
const expectedSlideSizeEmu = "12192000,6858000";

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });
const stagingDir = path.join(TMP_DIR, ".codex-finalizer");
await fs.mkdir(stagingDir, { recursive: true });
const candidatePath = path.join(stagingDir, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const tableOwners = requirements.requiredNativeTableOwnerSlides ?? [];
const result = await finalizePresentation({
  ...requirements,
  workspaceDir: ROOT,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools", "inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools", "inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", expectedSlideSizeEmu,
    "--validate-bullet-geometry",
    "--validate-heading-fit",
    ...tableOwners.flatMap((number) => ["--require-native-table-slide", String(number)]),
  ],
  requiredNativeTableOwnerSlides: tableOwners,
  fontPolicy,
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, `${path.basename(FINAL_PPTX)}.validation.json`),
});

for (let i = 0; i < presentation.slides.length; i += 1) {
  const preview = await presentation.export({ slide: presentation.slides[i], format: "png", scale: 1 });
  await fs.writeFile(path.join(TMP_DIR, `slide-${String(i + 1).padStart(2, "0")}.png`), new Uint8Array(await preview.arrayBuffer()));
}

await fs.writeFile(path.join(TMP_DIR, "finalizer_result.json"), JSON.stringify(result, null, 2));
console.log(JSON.stringify({ finalPath: FINAL_PPTX, slides: presentation.slides.length, status: "built" }, null, 2));
