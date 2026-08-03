// Patched baseline driver: injects the workspace-patched inventory module
// (QoderCN home + Windows drive-letter slug fixes) into the stock
// asset-baseline and asset-integrity review logic from the plugin cache.
import { collectAssetBaseline } from "file:///C:/Users/user/.qoder-cn/plugins/cache/qoder-bundler/better-harness/scripts/coding-agent-practices/asset-baseline.mjs";
import { reviewAssetIntegrity } from "file:///C:/Users/user/.qoder-cn/plugins/cache/qoder-bundler/better-harness/scripts/coding-agent-practices/asset-integrity.mjs";
import { collectQoderInventory } from "./inventory.mjs";

const workspace = "d:\\rxs-repos\\visionocr-pro";

const baseline = await collectAssetBaseline(
  { provider: "qoder", workspace, language: "zh-CN" },
  { collectPublicInventory: collectQoderInventory },
);
console.log("=== asset-baseline (patched inventory) ===");
console.log(JSON.stringify(baseline, null, 2));

const inventory = await collectQoderInventory({ workspace, includeMemories: true });
const integrity = reviewAssetIntegrity(inventory, { locale: "zh-CN" });
console.log("=== asset-integrity (patched inventory) ===");
console.log(JSON.stringify(integrity, null, 2));
