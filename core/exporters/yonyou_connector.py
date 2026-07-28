"""用友 ERP 连接器骨架 (Phase 3E)

适配: 用友 U8+ / T+ / YonSuite
接口: OpenAPI (REST) 或 XML-RPC, 视版本而定
状态: 骨架, 需填入实际 API 地址和凭据后启用

典型对接流程:
1. 认证: POST /api/auth/token (app_key + app_secret)
2. 推送应收单: POST /api/arap/receivable/save
3. 推送合同: POST /api/contract/save
4. 查询状态: GET /api/arap/receivable/{id}

配置 (config.yaml):
  export:
    yonyou:
      base_url: "http://192.168.1.100:8080"
      app_key: ""
      app_secret: ""
      enabled: false
"""
from __future__ import annotations

from core.exporters.base import BaseExporter, ExportResult


class YonyouExporter(BaseExporter):
    name = "yonyou"

    def export(self, contracts: list[dict],
               receivables: list[dict]) -> ExportResult:
        ycfg = self.export_cfg.get("yonyou", {}) or {}
        base_url = ycfg.get("base_url", "")
        app_key = ycfg.get("app_key", "")
        app_secret = ycfg.get("app_secret", "")

        if not base_url or not app_key:
            return ExportResult(
                self.name, False,
                "用友连接器未配置 (需 base_url + app_key), 跳过")

        # ─── 实际对接时在此实现 ───────────────────────────────
        # token = self._auth(base_url, app_key, app_secret)
        # for c in contracts:
        #     self._push_contract(base_url, token, c)
        # for r in receivables:
        #     self._push_receivable(base_url, token, r)
        # ─────────────────────────────────────────────────────

        return ExportResult(
            self.name, False,
            "用友连接器为骨架状态, 需实现 _auth/_push_contract/_push_receivable")

    # ─── 预留方法 ────────────────────────────────────────────
    def _auth(self, base_url: str, app_key: str, app_secret: str) -> str:
        """获取 access_token。"""
        raise NotImplementedError("用友认证待实现")

    def _push_contract(self, base_url: str, token: str, contract: dict) -> dict:
        """推送合同主数据到用友。"""
        raise NotImplementedError("用友合同推送待实现")

    def _push_receivable(self, base_url: str, token: str, receivable: dict) -> dict:
        """推送应收单到用友。"""
        raise NotImplementedError("用友应收推送待实现")
