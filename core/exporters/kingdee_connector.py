"""金蝶 ERP 连接器骨架 (Phase 3E)

适配: 金蝶 K/3 WISE / K/3 Cloud (云星空) / KIS
接口: WebAPI (REST JSON), 云星空标准 OpenAPI
状态: 骨架, 需填入实际 API 地址和凭据后启用

典型对接流程 (云星空):
1. 认证: POST /k3cloud/Kingdee.BOS.WebApi.ServicesStub.AuthService.ValidateUser
2. 推送应收单: POST /k3cloud/Kingdee.BOS.WebApi.ServicesStub.DynamicFormService.Save.common.kdsvc
3. 表单ID: FIN_ARAP (应收) / BD_Contract (合同)

配置 (config.yaml):
  export:
    kingdee:
      base_url: "http://192.168.1.200/k3cloud"
      acct_id: ""          # 账套ID
      username: ""
      password: ""
      lcid: 2052           # 语言 (2052=中文)
      enabled: false
"""
from __future__ import annotations

from core.exporters.base import BaseExporter, ExportResult


class KingdeeExporter(BaseExporter):
    name = "kingdee"

    def export(self, contracts: list[dict],
               receivables: list[dict]) -> ExportResult:
        kcfg = self.export_cfg.get("kingdee", {}) or {}
        base_url = kcfg.get("base_url", "")
        username = kcfg.get("username", "")

        if not base_url or not username:
            return ExportResult(
                self.name, False,
                "金蝶连接器未配置 (需 base_url + username), 跳过")

        # ─── 实际对接时在此实现 ───────────────────────────────
        # session = self._auth(base_url, kcfg)
        # for c in contracts:
        #     self._save_bill(base_url, session, "BD_Contract", c)
        # for r in receivables:
        #     self._save_bill(base_url, session, "FIN_ARAP", r)
        # ─────────────────────────────────────────────────────

        return ExportResult(
            self.name, False,
            "金蝶连接器为骨架状态, 需实现 _auth/_save_bill")

    # ─── 预留方法 ────────────────────────────────────────────
    def _auth(self, base_url: str, kcfg: dict) -> dict:
        """金蝶云星空登录, 返回 session cookies。"""
        raise NotImplementedError("金蝶认证待实现")

    def _save_bill(self, base_url: str, session: dict,
                   form_id: str, data: dict) -> dict:
        """调用金蝶 Save 接口推送单据。"""
        raise NotImplementedError("金蝶单据推送待实现")
