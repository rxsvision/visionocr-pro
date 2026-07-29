"""Phase 3A 功能验证: 迁移幂等 / 方向判定 / 金额勾稽 / 应收余额 / 逾期提醒 / 错误日志"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import date, timedelta
from core.database import get_conn
from core.contract_extractor import extract_contract
from core.payment_store import (
    save_contract, save_receivables, add_collection,
    list_contracts, list_receivables, check_reminders,
    log_error, list_errors,
)

PASS = 0
FAIL = 0

def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


print("=" * 60)
print("1. 数据库迁移幂等性")
print("=" * 60)
import tempfile, sqlite3
tmp = tempfile.mkdtemp()
conn = get_conn(tmp)
# 二次调用不应报错
conn2 = get_conn(tmp)
tables = {r[0] for r in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
check("contracts 表存在", "contracts" in tables)
check("receivables 表存在", "receivables" in tables)
check("collections 表存在", "collections" in tables)
check("signer_map 表存在", "signer_map" in tables)
check("error_log 表存在", "error_log" in tables)
check("risk_alert 表存在", "risk_alert" in tables)
# 检查 contracts 新列
cols = {r[1] for r in conn.execute("PRAGMA table_info(contracts)").fetchall()}
for c in ("contract_no", "our_party", "signer", "direction", "confidence", "reviewed"):
    check(f"contracts.{c} 列", c in cols)
# receivables reminded_overdue 列
rcols = {r[1] for r in conn.execute("PRAGMA table_info(receivables)").fetchall()}
check("receivables.reminded_overdue 列", "reminded_overdue" in rcols)
conn2.close()

print("\n" + "=" * 60)
print("2. 方向判定 + 金额勾稽 (规则路径)")
print("=" * 60)
SAMPLE = """
合同编号: RX-2026-001
合同名称: 设备采购合同
甲方: 苏州锐新视科技有限公司
乙方: 某电子科技有限公司

一、合同总金额: 人民币80万元整。
二、付款方式:
  1. 合同签订后5日内, 乙方向甲方支付预付款人民币30万元。
  2. 设备验收合格后30日内, 乙方向甲方支付尾款人民币50万元。
三、违约责任: 逾期付款每日按未付金额的0.05%支付违约金。
"""
company = {"name": "苏州锐新视科技有限公司", "aliases": ["锐新视"]}
result = extract_contract(SAMPLE, llm=None, company=company)
check("方向=应收", result["direction"] == "receivable",
      f"got {result['direction']}")
check("我方主体识别", result["our_party"] == "苏州锐新视科技有限公司",
      f"got {result['our_party']}")
check("对方主体", "某电子" in result["counterparty"],
      f"got {result['counterparty']}")
check("合同总额=800000", result["total_amount"] == 800000,
      f"got {result['total_amount']}")
check("条款数=2", len(result["payments"]) == 2,
      f"got {len(result['payments'])}")
pay_sum = sum(p["amount"] for p in result["payments"] if p["amount"])
check("条款合计=800000", pay_sum == 800000, f"got {pay_sum}")
check("勾稽通过 valid=True", result["valid"] is True,
      f"got {result['valid']}, warnings={result['warnings']}")
check("置信度>0", result["confidence"] > 0, f"got {result['confidence']}")
check("_method=regex", result.get("_method") == "regex")

print("\n" + "=" * 60)
print("3. 落库 + 应收余额计算")
print("=" * 60)
cid = save_contract(conn, "/fake/contract.pdf", result, SAMPLE)
check("save_contract 返回 id", cid > 0, f"got {cid}")
n = save_receivables(conn, cid, result["payments"])
check("save_receivables 写入 2 条", n == 2, f"got {n}")

# 登记一笔实收
add_collection(conn, cid, 300000, note="预付款到账")
contracts = list_contracts(conn)
check("list_contracts 有 1 条", len(contracts) == 1)
c0 = contracts[0]
check("已收=300000", c0["collected_sum"] == 300000, f"got {c0['collected_sum']}")
check("未收=500000", c0["outstanding"] == 500000, f"got {c0['outstanding']}")

print("\n" + "=" * 60)
print("4. 逾期提醒 (含 overdue 优先级)")
print("=" * 60)
# 手动插入一条已逾期的应收
past = (date.today() - timedelta(days=10)).isoformat()
conn.execute(
    "INSERT INTO receivables (contract_id, due_date, amount, currency, "
    "condition_text, direction, status, source) VALUES (?,?,?,?,?,?,?,?)",
    (cid, past, 99999, "CNY", "测试逾期", "receivable", "pending", "regex"),
)
conn.commit()
fired = check_reminders(conn, today=date.today(), do_notify=False)
overdue_items = [f for f in fired if f["level"] == "逾期"]
check("逾期提醒触发", len(overdue_items) >= 1, f"fired={len(fired)}")
# 再次调用不应重复提醒
fired2 = check_reminders(conn, today=date.today(), do_notify=False)
overdue2 = [f for f in fired2 if f["level"] == "逾期"]
check("逾期不重复提醒", len(overdue2) == 0, f"got {len(overdue2)}")

print("\n" + "=" * 60)
print("5. 错误日志")
print("=" * 60)
eid = log_error(conn, "ocr", "EMPTY_TEXT", "测试错误", file_path="/x.pdf",
                field="text", suggestion="检查扫描件质量")
check("log_error 返回 id", eid > 0)
errs = list_errors(conn)
check("list_errors 有记录", len(errs) >= 1)
check("error_code 正确", errs[0]["error_code"] == "EMPTY_TEXT",
      f"got {errs[0].get('error_code')}")

print("\n" + "=" * 60)
print("6. 金额勾稽失败场景")
print("=" * 60)
BAD = """
合同名称: 测试合同
甲方: 苏州锐新视科技有限公司
乙方: 某公司
合同总额: 人民币100万元。
付款: 签订后支付人民币30万元。
"""
r2 = extract_contract(BAD, llm=None, company=company)
check("勾稽失败 valid=False", r2["valid"] is False, f"got {r2['valid']}")
check("有勾稽警告", any("勾稽" in w for w in r2["warnings"]),
      f"warnings={r2['warnings']}")

conn.close()
print("\n" + "=" * 60)
print(f"结果: {PASS} PASS / {FAIL} FAIL")
print("=" * 60)
sys.exit(1 if FAIL else 0)
