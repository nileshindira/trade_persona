
import json

try:
    with open("/home/system-4/PycharmProjects/trade_persona/data/reports/Trader_report.json", "r") as f:
        data = json.load(f)

    web_data = data.get("web_data", {})
    charts = web_data.get("charts", {})
    
    # Check PnL Timeline
    pnl = charts.get("pnl_timeline", {})
    print(f"PnL Timeline Keys: {list(pnl.keys())}")
    if "values" in pnl:
        print(f"PnL Values Sample: {pnl['values'][:3]}")

    # Check Instrument Distribution
    inst_dist = charts.get("instrument_distribution", [])
    print(f"Instrument Distribution type: {type(inst_dist)}")
    if isinstance(inst_dist, list) and len(inst_dist) > 0:
        print(f"Instrument Distribution Sample: {inst_dist[0]}")
    else:
        print("Instrument Distribution is empty or not a list")

    # Check Segment Distribution
    seg_dist = charts.get("segment_distribution", [])
    print(f"Segment Distribution type: {type(seg_dist)}")
    if isinstance(seg_dist, list) and len(seg_dist) > 0:
        print(f"Segment Distribution Sample: {seg_dist[0]}")

except Exception as e:
    print(e)
