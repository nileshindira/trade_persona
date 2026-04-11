
import json
import sys

with open('data/reports/Aninsh17_report.json', 'r') as f:
    data = json.load(f)

for k, v in data.items():
    s = len(json.dumps(v))
    print(f"{k}: {s/1024/1024:.2f} MB")
    if k == "web_data":
        for wk, wv in v.items():
            ws = len(json.dumps(wv))
            print(f"  web_data.{wk}: {ws/1024/1024:.2f} MB")
    if k == "appendix":
        for ak, av in v.items():
            asz = len(json.dumps(av))
            print(f"  appendix.{ak}: {asz/1024/1024:.2f} MB")
