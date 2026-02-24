
with open('/home/system-4/PycharmProjects/trade_persona/src/templates/report.html', 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if 619 <= i <= 630:
            print(f"{i+1}: {repr(line)}")
