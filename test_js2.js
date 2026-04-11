const fs = require('fs');
const html = fs.readFileSync('data/reports/restored_report.html', 'utf8');
const match = html.match(/<script>\s*const report = (\{.*?\});(?=\s*\/\/ P&L Timeline Chart)/s);
if (match) {
    console.log("JSON parsed length:", match[1].length);
    try {
        JSON.parse(match[1]);
        console.log("JSON is perfectly valid");
    } catch (e) {
        console.error(e);
    }
} else {
    console.log("Regex did not match");
}
