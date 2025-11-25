window.report = window.report || {};

function formatINR(amount) {
    if (amount === null || amount === undefined) return "0";
    return "₹" + Number(amount).toLocaleString("en-IN", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
}

function colorize(value) {
    if (value > 0) return "text-success";
    if (value < 0) return "text-danger";
    return "text-muted";
}
