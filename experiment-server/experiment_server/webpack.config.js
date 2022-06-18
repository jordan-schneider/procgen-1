const path = require('path');

module.exports = {
    mode: "development",
    entry: {
        interact: { import: "./js/interact.js", filename: "[name].js" },
        record: { import: "./js/record.js", filename: "[name].js" },
        replay: { import: "./js/replay.js", filename: "[name].js" },
        yuri: { import: "./js/yuri_demo.js", filename: "[name].js" },

    },
    output: {
        path: path.resolve(__dirname, 'static'),
    },
};