const path = require('path');

module.exports = {
    mode: "development",
    entry: {
        interact: { import: "./js/interact.js", filename: "[name].js" },
        replay: { import: "./js/replay.js", filename: "[name][ext]" },
        yuri: { import: "./js/yuri_demo.js", filename: "[name][ext]" },
    },
    output: {
        path: path.resolve(__dirname, 'static'),
    },
};