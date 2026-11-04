// Reads a JSON list of console strings on stdin and prints their French translations (used by test_i18n.py).
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const source = fs.readFileSync(path.join(__dirname, "../src/glados/webapp/static/glados-i18n.js"), "utf8");
const window = {};
vm.runInNewContext(source, {window, navigator: {language: "fr-FR"}, localStorage: {getItem: () => null, setItem() {}},
  document: {readyState: "loading", addEventListener() {}}});
const strings = JSON.parse(fs.readFileSync(0, "utf8"));
process.stdout.write(JSON.stringify(strings.map(text => window.GladosI18n.translate(text))));
