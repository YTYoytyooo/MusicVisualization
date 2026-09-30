const path = require('node:path');
const fs = require('node:fs');
const codeRoot = path.resolve(__dirname, '..');
const workspace = path.resolve(codeRoot, '../..');
const output = path.join(workspace, 'data/v2.0/validation-output');
exports.validationPath = (name = '') => {
  fs.mkdirSync(output, {recursive:true});
  return path.join(output, name);
};
exports.studioPython = process.env.STUDIO_PYTHON || process.env.MUSIC_PYTHON || path.join(workspace, 'venv/Scripts/python.exe');
exports.codeRoot = codeRoot;
