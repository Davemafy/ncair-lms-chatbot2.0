const { forward } = require("./_proxy");

module.exports = async function handler(req, res) {
  return forward(req, res, "/api/health", ["GET"]);
};
