const http = require('http');

const API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0NGZkZDNmNS01MzE2LTRkYmEtYTBjZC0yNjBiMTY4NTBkYWEiLCJpc3MiOiJuOG4iLCJhdWQiOiJwdWJsaWMtYXBpIiwiaWF0IjoxNzc5NTc1MzE3fQ.TgZvU0fEwyB8WzuQgY3kkR2flS5RzM5vAI_AcZYaR5Q';
const WORKFLOW_ID = 'COUk1fsntGTZr6Ar';

function request(path, method, data) {
  return new Promise((resolve, reject) => {
    const bodyStr = data ? JSON.stringify(data) : null;
    const req = http.request({
      hostname: '127.0.0.1',
      port: 5678,
      path: path,
      method: method,
      headers: {
        'X-N8N-API-KEY': API_KEY,
        'Content-Type': 'application/json',
        ...(bodyStr ? { 'Content-Length': Buffer.byteLength(bodyStr) } : {})
      }
    }, res => {
      let chunks = '';
      res.on('data', d => chunks += d);
      res.on('end', () => {
        try {
          resolve({ status: res.statusCode, body: JSON.parse(chunks) });
        } catch (e) {
          resolve({ status: res.statusCode, body: chunks });
        }
      });
    });
    req.on('error', reject);
    if (bodyStr) req.write(bodyStr);
    req.end();
  });
}

async function updateMatchInputConfig() {
  const getRes = await request('/api/v1/workflows/' + WORKFLOW_ID, 'GET');
  if (getRes.status !== 200) {
    console.error('Failed to get workflow:', getRes);
    process.exit(1);
  }
  const wf = getRes.body;
  const configNode = wf.nodes.find(n => n.name === 'Match Input Config');
  if (configNode) {
    configNode.parameters.jsCode = `const item = $input.first();
const b = (item && item.json && item.json.body) ? item.json.body : (item && item.json ? item.json : {});

// Dynamic match input parameters (supports webhook payload, manual parameters, or defaults)
const home_team = b.home_team || "Galatasaray";
const away_team = b.away_team || "Barcelona";
const score = b.score || "FT";
const duration = Number(b.duration) || 75; // 75s high-impact masterclass default
const video_path = b.video_path || "";
const download_url = b.download_url || "";

return [{
    json: {
        home_team,
        away_team,
        score,
        duration,
        video_path,
        download_url,
        query: home_team + " vs " + away_team
    }
}];`;
    console.log('Updated Match Input Config code in memory.');
  }

  const execNode = wf.nodes.find(n => n.name === 'Execute Tactical Pipeline');
  if (execNode) {
    execNode.parameters.jsonBody = '={{ JSON.stringify({ home_team: $("Match Input Config").first().json.home_team, away_team: $("Match Input Config").first().json.away_team, duration: $("Match Input Config").first().json.duration, video_path: ($json.video_path || $("Match Input Config").first().json.video_path || "").replace(/\\\\/g, "/"), download_url: ($json.resolved_url || "").replace(/\\\\/g, "/") }) }}';
    console.log('Updated Execute Tactical Pipeline jsonBody in memory.');
  }

  const putRes = await request('/api/v1/workflows/' + WORKFLOW_ID, 'PUT', {
    name: wf.name,
    nodes: wf.nodes,
    connections: wf.connections,
    settings: wf.settings
  });
  console.log('Workflow update response status:', putRes.status);
}

updateMatchInputConfig().catch(console.error);
