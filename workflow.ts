import { workflow, node, links } from '@n8n-as-code/transformer';

// <workflow-map>
// Workflow : Tactical Match Masterclass
// Nodes   : 7  |  Connections: 6
//
// NODE INDEX
// ──────────────────────────────────────────────────────────────────
// Property name                    Node type (short)         Flags
// ManualTrigger                      manualTrigger
// WebhookTrigger                     webhook
// MatchInputConfig                   code
// TelegramReplayFinder               httpRequest
// ExecuteTacticalPipeline            httpRequest
// FormatMasterclassResult            code
// RespondToWebhook                   respondToWebhook
//
// ROUTING MAP
// ──────────────────────────────────────────────────────────────────
// ManualTrigger
//    → MatchInputConfig
//      → TelegramReplayFinder
//        → ExecuteTacticalPipeline
//          → FormatMasterclassResult
//            → RespondToWebhook
// WebhookTrigger
//    → MatchInputConfig (↩ loop)
// </workflow-map>

// =====================================================================
// METADATA DU WORKFLOW
// =====================================================================

@workflow({
    id: 'COUk1fsntGTZr6Ar',
    name: 'Tactical Match Masterclass',
    active: false,
    settings: { executionOrder: 'v1' },
})
export class TacticalMatchMasterclassWorkflow {
    // =====================================================================
    // CONFIGURATION DES NOEUDS
    // =====================================================================

    @node({
        id: '7b8f9e12-4c56-4d78-9a01-2e3f4a5b6c7d',
        name: 'Manual Trigger',
        type: 'n8n-nodes-base.manualTrigger',
        version: 1,
        position: [100, 200],
    })
    ManualTrigger = {};

    @node({
        id: 'f9c94e67-be78-4636-8b4a-1f5a4e59337d',
        webhookId: '3fa7c442-0a67-4f73-8548-5ee7c0832fca',
        name: 'Webhook Trigger',
        type: 'n8n-nodes-base.webhook',
        version: 2,
        position: [100, 400],
    })
    WebhookTrigger = {
        httpMethod: 'POST',
        path: 'tactical-masterclass',
        responseMode: 'responseNode',
        options: {},
    };

    @node({
        id: '8e7b1521-dab6-4e0a-9f0f-ce43d30bb865',
        name: 'Match Input Config',
        type: 'n8n-nodes-base.code',
        version: 2,
        position: [340, 300],
    })
    MatchInputConfig = {
        jsCode: `const item = $input.first();
const b = (item && item.json && item.json.body) ? item.json.body : (item && item.json ? item.json : {});

// User-provided match parameters (or default test fixture)
const home_team = b.home_team || "Galatasaray";
const away_team = b.away_team || "Barcelona";
const score = b.score || "FT";
const duration = Number(b.duration) || 75; // Default 75s fast render; passes full duration if requested
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
}];`,
    };

    @node({
        id: '502f69b1-507c-4861-bf96-b6fb3e74be88',
        name: 'Telegram Replay Finder',
        type: 'n8n-nodes-base.httpRequest',
        version: 4.2,
        position: [580, 300],
    })
    TelegramReplayFinder = {
        method: 'POST',
        url: 'http://host.docker.internal:5055/search_telegram',
        sendBody: true,
        specifyBody: 'json',
        jsonBody: '={{ JSON.stringify({ home_team: $json.home_team, away_team: $json.away_team }) }}',
        options: {
            timeout: 10000,
            neverError: true,
        },
    };

    @node({
        id: '20501d31-987b-400d-8fb5-9b5cebbab668',
        name: 'Execute Tactical Pipeline',
        type: 'n8n-nodes-base.httpRequest',
        version: 4.2,
        position: [820, 300],
    })
    ExecuteTacticalPipeline = {
        method: 'POST',
        url: 'http://host.docker.internal:5055/analyze',
        sendBody: true,
        specifyBody: 'json',
        jsonBody:
            '={{ JSON.stringify({ home_team: $("Match Input Config").first().json.home_team, away_team: $("Match Input Config").first().json.away_team, duration: $("Match Input Config").first().json.duration, video_path: ($json.video_path || $("Match Input Config").first().json.video_path || "").replace(/\\\\/g, "/"), download_url: ($json.resolved_url || "").replace(/\\\\/g, "/") }) }}',
        options: {
            timeout: 600000,
        },
    };

    @node({
        id: '4b4098a8-b60a-4693-a17e-c107fb1f0c32',
        name: 'Format Masterclass Result',
        type: 'n8n-nodes-base.code',
        version: 2,
        position: [1060, 300],
    })
    FormatMasterclassResult = {
        jsCode: `const data = $input.first().json;

return [{
    json: {
        success: data.status === "success",
        match: data.match,
        video_url: data.video_url,
        video_file: data.video_file,
        job_id: data.job_id,
        events_analyzed: data.events_analyzed,
        resolution: data.resolution,
        message: "Tactical Masterclass 1080p breakdown successfully generated!"
    }
}];`,
    };

    @node({
        id: '63bc70fa-dffb-403d-9f21-ae0b857e999c',
        name: 'Respond To Webhook',
        type: 'n8n-nodes-base.respondToWebhook',
        version: 1.1,
        position: [1300, 300],
    })
    RespondToWebhook = {
        respondWith: 'json',
        responseBody: '={{ $json }}',
        options: {},
    };

    // =====================================================================
    // ROUTAGE ET CONNEXIONS
    // =====================================================================

    @links()
    defineRouting() {
        this.ManualTrigger.out(0).to(this.MatchInputConfig.in(0));
        this.WebhookTrigger.out(0).to(this.MatchInputConfig.in(0));
        this.MatchInputConfig.out(0).to(this.TelegramReplayFinder.in(0));
        this.TelegramReplayFinder.out(0).to(this.ExecuteTacticalPipeline.in(0));
        this.ExecuteTacticalPipeline.out(0).to(this.FormatMasterclassResult.in(0));
        this.FormatMasterclassResult.out(0).to(this.RespondToWebhook.in(0));
    }
}
