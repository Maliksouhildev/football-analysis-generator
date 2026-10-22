# Football Analysis Generator

n8n workflow automation for generating football (soccer) match analysis, tactical commentary, and masterclass-style content.

## Workflow: Tactical Match Masterclass

Main file: `workflow.ts` (n8n-as-code TypeScript format)

### What it does
- Accepts a match input via **webhook** or **manual trigger**
- Fetches Telegram replay data for the match
- Runs a **tactical analysis pipeline** (AI-powered)
- Formats and returns masterclass-style commentary output

### Node Map

| Node | Type | Description |
|------|------|-------------|
| ManualTrigger | manualTrigger | Manual test start |
| WebhookTrigger | webhook | HTTP entry point |
| MatchInputConfig | code | Builds match config payload |
| TelegramReplayFinder | httpRequest | Fetches replay clips from Telegram |
| ExecuteTacticalPipeline | httpRequest | Calls AI analysis pipeline |
| FormatMasterclassResult | code | Formats final commentary output |
| RespondToWebhook | respondToWebhook | Returns JSON result |

## Files

| File | Description |
|------|-------------|
| `workflow.ts` | n8n-as-code TypeScript workflow definition |
| `n8n-workflows.d.ts` | TypeScript type declarations for n8n-as-code |
| `package.json` | n8n-as-code package config |
| `tsconfig.json` | TypeScript compiler config |

## Setup

### 1. Install n8n
```bash
npm install -g n8n
```

### 2. Import workflow
Option A — Import JSON directly in n8n UI:
```
n8n > Workflows > Import > paste workflow.json content
```

Option B — Use n8n-as-code CLI:
```bash
npm install
npx n8nac push
```

### 3. Configure credentials
- **Telegram Bot API** — add in n8n credentials panel
- **AI Pipeline endpoint** — set the HTTP endpoint in ExecuteTacticalPipeline node

### 4. Activate
Toggle the workflow ON in n8n and test via the webhook URL or manual trigger.

## Requirements
- n8n v1.0+
- Node.js 18+
- Active Telegram bot token
- AI pipeline backend (e.g. Ollama, OpenAI, custom endpoint)
