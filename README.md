---
title: Nexusagent
emoji: 🌍
colorFrom: red
colorTo: green
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

cd /Users/udghoshrao/Downloads/nexusagent/nexus-agent-main
source venv/bin/activate
python3 -m uvicorn app.api.app:app --host 0.0.0.0 --port 7860


cd /Users/udghoshrao/Downloads/nexusagent/nexus-agent-main/frontend
npm run dev
