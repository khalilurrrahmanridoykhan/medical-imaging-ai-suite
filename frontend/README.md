# Medical Imaging AI Suite, web UI

React (Vite) frontend for the backend in `../backend`. Upload a file, pick a modality, see
the prediction, then ask questions about it. See the main repo README's "Web UI (local)"
section for how to run this together with the backend.

```bash
npm install
npm run dev
```

Set `VITE_API_BASE` in a local `.env` (copy from `.env.example`) if the backend isn't on
the default `http://localhost:8000`.
