# K-SHORTS AI — starter website

A responsive dark-premium creator workspace prototype.

## Run locally
Open `index.html` in a browser. No build step is required.

## What works in this starter
- Responsive landing/workspace UI
- YouTube URL validation and paste control
- Local video file selection
- Language, aspect ratio, clip count, duration and caption-style controls
- Interactive demo progress and recent-clip placeholders

## Important: what is not connected yet
This is a front-end MVP prototype, not a production video-processing SaaS. The progress animation is a demo; it does not download YouTube videos, call an AI model, render clips, save projects, authenticate users, or charge subscriptions.

## Production architecture to add
1. **API server**: Node.js + Express or Next.js route handlers.
2. **Video ingestion**: accept user uploads to private object storage (S3/R2). For URL imports, use an authorized provider/API and respect YouTube terms and copyright; do not bypass DRM or access controls.
3. **Processing queue**: Redis + BullMQ for long-running jobs.
4. **Media pipeline**: FFmpeg for cuts, reframing, audio normalization and 9:16 exports.
5. **AI services**: speech-to-text (Whisper or a hosted transcription API), highlight scoring/LLM, translation and subtitle timing.
6. **Database/auth**: PostgreSQL + Prisma, with secure auth and per-user project ownership.
7. **Storage/delivery**: private object storage plus signed download URLs.
8. **Billing**: a payment provider such as Razorpay/Stripe, with server-side webhook verification.
9. **Safety**: file size/type validation, rate limits, malware scanning, retention/deletion policy, consent and copyright checks.

## Suggested API contract
- `POST /api/projects` — create project from authorized source/upload.
- `GET /api/projects/:id` — status and project metadata.
- `GET /api/projects/:id/clips` — generated clip list.
- `GET /api/clips/:id/download` — issue a short-lived signed URL.
- `POST /api/webhooks/billing` — verify billing events.

Never expose API keys in browser JavaScript. Keep provider credentials in server-side environment variables.
