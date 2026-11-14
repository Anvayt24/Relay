# web/

The demo page: plain HTML, CSS and TypeScript, no framework and no runtime dependency.
`tsc` compiles `src/` to `dist/`; Relay serves this folder at `/` (ADR 0028).

    npm install        # once: installs TypeScript
    npm run build      # src/ -> dist/
    npm test           # build, then the pure-logic tests in test/

`format.ts` holds everything that is plain logic (numbers, units, the decision summary) and has
no DOM, so Node can test it. The other modules talk to the page and to `/v1/demo`.
