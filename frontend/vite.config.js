import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Pre-bundle these up front. Plotly is only imported by the results screen, so
  // without this the dev server discovers it mid-session and force-reloads the
  // page -- which would throw away the column mapping the user just filled in.
  optimizeDeps: {
    include: ['plotly.js-dist-min', 'papaparse'],
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './tests/setup.js',
  },
})
