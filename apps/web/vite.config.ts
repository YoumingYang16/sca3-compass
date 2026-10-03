import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => ({
  base: './',
  define: mode === 'public' ? { 'import.meta.env.VITE_PUBLIC_REVIEW_ONLY': JSON.stringify('true') } : {},
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
}))
