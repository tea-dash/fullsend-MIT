import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  root: '.',
  base: '/',
  plugins: [react()],
  server: {
    proxy: (() => {
      const target = process.env.VITE_API_TARGET || 'http://127.0.0.1:8000'
      return {
        '/api': {
          target,
          changeOrigin: true,
          secure: false,
        },
        '/data': {
          target,
          changeOrigin: true,
          secure: false,
        },
      }
    })()
  },
  build: {
    outDir: 'dist'
  }
})


