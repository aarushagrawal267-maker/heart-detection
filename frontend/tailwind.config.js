/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { 950: "#070A0E", 900: "#0A0E13", 850: "#0D1218", 800: "#10151C", 750: "#131922", 700: "#171E28", 600: "#1E2733", 500: "#273241" },
        line: { DEFAULT: "#212A36", soft: "#19202A", strong: "#2D3847" },
        fg: { DEFAULT: "#E4E8EE", muted: "#9AA3B2", dim: "#7E8899" },
        teal: { DEFAULT: "#2BB3A3", soft: "#1B6F66", glow: "#2BB3A31A" },
        ok: "#4CC38A",
        trace: { DEFAULT: "#F2555A", soft: "#F2555A33" },
        cls: { N: "#94A3B8", S: "#E8A33D", V: "#A78BFA", F: "#4EA8DE", Q: "#6B7280" },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "Segoe UI", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "Consolas", "monospace"],
      },
      fontSize: { "2xs": ["0.6875rem", "1rem"] },
      boxShadow: { panel: "0 1px 0 0 rgba(255,255,255,0.03) inset, 0 8px 24px -12px rgba(0,0,0,0.6)" },
    },
  },
  plugins: [],
};
