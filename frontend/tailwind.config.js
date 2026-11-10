/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        background: "#0B0F19",
        surface: {
          DEFAULT: "#121A29",
          elevated: "#192438",
          subtle: "#162032",
        },
        border: {
          DEFAULT: "#1E293B",
          subtle: "#24324D",
          highlight: "#334155",
        },
        cyan: {
          DEFAULT: "#38BDF8",
          glow: "rgba(56, 189, 248, 0.15)",
        },
        violet: {
          DEFAULT: "#A78BFA",
          glow: "rgba(167, 139, 250, 0.15)",
        },
        success: {
          DEFAULT: "#34D399",
          glow: "rgba(52, 211, 153, 0.15)",
        },
        warning: {
          DEFAULT: "#FBBF24",
          glow: "rgba(251, 191, 36, 0.15)",
        },
        critical: {
          DEFAULT: "#FB7185",
          glow: "rgba(251, 113, 133, 0.15)",
        },
        text: {
          primary: "#F1F5F9",
          secondary: "#94A3B8",
          muted: "#64748B",
        },
      },
      fontFamily: {
        sans: ["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
        mono: ["JetBrains Mono", "Menlo", "Consolas", "monospace"],
      },
      boxShadow: {
        "cyan-glow": "0 0 15px -3px rgba(56, 189, 248, 0.25)",
        "violet-glow": "0 0 15px -3px rgba(167, 139, 250, 0.25)",
        "surface": "0 4px 20px -2px rgba(0, 0, 0, 0.5)",
      },
    },
  },
  plugins: [],
};

