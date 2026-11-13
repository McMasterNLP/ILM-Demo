export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          DEFAULT: "#7A003C",
          dark: "#5C002D",
          deep: "#63002F",
          tint: "#F6EBF1",
        },
        ink: {
          DEFAULT: "#1C1917",
          soft: "#57534E",
          muted: "#A8A29E",
        },
        canvas: "#F4F2F0",
        hair: "#E7E5E4",
      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "Segoe UI",
          "sans-serif",
        ],
        arabic: [
          "Noto Kufi Arabic",
          "Segoe UI",
          "Tahoma",
          "ui-sans-serif",
          "sans-serif",
        ],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      boxShadow: {
        card: "0 1px 2px rgba(28,25,23,0.04), 0 1px 3px rgba(28,25,23,0.06)",
        pop: "0 4px 16px rgba(28,25,23,0.10)",
      },
    },
  },
  plugins: [],
};
