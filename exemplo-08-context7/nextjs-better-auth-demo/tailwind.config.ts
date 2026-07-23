import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      boxShadow: {
        glow: "0 20px 60px rgba(15, 23, 42, 0.18)",
      },
      colors: {
        ink: {
          950: "#06111f",
        },
      },
    },
  },
  plugins: [],
};

export default config;
