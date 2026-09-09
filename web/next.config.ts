import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  typescript: { ignoreBuildErrors: false },
  // Match footage is runtime local data, never an application build dependency.
  outputFileTracingExcludes: { "/*": ["../data/**/*", "../out/**/*", "../.venv/**/*"] },
};

export default nextConfig;
