import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  experimental: {
    serverActions: {
      // Vinext also applies this multipart preflight before App Router API handlers.
      // Allow the UI/backend's 50 MiB file plus bounded multipart headers/fields.
      // The backend still enforces 50 MiB per file; never disable the limit.
      bodySizeLimit: '51mb',
    },
  },
};

export default nextConfig;
