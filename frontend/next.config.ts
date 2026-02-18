/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    // Helps Turbopack respect tsconfig paths better
    turbopack: {
      resolveAlias: {
        "@/*": "./src/*",
      },
    },
    // Optional: disable Turbopack if it keeps failing (use webpack instead)
    // turbopack: false,
  },
};

export default nextConfig;