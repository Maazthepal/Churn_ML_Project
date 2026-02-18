/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    // Helps Turbopack respect tsconfig paths better
    // Optional: disable Turbopack if it keeps failing (use webpack instead)
    turbopack: false,
  },
};

export default nextConfig;