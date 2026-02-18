/** @type {import('next').NextConfig} */
const nextConfig = {
  // Optional: enable SWC minifier (modern & fast)
  swcMinify: true,

  // Optional: if you want to disable Turbopack completely (sometimes more stable for shadcn)
  // experimental: {
  //   turbopack: false,
  // },

  // Optional: if you have images from external domains
  images: {
    domains: [], // add if needed, e.g. 'images.unsplash.com'
  },

  // Optional: output standalone for better Vercel compatibility
  // output: 'standalone',
};

export default nextConfig;