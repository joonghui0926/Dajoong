import type { CapacitorConfig } from "@capacitor/cli";

const config: CapacitorConfig = {
  appId: "com.dajoong.plan2bim",
  appName: "Dajoong",
  webDir: "dist",
  server: {
    androidScheme: "https",
  },
};

export default config;
