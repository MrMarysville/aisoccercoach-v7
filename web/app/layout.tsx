import type { ReactNode } from "react";
import type { Metadata } from "next";
import { Barlow, Barlow_Condensed, IBM_Plex_Mono } from "next/font/google";
import "@/styles/globals.css";
import { AppShell } from "@/components/AppShell";

const barlow = Barlow({ weight: ["400", "500", "600", "700"], subsets: ["latin"], display: "swap", variable: "--font-barlow" });
const condensed = Barlow_Condensed({ weight: ["500", "600", "700"], subsets: ["latin"], display: "swap", variable: "--font-barlow-condensed" });
const mono = IBM_Plex_Mono({ weight: ["400", "500"], subsets: ["latin"], display: "swap", variable: "--font-ibm-plex-mono" });

export const metadata: Metadata = { title: "AI Soccer Coach · Field alignment" };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${barlow.variable} ${condensed.variable} ${mono.variable}`}>
      <body>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
