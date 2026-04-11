import type { Metadata } from "next";
import { Inter, Outfit } from "next/font/google";
import "bootstrap/dist/css/bootstrap.min.css";
import "bootstrap-icons/font/bootstrap-icons.css";
import "./globals.css";
import "../styles/theme.css";
import "../styles/bootstrap_overrides.css";
import "../styles/themes/dark.css"; // Default to dark as it's premium

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const outfit = Outfit({ subsets: ["latin"], variable: "--font-outfit" });

export const metadata: Metadata = {
  title: "Trade Persona Analyzer",
  description: "Advanced Trading Pattern Analysis and Psychometric Profiling",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.variable} ${outfit.variable}`}>
        <div className="container-xl py-4">{children}</div>
      </body>
    </html>
  );
}
