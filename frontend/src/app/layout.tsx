import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Protein-Drug Binding Affinity Predictor",
  description:
    "Advanced molecular visualization for protein-drug binding affinity prediction",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body
        className={`${inter.className} bg-gradient-to-br from-slate-900 via-blue-900 to-purple-900 min-h-screen`}
      >
        <div className="molecular-pattern min-h-screen">{children}</div>
      </body>
    </html>
  );
}
