"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import {
  Send,
  Eraser,
  Flag,
  ListChecks,
  Info,
  ExternalLink,
  BookOpen,
} from "lucide-react";

interface ExampleRow {
  title: string;
  smiles: string;
  fasta_full: string;
  fasta_pocket: string;
}

const EXAMPLES: ExampleRow[] = [
  {
    title: "Aspirin / Kinase pocket",
    smiles: "CC(=O)Oc1ccccc1C(=O)O",
    fasta_full: "MSTNPKPQRKTKRNTNRRPQDVKFPGGPGVPGKKQGPRLHHQQQQQQIPATPNRAPWQAFKQGQPVGGQ",
    fasta_pocket: "KQGPRLHHQQQQQQI",
  },
  {
    title: "Solvent exposed pocket",
    smiles: "CCO",
    fasta_full: "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANLQDLKIDNAGDQVNELSNTTAIKEYAEQLD",
    fasta_pocket: "MKTAYIAKQ",
  },
  {
    title: "Hydrophobic cleft",
    smiles: "C1=CC(=CC=C1O)C(=O)O",
    fasta_full: "GAMGKKVLDSGDGVTHVVPIYDGSYRVTKELGKVTAVEKVGSKKVGDKELGPKAVVIRVRDPKANK",
    fasta_pocket: "VIRVRDPKA",
  },
];

export default function AffiNNityPredictor() {
  const [smiles, setSmiles] = useState("");
  const [fastaFull, setFastaFull] = useState("");
  const [fastaPocket, setFastaPocket] = useState("");
  const [predictedPkD, setPredictedPkD] = useState<number | null>(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [flagged, setFlagged] = useState(false);

  const handleSubmit = async () => {
    if (!smiles.trim() || !fastaFull.trim()) {
      setError("SMILES and full FASTA are required.");
      return;
    }
    setIsLoading(true);
    setError(null);
    setStatus("Computing Morgan fingerprints + ESM-2 embeddings...");
    setFlagged(false);

    try {
      const response = await fetch("/api/hf-predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          smiles,
          fasta_full: fastaFull,
          fasta_pocket: fastaPocket,
        }),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => null);
        throw new Error(err?.error || "Prediction request failed.");
      }

      const data = (await response.json()) as {
        predictedPkD?: number;
        error?: string;
      };

      if (typeof data.predictedPkD !== "number") {
        throw new Error("No predicted pKd returned from model.");
      }

      setPredictedPkD(data.predictedPkD);
      setStatus("Prediction complete via trained GIN model.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed.");
      setStatus(null);
      setPredictedPkD(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setSmiles("");
    setFastaFull("");
    setFastaPocket("");
    setPredictedPkD(0);
    setError(null);
    setStatus(null);
    setFlagged(false);
  };

  const handleExampleSelect = (example: ExampleRow) => {
    setSmiles(example.smiles);
    setFastaFull(example.fasta_full);
    setFastaPocket(example.fasta_pocket);
    setPredictedPkD(0);
    setError(null);
    setStatus("Loaded example; ready to submit.");
    setFlagged(false);
  };

  const handleFlag = () => {
    setFlagged(true);
    setStatus("Flag recorded. Please share details in support if needed.");
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.6 }}
      className="bg-slate-900/60 border border-white/10 rounded-3xl p-8 backdrop-blur-xl shadow-xl"
    >
      <div className="flex items-start justify-between gap-6 flex-wrap">
        <div className="space-y-3">
          <h3 className="text-2xl font-bold text-white">
            AffiNNity — pKd Prediction (SMILES + FASTA)
          </h3>
          <p className="text-slate-300 max-w-2xl">
            Enter a ligand SMILES and two protein FASTA sequences (full protein and
            binding pocket). The app computes Morgan fingerprints + ESM-2 embeddings
            and runs the trained GIN model to return a pKd estimate.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-8">
        <div className="lg:col-span-2 space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <label className="text-sm text-slate-300 font-medium">
                Ligand SMILES
              </label>
              <input
                value={smiles}
                onChange={(e) => setSmiles(e.target.value)}
                placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O"
                className="w-full rounded-xl border border-slate-700 bg-black/30 px-4 py-3 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/40"
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm text-slate-300 font-medium">
                Protein FASTA (full)
              </label>
              <input
                value={fastaFull}
                onChange={(e) => setFastaFull(e.target.value)}
                placeholder="MSTNPKPQRKTK..."
                className="w-full rounded-xl border border-slate-700 bg-black/30 px-4 py-3 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/40"
              />
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm text-slate-300 font-medium">
              Protein FASTA (pocket only)
            </label>
            <textarea
              value={fastaPocket}
              onChange={(e) => setFastaPocket(e.target.value)}
              placeholder="Subset sequence for binding pocket"
              className="w-full rounded-xl border border-slate-700 bg-black/30 px-4 py-3 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/40 h-28"
            />
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={handleClear}
              className="flex items-center gap-2 px-4 py-2 rounded-xl border border-slate-600 bg-slate-800/50 text-slate-100 hover:border-slate-500 transition"
              type="button"
              disabled={isLoading}
            >
              <Eraser className="w-4 h-4" />
              <span>Clear</span>
            </button>
            <button
              onClick={handleSubmit}
              disabled={isLoading}
              className="flex items-center gap-2 px-5 py-2 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold shadow-lg shadow-cyan-500/30 hover:from-cyan-600 hover:to-blue-700 transition disabled:opacity-60 disabled:cursor-not-allowed"
              type="button"
            >
              <Send className={`w-4 h-4 ${isLoading ? "animate-pulse" : ""}`} />
              <span>{isLoading ? "Submitting..." : "Submit"}</span>
            </button>
            <button
              onClick={handleFlag}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl border ${
                flagged
                  ? "border-amber-400 bg-amber-500/10 text-amber-100"
                  : "border-slate-600 bg-slate-800/50 text-slate-200 hover:border-amber-400/70 hover:text-amber-100"
              } transition`}
              type="button"
            >
              <Flag className="w-4 h-4" />
              <span>{flagged ? "Flagged" : "Flag"}</span>
            </button>
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <ListChecks className="w-4 h-4" />
              <span>Predicted pKd shown below. Pocket sequence optional.</span>
            </div>
          </div>

          <div className="rounded-2xl border border-slate-700 bg-black/30 p-4 space-y-3">
            <div className="flex items-center gap-2 text-sm text-slate-300">
              <Info className="w-4 h-4 text-cyan-300" />
              <span>Examples</span>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full text-left text-sm text-slate-200 border-separate border-spacing-y-2">
                <thead className="text-xs uppercase text-slate-400">
                  <tr>
                    <th className="px-3">Ligand SMILES</th>
                    <th className="px-3">Protein FASTA (full)</th>
                    <th className="px-3">Protein FASTA (pocket only)</th>
                    <th className="px-3"> </th>
                  </tr>
                </thead>
                <tbody>
                  {EXAMPLES.map((example) => (
                    <tr
                      key={example.title}
                      className="bg-slate-800/40 hover:bg-slate-800/80 transition rounded-xl"
                    >
                      <td className="px-3 py-2 font-mono text-xs break-all align-top">
                        {example.smiles}
                      </td>
                      <td className="px-3 py-2 text-xs text-slate-300 align-top">
                        {example.fasta_full}
                      </td>
                      <td className="px-3 py-2 text-xs text-slate-300 align-top">
                        {example.fasta_pocket}
                      </td>
                      <td className="px-3 py-2 align-top">
                        <button
                          onClick={() => handleExampleSelect(example)}
                          className="text-cyan-300 text-xs underline hover:text-cyan-200"
                        >
                          Load
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <div className="rounded-2xl border border-cyan-500/30 bg-gradient-to-br from-slate-900 to-slate-800 p-6 text-center">
            <div className="text-xs uppercase tracking-widest text-cyan-200 mb-3">
              Predicted pKd
            </div>
            <div className="text-6xl font-black text-white">
              {predictedPkD ?? 0}
            </div>
            <p className="text-sm text-slate-400 mt-2">
              Output direct from the trained GIN model.
            </p>
          </div>
          {error && (
            <div className="rounded-xl border border-red-400/40 bg-red-500/10 text-red-100 text-sm p-3">
              {error}
            </div>
          )}
          {status && (
            <div className="rounded-xl border border-emerald-400/30 bg-emerald-500/10 text-emerald-100 text-sm p-3">
              {status}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
